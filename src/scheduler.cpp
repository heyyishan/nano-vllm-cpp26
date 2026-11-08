#include "nano_vllm/scheduler.hpp"

#include <cassert>
#include <format>

namespace nano_vllm {

SequenceState::SequenceState(
    SeqId id,
    std::vector<std::size_t> prompt_tokens,
    std::size_t max_new_toks,
    std::shared_ptr<TokenStreamQueue> stream)
    : seq_id(id),
      status(SequenceStatus::Waiting),
      tokens(prompt_tokens.empty() ? std::vector<std::size_t>{1} : std::move(prompt_tokens)),
      prompt_len(tokens.size()),
      num_computed_tokens(0),
      max_new_tokens(max_new_toks > 0 ? max_new_toks : 1),
      arrival_time(SteadyClock::now()),
      stream_queue(std::move(stream)) {}

ContinuousScheduler::ContinuousScheduler(SchedulerConfig config) : config_(config) {
    running_.reserve(config_.max_batch_size);
}

void ContinuousScheduler::add_request(SequenceState seq) {
    waiting_.push_back(std::move(seq));
}

StepSummary ContinuousScheduler::schedule_step(PagedKVCache& kv_cache) {
    StepSummary summary{};
    const std::size_t block_size = kv_cache.block_size();

    // Phase 1: Ensure every running sequence has physical KV blocks for its newest token slot.
    std::size_t idx = 0;
    while (idx < running_.size()) {
        const std::size_t needed_blocks =
            blocks_needed_for_len(running_[idx].total_len(), block_size);
        const std::size_t current_blocks = running_[idx].block_table.size();

        if (current_blocks < needed_blocks) {
            assert(needed_blocks - current_blocks == 1);

            // Preempt youngest running sequences if KV cache has 0 free blocks
            while (kv_cache.num_free_blocks() == 0 && running_.size() > idx + 1) {
                SequenceState victim = std::move(running_.back());
                running_.pop_back();

                kv_cache.free_blocks(victim.block_table);
                victim.block_table.clear();
                victim.num_computed_tokens = 0;
                victim.status = SequenceStatus::Waiting;

                waiting_.push_front(std::move(victim));
                ++total_preemptions_;
                ++summary.preempted_sequences;
            }

            if (auto block_res = kv_cache.allocate_block(); block_res.has_value()) {
                running_[idx].block_table.push_back(*block_res);
                ++idx;
            } else {
                // Self-preemption if no other victim could free a block
                SequenceState victim = std::move(running_[idx]);
                running_[idx] = std::move(running_.back());
                running_.pop_back();

                kv_cache.free_blocks(victim.block_table);
                victim.block_table.clear();
                victim.num_computed_tokens = 0;
                victim.status = SequenceStatus::Waiting;

                waiting_.push_front(std::move(victim));
                ++total_preemptions_;
                ++summary.preempted_sequences;
            }
        } else {
            ++idx;
        }
    }

    // Count uncomputed tokens already scheduled in this iteration
    std::size_t scheduled_tokens = 0;
    for (const auto& seq : running_) {
        scheduled_tokens += seq.uncomputed_tokens().size();
    }

    // Phase 2: Continuous Batching Admission from `waiting_` queue
    while (running_.size() < config_.max_batch_size && !waiting_.empty()) {
        const SequenceState& candidate = waiting_.front();
        const std::size_t uncomputed_len = candidate.uncomputed_tokens().size();

        if (scheduled_tokens + uncomputed_len > config_.max_tokens_per_step && !running_.empty()) {
            break;
        }

        const std::size_t blocks_needed =
            blocks_needed_for_len(candidate.total_len(), block_size);
        if (kv_cache.num_free_blocks() < blocks_needed + config_.watermark_blocks) {
            break;
        }

        auto alloc_res = kv_cache.allocate_blocks(blocks_needed);
        if (!alloc_res.has_value()) {
            break;
        }

        SequenceState seq = std::move(waiting_.front());
        waiting_.pop_front();
        seq.block_table = std::move(*alloc_res);
        seq.status = SequenceStatus::Running;

        scheduled_tokens += uncomputed_len;
        running_.push_back(std::move(seq));
        ++summary.admitted_sequences;
    }

    summary.active_batch_size = running_.size();
    summary.flattened_tokens_computed = scheduled_tokens;
    return summary;
}

std::pair<std::size_t, std::vector<std::uint64_t>> ContinuousScheduler::process_step_outputs(
    std::span<const std::size_t> sampled_tokens,
    PagedKVCache& kv_cache,
    std::span<const std::string> vocab) {
    assert(running_.size() == sampled_tokens.size());
    const auto now = SteadyClock::now();
    std::size_t retired_count = 0;
    std::vector<std::uint64_t> newly_recorded_ttfts_us;

    for (std::size_t i = 0; i < running_.size(); ++i) {
        auto& seq = running_[i];
        const std::size_t next_token = sampled_tokens[i];

        seq.num_computed_tokens = seq.tokens.size();
        seq.tokens.push_back(next_token);

        std::optional<std::uint64_t> ttft_us = std::nullopt;
        if (!seq.first_token_time.has_value()) {
            seq.first_token_time = now;
            const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                        now - seq.arrival_time)
                                        .count();
            ttft_us = static_cast<std::uint64_t>(elapsed_us);
            newly_recorded_ttfts_us.push_back(*ttft_us);
        }

        const bool finished = seq.is_finished(config_.eos_token_id);
        if (finished) {
            seq.status = SequenceStatus::Finished;
        }

        if (seq.stream_queue) {
            std::string tok_str = next_token < vocab.size()
                                      ? vocab[next_token]
                                      : std::format("tok_{}", next_token);
            seq.stream_queue->push(TokenStreamEvent{
                .seq_id = seq.seq_id,
                .token_id = next_token,
                .text = std::move(tok_str),
                .is_finished = finished,
                .ttft_us = ttft_us,
            });
        }
    }

    // Immediately retire finished sequences and return their KV blocks to the free pool
    std::size_t i = 0;
    while (i < running_.size()) {
        if (running_[i].status == SequenceStatus::Finished) {
            kv_cache.free_blocks(running_[i].block_table);
            if (i + 1 != running_.size()) {
                running_[i] = std::move(running_.back());
            }
            running_.pop_back();
            ++retired_count;
            ++total_completed_;
        } else {
            ++i;
        }
    }

    return {retired_count, std::move(newly_recorded_ttfts_us)};
}

} // namespace nano_vllm
