#pragma once

#include "nano_vllm/kv_cache.hpp"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace nano_vllm {

using SeqId = std::uint64_t;
using SteadyClock = std::chrono::steady_clock;

/// Streaming token event emitted after each forward pass iteration.
struct TokenStreamEvent {
    SeqId seq_id{0};
    std::size_t token_id{0};
    std::string text;
    bool is_finished{false};
    std::optional<std::uint64_t> ttft_us{std::nullopt};
};

/// Thread-safe per-request streaming channel linking the continuous batching loop to an SSE client.
class TokenStreamQueue {
public:
    void push(TokenStreamEvent ev) {
        {
            std::scoped_lock lock(mutex_);
            if (closed_) {
                return;
            }
            if (ev.is_finished) {
                closed_ = true;
            }
            queue_.push_back(std::move(ev));
        }
        cv_.notify_one();
    }

    void close() {
        {
            std::scoped_lock lock(mutex_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    /// Blocks until the next token event arrives or the stream finishes.
    [[nodiscard]] std::optional<TokenStreamEvent> pop_wait() {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || closed_; });
        if (queue_.empty()) {
            return std::nullopt;
        }
        TokenStreamEvent ev = std::move(queue_.front());
        queue_.pop_front();
        return ev;
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<TokenStreamEvent> queue_;
    bool closed_{false};
};

enum class SequenceStatus {
    Waiting,
    Running,
    Finished,
};

/// Per-sequence state managed by the ContinuousScheduler.
struct SequenceState {
    SeqId seq_id{0};
    SequenceStatus status{SequenceStatus::Waiting};
    std::vector<std::size_t> tokens;
    std::size_t prompt_len{0};
    std::size_t num_computed_tokens{0};
    std::size_t max_new_tokens{16};
    std::vector<BlockId> block_table;
    SteadyClock::time_point arrival_time{SteadyClock::now()};
    std::optional<SteadyClock::time_point> first_token_time{std::nullopt};
    std::shared_ptr<TokenStreamQueue> stream_queue{nullptr};

    SequenceState(
        SeqId id,
        std::vector<std::size_t> prompt_tokens,
        std::size_t max_new_toks,
        std::shared_ptr<TokenStreamQueue> stream = nullptr);

    [[nodiscard]] std::size_t total_len() const noexcept { return tokens.size(); }

    [[nodiscard]] std::size_t generated_len() const noexcept {
        return tokens.size() > prompt_len ? tokens.size() - prompt_len : 0;
    }

    [[nodiscard]] std::span<const std::size_t> uncomputed_tokens() const noexcept {
        return std::span<const std::size_t>{tokens}.subspan(num_computed_tokens);
    }

    [[nodiscard]] bool is_finished(std::optional<std::size_t> eos_token_id) const noexcept {
        if (generated_len() >= max_new_tokens) {
            return true;
        }
        if (eos_token_id.has_value() && generated_len() > 0 && !tokens.empty() &&
            tokens.back() == *eos_token_id) {
            return true;
        }
        return false;
    }
};

struct SchedulerConfig {
    std::size_t max_batch_size{32};
    std::size_t max_tokens_per_step{512};
    std::size_t watermark_blocks{4};
    std::optional<std::size_t> eos_token_id{std::nullopt};
};

struct StepSummary {
    std::size_t admitted_sequences{0};
    std::size_t preempted_sequences{0};
    std::size_t retired_sequences{0};
    std::size_t active_batch_size{0};
    std::size_t flattened_tokens_computed{0};
};

/// Iteration-Level Continuous Batching Scheduler with KV OOM Preemption (RECOMPUTE).
class ContinuousScheduler {
public:
    explicit ContinuousScheduler(SchedulerConfig config);

    [[nodiscard]] std::size_t waiting_count() const noexcept { return waiting_.size(); }
    [[nodiscard]] std::size_t running_count() const noexcept { return running_.size(); }
    [[nodiscard]] bool has_pending_work() const noexcept {
        return !waiting_.empty() || !running_.empty();
    }
    [[nodiscard]] std::uint64_t total_preemptions() const noexcept { return total_preemptions_; }
    [[nodiscard]] std::uint64_t total_completed() const noexcept { return total_completed_; }
    [[nodiscard]] std::span<SequenceState> running_mut() noexcept { return running_; }
    [[nodiscard]] std::span<const SequenceState> running() const noexcept { return running_; }

    void add_request(SequenceState seq);

    /// Schedules one forward iteration:
    /// 1. Allocates 1 new physical block for running decode sequences crossing a block boundary,
    ///    preempting the youngest running sequence if the KV cache is out of blocks.
    /// 2. Admits waiting sequences into open batch slots up to token and block watermarks.
    [[nodiscard]] StepSummary schedule_step(PagedKVCache& kv_cache);

    /// Appends sampled tokens, emits SSE events, and immediately retires finished sequences,
    /// returning `(retired_count, newly_recorded_ttfts_us)`.
    std::pair<std::size_t, std::vector<std::uint64_t>> process_step_outputs(
        std::span<const std::size_t> sampled_tokens,
        PagedKVCache& kv_cache,
        std::span<const std::string> vocab);

private:
    [[nodiscard]] static constexpr std::size_t blocks_needed_for_len(
        std::size_t token_len,
        std::size_t block_size) noexcept {
        return (token_len + block_size - 1) / block_size;
    }

    SchedulerConfig config_;
    std::deque<SequenceState> waiting_;
    std::vector<SequenceState> running_;
    std::uint64_t total_preemptions_{0};
    std::uint64_t total_completed_{0};
};

} // namespace nano_vllm
