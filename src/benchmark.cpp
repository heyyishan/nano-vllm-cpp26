#include "nano_vllm/kv_cache.hpp"
#include "nano_vllm/model.hpp"
#include "nano_vllm/scheduler.hpp"
#include "nano_vllm/thread_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <string_view>
#include <vector>

namespace nano_vllm {

namespace {

struct WorkloadRequest {
    SeqId id{0};
    std::vector<std::size_t> prompt_tokens;
    std::size_t max_new_tokens{16};
};

struct BenchmarkResult {
    std::string_view strategy_name;
    std::size_t total_requests{0};
    std::size_t total_output_tokens{0};
    double wall_time_ms{0.0};
    double throughput_tok_per_sec{0.0};
    double avg_batch_utilization_pct{0.0};
    double p50_ttft_ms{0.0};
    double p95_ttft_ms{0.0};
    double p50_e2e_ms{0.0};
    double p95_e2e_ms{0.0};
};

double percentile(const std::vector<double>& sorted, double pct) {
    if (sorted.empty()) {
        return 0.0;
    }
    const auto idx = static_cast<std::size_t>(
        std::llround(static_cast<double>(sorted.size() - 1) * pct));
    return sorted[std::min(idx, sorted.size() - 1)];
}

std::vector<WorkloadRequest> generate_bimodal_workload(
    std::size_t num_requests,
    std::size_t vocab_size) {
    std::vector<WorkloadRequest> workload;
    workload.reserve(num_requests);

    for (std::size_t i = 0; i < num_requests; ++i) {
        const std::size_t prompt_len = 8 + (i % 12);
        std::vector<std::size_t> prompt_tokens(prompt_len);
        for (std::size_t j = 0; j < prompt_len; ++j) {
            prompt_tokens[j] = ((i * 31 + j * 17) % (vocab_size - 1)) + 1;
        }
        // Bimodal output distribution: 80% short (16 tokens), 20% long (128 tokens)
        const std::size_t max_new_tokens = (i % 5 == 0) ? 128 : 16;
        workload.push_back(WorkloadRequest{
            .id = static_cast<SeqId>(i + 1),
            .prompt_tokens = std::move(prompt_tokens),
            .max_new_tokens = max_new_tokens,
        });
    }
    return workload;
}

BenchmarkResult run_static_batching(
    const TransformerModel& model,
    KVCacheConfig kv_cfg,
    std::size_t max_batch_size,
    std::span<const WorkloadRequest> workload,
    ThreadPool& pool) {
    const auto start = SteadyClock::now();
    PagedKVCache kv_cache(kv_cfg);
    const std::size_t block_size = kv_cfg.block_size;

    std::vector<double> ttfts_ms;
    std::vector<double> e2e_ms;
    ttfts_ms.reserve(workload.size());
    e2e_ms.reserve(workload.size());

    std::size_t total_output_tokens = 0;
    std::size_t active_slot_steps = 0;
    std::size_t total_slot_capacity_steps = 0;

    for (std::size_t offset = 0; offset < workload.size(); offset += max_batch_size) {
        const std::size_t chunk_len = std::min(max_batch_size, workload.size() - offset);
        const auto chunk = workload.subspan(offset, chunk_len);

        std::size_t max_target_tokens = 0;
        for (const auto& req : chunk) {
            max_target_tokens = std::max(max_target_tokens, req.max_new_tokens);
        }

        std::vector<SequenceState> batch_seqs;
        batch_seqs.reserve(chunk.size());
        for (const auto& req : chunk) {
            const std::size_t total_needed_len = req.prompt_tokens.size() + max_target_tokens;
            const std::size_t blocks_needed = (total_needed_len + block_size - 1) / block_size;
            SequenceState seq(req.id, req.prompt_tokens, req.max_new_tokens, nullptr);
            seq.block_table = kv_cache.allocate_blocks(blocks_needed).value();
            batch_seqs.push_back(std::move(seq));
        }

        std::vector<bool> completed_flags(batch_seqs.size(), false);

        for (std::size_t step = 0; step < max_target_tokens; ++step) {
            const auto sampled = model.forward_batch(batch_seqs, kv_cache, pool);
            const double now_ms =
                std::chrono::duration<double, std::milli>(SteadyClock::now() - start).count();

            total_slot_capacity_steps += max_batch_size;

            for (std::size_t i = 0; i < batch_seqs.size(); ++i) {
                batch_seqs[i].num_computed_tokens = batch_seqs[i].tokens.size();
                batch_seqs[i].tokens.push_back(sampled[i]);

                if (step == 0) {
                    ttfts_ms.push_back(now_ms);
                }

                if (!completed_flags[i]) {
                    ++active_slot_steps;
                    ++total_output_tokens;
                    if (step + 1 >= batch_seqs[i].max_new_tokens) {
                        completed_flags[i] = true;
                        e2e_ms.push_back(now_ms);
                    }
                }
            }
        }

        for (const auto& seq : batch_seqs) {
            kv_cache.free_blocks(seq.block_table);
        }
    }

    const double wall_time_ms =
        std::chrono::duration<double, std::milli>(SteadyClock::now() - start).count();
    std::ranges::sort(ttfts_ms);
    std::ranges::sort(e2e_ms);

    return BenchmarkResult{
        .strategy_name = "Static Batching (Baseline)",
        .total_requests = workload.size(),
        .total_output_tokens = total_output_tokens,
        .wall_time_ms = wall_time_ms,
        .throughput_tok_per_sec =
            static_cast<double>(total_output_tokens) / (wall_time_ms / 1000.0),
        .avg_batch_utilization_pct =
            (static_cast<double>(active_slot_steps) /
             static_cast<double>(total_slot_capacity_steps)) *
            100.0,
        .p50_ttft_ms = percentile(ttfts_ms, 0.50),
        .p95_ttft_ms = percentile(ttfts_ms, 0.95),
        .p50_e2e_ms = percentile(e2e_ms, 0.50),
        .p95_e2e_ms = percentile(e2e_ms, 0.95),
    };
}

BenchmarkResult run_continuous_batching(
    const TransformerModel& model,
    KVCacheConfig kv_cfg,
    SchedulerConfig sched_cfg,
    std::span<const WorkloadRequest> workload,
    ThreadPool& pool) {
    const auto start = SteadyClock::now();
    PagedKVCache kv_cache(kv_cfg);
    ContinuousScheduler scheduler(sched_cfg);

    for (const auto& req : workload) {
        scheduler.add_request(
            SequenceState(req.id, req.prompt_tokens, req.max_new_tokens, nullptr));
    }

    std::vector<double> ttfts_ms;
    std::vector<double> e2e_ms;
    ttfts_ms.reserve(workload.size());
    e2e_ms.reserve(workload.size());

    std::size_t total_output_tokens = 0;
    std::size_t active_slot_steps = 0;
    std::size_t total_slot_capacity_steps = 0;

    while (scheduler.has_pending_work()) {
        const StepSummary summary = scheduler.schedule_step(kv_cache);
        active_slot_steps += summary.active_batch_size;
        total_slot_capacity_steps += sched_cfg.max_batch_size;

        const auto sampled = model.forward_batch(scheduler.running(), kv_cache, pool);
        total_output_tokens += sampled.size();

        const auto [retired, new_ttfts_us] =
            scheduler.process_step_outputs(sampled, kv_cache, model.vocab());

        const double now_ms =
            std::chrono::duration<double, std::milli>(SteadyClock::now() - start).count();
        for (const std::uint64_t t_us : new_ttfts_us) {
            ttfts_ms.push_back(static_cast<double>(t_us) / 1000.0);
        }
        for (std::size_t r = 0; r < retired; ++r) {
            e2e_ms.push_back(now_ms);
        }
    }

    const double wall_time_ms =
        std::chrono::duration<double, std::milli>(SteadyClock::now() - start).count();
    std::ranges::sort(ttfts_ms);
    std::ranges::sort(e2e_ms);

    return BenchmarkResult{
        .strategy_name = "Continuous Batching (vLLM)",
        .total_requests = workload.size(),
        .total_output_tokens = total_output_tokens,
        .wall_time_ms = wall_time_ms,
        .throughput_tok_per_sec =
            static_cast<double>(total_output_tokens) / (wall_time_ms / 1000.0),
        .avg_batch_utilization_pct =
            (static_cast<double>(active_slot_steps) /
             static_cast<double>(total_slot_capacity_steps)) *
            100.0,
        .p50_ttft_ms = percentile(ttfts_ms, 0.50),
        .p95_ttft_ms = percentile(ttfts_ms, 0.95),
        .p50_e2e_ms = percentile(e2e_ms, 0.50),
        .p95_e2e_ms = percentile(e2e_ms, 0.95),
    };
}

} // namespace

} // namespace nano_vllm

int main() {
    using namespace nano_vllm;

    const ModelConfig model_cfg{
        .vocab_size = 512,
        .hidden_dim = 128,
        .num_layers = 2,
        .num_heads = 4,
        .head_dim = 32,
        .ffn_dim = 256,
    };

    const KVCacheConfig kv_cfg{
        .num_blocks = 1024,
        .block_size = 16,
        .num_layers = model_cfg.num_layers,
        .num_heads = model_cfg.num_heads,
        .head_dim = model_cfg.head_dim,
    };

    constexpr std::size_t kMaxBatchSize = 16;
    const SchedulerConfig sched_cfg{
        .max_batch_size = kMaxBatchSize,
        .max_tokens_per_step = 512,
        .watermark_blocks = 2,
        .eos_token_id = std::nullopt,
    };

    constexpr std::size_t kNumRequests = 96;
    const auto workload = generate_bimodal_workload(kNumRequests, model_cfg.vocab_size);
    const TransformerModel model(model_cfg);
    ThreadPool pool;

    std::cout << "====================================================================================================\n";
    std::cout << std::format(
        "  nano-vllm-cpp26 Benchmark: Continuous Batching vs. Static Batching\n"
        "  Workload: {} concurrent requests (80% short 16-tok, 20% long 128-tok) | Max Batch = {}\n",
        kNumRequests,
        kMaxBatchSize);
    std::cout << "====================================================================================================\n\n";

    const auto static_res = run_static_batching(model, kv_cfg, kMaxBatchSize, workload, pool);
    const auto cont_res = run_continuous_batching(model, kv_cfg, sched_cfg, workload, pool);

    const double speedup = cont_res.throughput_tok_per_sec / static_res.throughput_tok_per_sec;
    const double ttft_reduction =
        static_res.p50_ttft_ms / std::max(1e-6, cont_res.p50_ttft_ms);

    std::cout << std::format(
        "{:<28} | {:>12} | {:>14} | {:>11} | {:>11} | {:>11}\n",
        "Scheduling Strategy",
        "Wall Time",
        "Throughput",
        "Slot Util",
        "p50 TTFT",
        "p50 E2E");
    std::cout << std::string(100, '-') << "\n";

    for (const auto& r : {static_res, cont_res}) {
        std::cout << std::format(
            "{:<28} | {:>9.1f} ms | {:>10.1f} t/s | {:>9.1f} % | {:>8.1f} ms | {:>8.1f} ms\n",
            r.strategy_name,
            r.wall_time_ms,
            r.throughput_tok_per_sec,
            r.avg_batch_utilization_pct,
            r.p50_ttft_ms,
            r.p50_e2e_ms);
    }

    std::cout << std::string(100, '-') << "\n";
    std::cout << std::format(
        "Summary: Continuous Batching achieved {:.2f}x higher token throughput and {:.2f}x lower p50 TTFT!\n",
        speedup,
        ttft_reduction);

    return 0;
}
