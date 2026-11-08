#pragma once

#include "nano_vllm/kv_cache.hpp"
#include "nano_vllm/model.hpp"
#include "nano_vllm/scheduler.hpp"
#include "nano_vllm/thread_pool.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace nano_vllm {

struct NewRequestCommand {
    std::string prompt;
    std::size_t max_tokens{32};
    std::shared_ptr<TokenStreamQueue> stream_queue;
};

struct AtomicEngineMetrics {
    std::atomic<std::uint64_t> total_requests_received{0};
    std::atomic<std::uint64_t> total_requests_completed{0};
    std::atomic<std::uint64_t> total_tokens_generated{0};
    std::atomic<std::uint64_t> total_iterations{0};
    std::atomic<std::uint64_t> total_preemptions{0};
    std::atomic<std::size_t> active_batch_size{0};
    std::atomic<std::size_t> waiting_queue_depth{0};
    std::atomic<std::size_t> kv_free_blocks{0};
    std::atomic<std::size_t> kv_total_blocks{0};
    std::atomic<std::uint64_t> cumulative_ttft_us{0};
    std::atomic<std::uint64_t> ttft_samples{0};
    std::atomic<std::uint64_t> cumulative_step_time_us{0};
};

struct MetricsSnapshot {
    double uptime_secs{0.0};
    std::uint64_t total_requests_received{0};
    std::uint64_t total_requests_completed{0};
    std::uint64_t total_tokens_generated{0};
    double throughput_tokens_per_sec{0.0};
    std::size_t active_batch_size{0};
    std::size_t waiting_queue_depth{0};
    std::size_t kv_cache_used_blocks{0};
    std::size_t kv_cache_total_blocks{0};
    double kv_cache_utilization_pct{0.0};
    double avg_ttft_ms{0.0};
    double avg_iteration_latency_ms{0.0};
    std::uint64_t total_preemptions{0};

    [[nodiscard]] std::string to_json() const;
};

[[nodiscard]] std::vector<std::size_t> tokenize_prompt(
    std::string_view prompt,
    std::size_t vocab_size);

/// Background Continuous-Batching Execution Engine managed by C++26 std::jthread.
class ContinuousBatchingEngine {
public:
    ContinuousBatchingEngine(
        ModelConfig model_cfg,
        KVCacheConfig kv_cfg,
        SchedulerConfig sched_cfg);

    ~ContinuousBatchingEngine();

    ContinuousBatchingEngine(const ContinuousBatchingEngine&) = delete;
    ContinuousBatchingEngine& operator=(const ContinuousBatchingEngine&) = delete;

    /// Submits a request asynchronously and returns a streaming token queue for SSE consumption.
    [[nodiscard]] std::shared_ptr<TokenStreamQueue> submit_request(
        std::string prompt,
        std::size_t max_tokens);

    [[nodiscard]] MetricsSnapshot snapshot_metrics() const noexcept;

private:
    void engine_loop(std::stop_token stoken);

    ModelConfig model_cfg_;
    KVCacheConfig kv_cfg_;
    SchedulerConfig sched_cfg_;

    SteadyClock::time_point start_time_;
    AtomicEngineMetrics metrics_;

    std::mutex ingress_mutex_;
    std::condition_variable ingress_cv_;
    std::deque<NewRequestCommand> ingress_queue_;

    std::jthread worker_thread_;
};

} // namespace nano_vllm
