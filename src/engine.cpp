#include "nano_vllm/engine.hpp"

#include <algorithm>
#include <format>

namespace nano_vllm {

std::string MetricsSnapshot::to_json() const {
    return std::format(
        "{{\n"
        "  \"uptime_secs\": {:.3f},\n"
        "  \"total_requests_received\": {},\n"
        "  \"total_requests_completed\": {},\n"
        "  \"total_tokens_generated\": {},\n"
        "  \"throughput_tokens_per_sec\": {:.2f},\n"
        "  \"active_batch_size\": {},\n"
        "  \"waiting_queue_depth\": {},\n"
        "  \"kv_cache_used_blocks\": {},\n"
        "  \"kv_cache_total_blocks\": {},\n"
        "  \"kv_cache_utilization_pct\": {:.2f},\n"
        "  \"avg_ttft_ms\": {:.3f},\n"
        "  \"avg_iteration_latency_ms\": {:.3f},\n"
        "  \"total_preemptions\": {}\n"
        "}}",
        uptime_secs,
        total_requests_received,
        total_requests_completed,
        total_tokens_generated,
        throughput_tokens_per_sec,
        active_batch_size,
        waiting_queue_depth,
        kv_cache_used_blocks,
        kv_cache_total_blocks,
        kv_cache_utilization_pct,
        avg_ttft_ms,
        avg_iteration_latency_ms,
        total_preemptions);
}

std::vector<std::size_t> tokenize_prompt(std::string_view prompt, std::size_t vocab_size) {
    std::vector<std::size_t> tokens;
    std::size_t pos = 0;
    while (pos < prompt.size()) {
        while (pos < prompt.size() && (prompt[pos] == ' ' || prompt[pos] == '\t' ||
                                       prompt[pos] == '\n' || prompt[pos] == '\r')) {
            ++pos;
        }
        if (pos >= prompt.size()) {
            break;
        }
        const std::size_t start = pos;
        while (pos < prompt.size() && prompt[pos] != ' ' && prompt[pos] != '\t' &&
               prompt[pos] != '\n' && prompt[pos] != '\r') {
            ++pos;
        }
        std::size_t hash = 5381;
        for (std::size_t i = start; i < pos; ++i) {
            hash = ((hash << 5) + hash) + static_cast<unsigned char>(prompt[i]);
        }
        tokens.push_back((hash % (vocab_size - 1)) + 1);
    }
    if (tokens.empty()) {
        tokens.push_back(1);
    }
    return tokens;
}

ContinuousBatchingEngine::ContinuousBatchingEngine(
    ModelConfig model_cfg,
    KVCacheConfig kv_cfg,
    SchedulerConfig sched_cfg)
    : model_cfg_(model_cfg),
      kv_cfg_(kv_cfg),
      sched_cfg_(sched_cfg),
      start_time_(SteadyClock::now()) {
    metrics_.kv_total_blocks.store(kv_cfg_.num_blocks, std::memory_order::relaxed);
    metrics_.kv_free_blocks.store(kv_cfg_.num_blocks, std::memory_order::relaxed);

    worker_thread_ = std::jthread([this](std::stop_token stoken) {
        engine_loop(stoken);
    });
}

ContinuousBatchingEngine::~ContinuousBatchingEngine() {
    worker_thread_.request_stop();
    ingress_cv_.notify_all();
}

std::shared_ptr<TokenStreamQueue> ContinuousBatchingEngine::submit_request(
    std::string prompt,
    std::size_t max_tokens) {
    auto stream = std::make_shared<TokenStreamQueue>();
    metrics_.total_requests_received.fetch_add(1, std::memory_order::relaxed);
    {
        std::scoped_lock lock(ingress_mutex_);
        ingress_queue_.push_back(NewRequestCommand{
            .prompt = std::move(prompt),
            .max_tokens = std::max<std::size_t>(1, max_tokens),
            .stream_queue = stream,
        });
    }
    ingress_cv_.notify_one();
    return stream;
}

MetricsSnapshot ContinuousBatchingEngine::snapshot_metrics() const noexcept {
    const double uptime_secs = std::max(
        1e-6,
        std::chrono::duration<double>(SteadyClock::now() - start_time_).count());
    const std::uint64_t total_tokens =
        metrics_.total_tokens_generated.load(std::memory_order::relaxed);
    const std::size_t total_blocks =
        metrics_.kv_total_blocks.load(std::memory_order::relaxed);
    const std::size_t free_blocks =
        metrics_.kv_free_blocks.load(std::memory_order::relaxed);
    const std::size_t used_blocks =
        total_blocks > free_blocks ? total_blocks - free_blocks : 0;

    const std::uint64_t ttft_samples =
        metrics_.ttft_samples.load(std::memory_order::relaxed);
    const double avg_ttft_ms =
        ttft_samples > 0
            ? (static_cast<double>(
                   metrics_.cumulative_ttft_us.load(std::memory_order::relaxed)) /
               static_cast<double>(ttft_samples)) /
                  1000.0
            : 0.0;

    const std::uint64_t iterations =
        metrics_.total_iterations.load(std::memory_order::relaxed);
    const double avg_iteration_latency_ms =
        iterations > 0
            ? (static_cast<double>(
                   metrics_.cumulative_step_time_us.load(std::memory_order::relaxed)) /
               static_cast<double>(iterations)) /
                  1000.0
            : 0.0;

    return MetricsSnapshot{
        .uptime_secs = uptime_secs,
        .total_requests_received =
            metrics_.total_requests_received.load(std::memory_order::relaxed),
        .total_requests_completed =
            metrics_.total_requests_completed.load(std::memory_order::relaxed),
        .total_tokens_generated = total_tokens,
        .throughput_tokens_per_sec = static_cast<double>(total_tokens) / uptime_secs,
        .active_batch_size = metrics_.active_batch_size.load(std::memory_order::relaxed),
        .waiting_queue_depth =
            metrics_.waiting_queue_depth.load(std::memory_order::relaxed),
        .kv_cache_used_blocks = used_blocks,
        .kv_cache_total_blocks = total_blocks,
        .kv_cache_utilization_pct =
            total_blocks > 0
                ? (static_cast<double>(used_blocks) / static_cast<double>(total_blocks)) *
                      100.0
                : 0.0,
        .avg_ttft_ms = avg_ttft_ms,
        .avg_iteration_latency_ms = avg_iteration_latency_ms,
        .total_preemptions = metrics_.total_preemptions.load(std::memory_order::relaxed),
    };
}

void ContinuousBatchingEngine::engine_loop(std::stop_token stoken) {
    TransformerModel model(model_cfg_);
    PagedKVCache kv_cache(kv_cfg_);
    ContinuousScheduler scheduler(sched_cfg_);
    ThreadPool compute_pool;
    SeqId next_seq_id = 1;

    while (!stoken.stop_requested()) {
        // Wait on condition variable when completely idle
        if (!scheduler.has_pending_work()) {
            std::unique_lock lock(ingress_mutex_);
            ingress_cv_.wait(lock, [this, &stoken] {
                return stoken.stop_requested() || !ingress_queue_.empty();
            });
            if (stoken.stop_requested()) {
                break;
            }
        }

        // Non-blocking drain of all newly arrived requests before scheduling this iteration
        {
            std::scoped_lock lock(ingress_mutex_);
            while (!ingress_queue_.empty()) {
                NewRequestCommand cmd = std::move(ingress_queue_.front());
                ingress_queue_.pop_front();
                auto prompt_tokens = tokenize_prompt(cmd.prompt, model_cfg_.vocab_size);
                scheduler.add_request(SequenceState{
                    next_seq_id++,
                    std::move(prompt_tokens),
                    cmd.max_tokens,
                    std::move(cmd.stream_queue),
                });
            }
        }

        if (!scheduler.has_pending_work()) {
            continue;
        }

        const auto step_start = SteadyClock::now();

        // 1. Iteration-Level Continuous Scheduling & Block Allocation
        const StepSummary summary = scheduler.schedule_step(kv_cache);
        if (summary.active_batch_size == 0) {
            std::this_thread::yield();
            continue;
        }

        // 2. Unified Batched Forward Pass (Prefill + Decode simultaneously)
        const auto sampled_tokens =
            model.forward_batch(scheduler.running(), kv_cache, compute_pool);
        const auto generated_count = static_cast<std::uint64_t>(sampled_tokens.size());

        // 3. Emit SSE Tokens & Immediately Retire Finished Sequences
        const auto [retired, ttfts] =
            scheduler.process_step_outputs(sampled_tokens, kv_cache, model.vocab());
        (void)retired;

        const auto step_us = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                SteadyClock::now() - step_start)
                .count());

        // 4. Update Lock-Free Atomic Telemetry
        metrics_.total_tokens_generated.fetch_add(generated_count, std::memory_order::relaxed);
        metrics_.total_iterations.fetch_add(1, std::memory_order::relaxed);
        metrics_.cumulative_step_time_us.fetch_add(step_us, std::memory_order::relaxed);
        metrics_.total_requests_completed.store(
            scheduler.total_completed(), std::memory_order::relaxed);
        metrics_.total_preemptions.store(
            scheduler.total_preemptions(), std::memory_order::relaxed);
        metrics_.active_batch_size.store(
            scheduler.running_count(), std::memory_order::relaxed);
        metrics_.waiting_queue_depth.store(
            scheduler.waiting_count(), std::memory_order::relaxed);
        metrics_.kv_free_blocks.store(
            kv_cache.num_free_blocks(), std::memory_order::relaxed);

        for (const std::uint64_t ttft : ttfts) {
            metrics_.cumulative_ttft_us.fetch_add(ttft, std::memory_order::relaxed);
            metrics_.ttft_samples.fetch_add(1, std::memory_order::relaxed);
        }
    }
}

} // namespace nano_vllm
