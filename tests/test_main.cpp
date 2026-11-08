#include "nano_vllm/engine.hpp"
#include "nano_vllm/kv_cache.hpp"
#include "nano_vllm/model.hpp"
#include "nano_vllm/scheduler.hpp"

#include <cassert>
#include <format>
#include <iostream>
#include <thread>
#include <vector>

namespace {

void test_paged_kv_cache_allocation() {
    using namespace nano_vllm;
    const KVCacheConfig cfg{
        .num_blocks = 8,
        .block_size = 4,
        .num_layers = 2,
        .num_heads = 2,
        .head_dim = 8,
    };
    PagedKVCache cache(cfg);
    assert(cache.num_free_blocks() == 8);

    auto blocks_res = cache.allocate_blocks(5);
    assert(blocks_res.has_value());
    assert(cache.num_free_blocks() == 3);

    auto fail_res = cache.allocate_blocks(4);
    assert(!fail_res.has_value());
    assert(fail_res.error() == KVError::OutOfPhysicalBlocks);

    cache.free_blocks(*blocks_res);
    assert(cache.num_free_blocks() == 8);
    std::cout << "[PASS] test_paged_kv_cache_allocation\n";
}

void test_scheduler_kv_oom_preemption() {
    using namespace nano_vllm;
    const KVCacheConfig cfg{
        .num_blocks = 4,
        .block_size = 2,
        .num_layers = 1,
        .num_heads = 2,
        .head_dim = 4,
    };
    PagedKVCache cache(cfg);
    ContinuousScheduler sched(SchedulerConfig{
        .max_batch_size = 4,
        .max_tokens_per_step = 64,
        .watermark_blocks = 0,
        .eos_token_id = std::nullopt,
    });

    // Two sequences with 4 prompt tokens each -> 2 blocks each -> all 4 physical blocks used
    sched.add_request(SequenceState(1, {1, 2, 3, 4}, 4, nullptr));
    sched.add_request(SequenceState(2, {5, 6, 7, 8}, 4, nullptr));

    const StepSummary step1 = sched.schedule_step(cache);
    assert(step1.active_batch_size == 2);
    assert(cache.num_free_blocks() == 0);

    // Emit 1 generated token per sequence -> each now has 5 tokens and needs a 3rd block
    std::vector<std::string> dummy_vocab(16, "tok");
    const std::vector<std::size_t> sampled = {1, 1};
    sched.process_step_outputs(sampled, cache, dummy_vocab);

    // Next step: Seq 1 needs a 3rd block -> Seq 2 is preempted and returned to waiting queue!
    const StepSummary step2 = sched.schedule_step(cache);
    assert(step2.preempted_sequences == 1);
    assert(step2.active_batch_size == 1);
    assert(sched.waiting_count() == 1);
    std::cout << "[PASS] test_scheduler_kv_oom_preemption\n";
}

void test_concurrent_engine_streaming() {
    using namespace nano_vllm;
    const ModelConfig model_cfg{};
    const KVCacheConfig kv_cfg{
        .num_blocks = 256,
        .block_size = 8,
        .num_layers = model_cfg.num_layers,
        .num_heads = model_cfg.num_heads,
        .head_dim = model_cfg.head_dim,
    };
    const SchedulerConfig sched_cfg{
        .max_batch_size = 8,
        .max_tokens_per_step = 256,
        .watermark_blocks = 2,
        .eos_token_id = std::nullopt,
    };

    ContinuousBatchingEngine engine(model_cfg, kv_cfg, sched_cfg);

    constexpr std::size_t kNumClients = 16;
    std::vector<std::jthread> client_threads;
    client_threads.reserve(kNumClients);

    for (std::size_t i = 0; i < kNumClients; ++i) {
        client_threads.emplace_back([&engine, i]() {
            const std::size_t max_tok = (i % 2 == 0) ? 6 : 18;
            auto stream =
                engine.submit_request(std::format("concurrent request {}", i), max_tok);
            std::size_t received = 0;
            bool saw_finished = false;
            while (auto ev = stream->pop_wait()) {
                ++received;
                if (ev->is_finished) {
                    saw_finished = true;
                }
            }
            assert(received == max_tok);
            assert(saw_finished);
        });
    }

    client_threads.clear(); // Join all client jthreads

    const MetricsSnapshot snap = engine.snapshot_metrics();
    assert(snap.total_requests_completed == kNumClients);
    assert(snap.kv_cache_used_blocks == 0);
    std::cout << "[PASS] test_concurrent_engine_streaming\n";
}

} // namespace

int main() {
    test_paged_kv_cache_allocation();
    test_scheduler_kv_oom_preemption();
    test_concurrent_engine_streaming();
    std::cout << "All nano-vllm-cpp26 tests passed successfully!\n";
    return 0;
}
