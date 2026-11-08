#include "nano_vllm/engine.hpp"
#include "nano_vllm/server.hpp"

#include <format>
#include <iostream>

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

    const SchedulerConfig sched_cfg{
        .max_batch_size = 32,
        .max_tokens_per_step = 512,
        .watermark_blocks = 4,
        .eos_token_id = std::nullopt,
    };

    std::cout << std::format(
        "[nano-vllm-cpp26] Starting Continuous Batching Engine | max_batch_size={} | kv_blocks={} (block_size={})\n",
        sched_cfg.max_batch_size,
        kv_cfg.num_blocks,
        kv_cfg.block_size);

    ContinuousBatchingEngine engine(model_cfg, kv_cfg, sched_cfg);

    constexpr std::uint16_t kPort = 8000;
    HttpSseServer server(engine, kPort);

    std::cout << std::format("[nano-vllm-cpp26] Listening on http://0.0.0.0:{}\n", kPort);
    std::cout << "  POST /v1/generate  (Server-Sent Events token streaming)\n";
    std::cout << "  GET  /v1/metrics   (Real-time throughput, TTFT, batch & KV cache telemetry)\n";
    std::cout << "  GET  /health       (Health check)\n";

    server.serve_forever();
    return 0;
}
