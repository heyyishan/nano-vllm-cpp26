#pragma once

#include "nano_vllm/kv_cache.hpp"
#include "nano_vllm/scheduler.hpp"
#include "nano_vllm/thread_pool.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace nano_vllm {

struct ModelConfig {
    std::size_t vocab_size{512};
    std::size_t hidden_dim{128};
    std::size_t num_layers{2};
    std::size_t num_heads{4};
    std::size_t head_dim{32};
    std::size_t ffn_dim{256};
};

struct LayerWeights {
    std::vector<float> rms_attn;
    std::vector<float> w_q;
    std::vector<float> w_k;
    std::vector<float> w_v;
    std::vector<float> w_o;
    std::vector<float> rms_ffn;
    std::vector<float> w_up;
    std::vector<float> w_down;
};

/// Miniature Transformer implementing Orca/vLLM Selective Token Flattening and Paged Attention.
class TransformerModel {
public:
    explicit TransformerModel(ModelConfig config = {});

    [[nodiscard]] const ModelConfig& config() const noexcept { return config_; }
    [[nodiscard]] std::span<const std::string> vocab() const noexcept { return vocab_; }

    /// Unified forward pass over a mixed batch of Prefill and Decode sequences.
    [[nodiscard]] std::vector<std::size_t> forward_batch(
        std::span<const SequenceState> sequences,
        PagedKVCache& kv_cache,
        ThreadPool& pool) const;

private:
    ModelConfig config_;
    std::vector<float> token_embedding_;
    std::vector<LayerWeights> layers_;
    std::vector<float> lm_head_;
    std::vector<std::string> vocab_;
};

/// Cache-friendly batched GEMM: `C[B, N] = A[B, K] * W[K, N]`.
void matmul_batched(
    std::span<const float> a,
    std::span<const float> w,
    std::span<float> c,
    std::size_t batch_size,
    std::size_t k_dim,
    std::size_t n_dim,
    ThreadPool& pool);

} // namespace nano_vllm
