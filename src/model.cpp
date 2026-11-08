#include "nano_vllm/model.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

namespace nano_vllm {

namespace {

struct FlattenedTokenMeta {
    std::size_t seq_batch_idx{0};
    std::size_t seq_pos{0};
    std::size_t token_id{0};
};

void rms_norm_batched(
    std::span<const float> input,
    std::span<const float> weight,
    std::span<float> out,
    std::size_t batch_size,
    std::size_t dim) noexcept {
    for (std::size_t b = 0; b < batch_size; ++b) {
        const float* row = input.data() + b * dim;
        float* out_row = out.data() + b * dim;
        float sum_sq = 0.0f;
        for (std::size_t i = 0; i < dim; ++i) {
            sum_sq += row[i] * row[i];
        }
        const float inv_rms = 1.0f / std::sqrt((sum_sq / static_cast<float>(dim)) + 1e-5f);
        for (std::size_t i = 0; i < dim; ++i) {
            out_row[i] = row[i] * inv_rms * weight[i];
        }
    }
}

} // namespace

void matmul_batched(
    std::span<const float> a,
    std::span<const float> w,
    std::span<float> c,
    std::size_t batch_size,
    std::size_t k_dim,
    std::size_t n_dim,
    ThreadPool& pool) {
    pool.parallel_for(batch_size, [&](std::size_t b) {
        float* c_row = c.data() + b * n_dim;
        std::fill_n(c_row, n_dim, 0.0f);
        const float* a_row = a.data() + b * k_dim;
        for (std::size_t k = 0; k < k_dim; ++k) {
            const float a_val = a_row[k];
            const float* w_row = w.data() + k * n_dim;
            for (std::size_t n = 0; n < n_dim; ++n) {
                c_row[n] += a_val * w_row[n];
            }
        }
    });
}

TransformerModel::TransformerModel(ModelConfig config) : config_(config) {
    assert(config_.hidden_dim == config_.num_heads * config_.head_dim);
    const std::size_t d = config_.hidden_dim;
    const std::size_t ffn = config_.ffn_dim;
    const std::size_t v = config_.vocab_size;

    // Deterministic LCG PRNG for reproducible benchmark weights
    std::uint64_t seed = 0x5DEECE66DULL;
    auto next_f32 = [&seed](float scale) noexcept -> float {
        seed = seed * 6364136223846793005ULL + 1ULL;
        const float unit =
            (static_cast<float>(seed >> 33) / static_cast<float>(0xFFFFFFFFULL)) - 0.5f;
        return unit * scale;
    };

    const float emb_scale = 1.0f / std::sqrt(static_cast<float>(d));
    const float ffn_scale = 1.0f / std::sqrt(static_cast<float>(ffn));

    token_embedding_.resize(v * d);
    for (float& val : token_embedding_) {
        val = next_f32(emb_scale);
    }

    layers_.reserve(config_.num_layers);
    for (std::size_t l = 0; l < config_.num_layers; ++l) {
        LayerWeights lw;
        lw.rms_attn.assign(d, 1.0f);
        lw.rms_ffn.assign(d, 1.0f);

        auto init_vec = [&](std::vector<float>& vec, std::size_t count, float sc) {
            vec.resize(count);
            for (float& x : vec) {
                x = next_f32(sc);
            }
        };

        init_vec(lw.w_q, d * d, emb_scale);
        init_vec(lw.w_k, d * d, emb_scale);
        init_vec(lw.w_v, d * d, emb_scale);
        init_vec(lw.w_o, d * d, emb_scale);
        init_vec(lw.w_up, d * ffn, emb_scale);
        init_vec(lw.w_down, ffn * d, ffn_scale);
        layers_.push_back(std::move(lw));
    }

    lm_head_.resize(d * v);
    for (float& val : lm_head_) {
        val = next_f32(emb_scale);
    }

    static constexpr std::array<const char*, 16> kWords = {
        "system", "kernel", "cpp26", "batch",  "token",   "cache",      "paged",   "memory",
        "tensor", "matrix", "stream", "async", "latency", "throughput", "compute", "sched"};

    vocab_.reserve(v);
    for (std::size_t i = 0; i < v; ++i) {
        vocab_.push_back(std::string(" ") + kWords[i % kWords.size()]);
    }
}

std::vector<std::size_t> TransformerModel::forward_batch(
    std::span<const SequenceState> sequences,
    PagedKVCache& kv_cache,
    ThreadPool& pool) const {
    if (sequences.empty()) {
        return {};
    }

    const std::size_t d = config_.hidden_dim;
    const std::size_t ffn = config_.ffn_dim;
    const std::size_t num_heads = config_.num_heads;
    const std::size_t head_dim = config_.head_dim;
    const std::size_t block_size = kv_cache.block_size();

    // 1. Selective Token Flattening across Prefill and Decode sequences
    std::vector<FlattenedTokenMeta> flat_meta;
    flat_meta.reserve(sequences.size() * 4);
    std::vector<std::size_t> last_token_indices;
    last_token_indices.reserve(sequences.size());

    for (std::size_t seq_idx = 0; seq_idx < sequences.size(); ++seq_idx) {
        const auto& seq = sequences[seq_idx];
        const std::size_t start_pos = seq.num_computed_tokens;
        const auto uncomputed = seq.uncomputed_tokens();
        for (std::size_t offset = 0; offset < uncomputed.size(); ++offset) {
            flat_meta.push_back(FlattenedTokenMeta{
                .seq_batch_idx = seq_idx,
                .seq_pos = start_pos + offset,
                .token_id = uncomputed[offset] % config_.vocab_size,
            });
        }
        assert(!flat_meta.empty());
        last_token_indices.push_back(flat_meta.size() - 1);
    }

    const std::size_t n_tokens = flat_meta.size();

    // 2. Token Embedding Lookup: [n_tokens, d]
    std::vector<float> hidden(n_tokens * d, 0.0f);
    for (std::size_t t = 0; t < n_tokens; ++t) {
        const std::size_t emb_offset = flat_meta[t].token_id * d;
        std::copy_n(token_embedding_.data() + emb_offset, d, hidden.data() + t * d);
    }

    std::vector<float> norm_buf(n_tokens * d, 0.0f);
    std::vector<float> q_buf(n_tokens * d, 0.0f);
    std::vector<float> k_buf(n_tokens * d, 0.0f);
    std::vector<float> v_buf(n_tokens * d, 0.0f);
    std::vector<float> attn_out(n_tokens * d, 0.0f);
    std::vector<float> proj_buf(n_tokens * d, 0.0f);
    std::vector<float> ffn_up_buf(n_tokens * ffn, 0.0f);

    // 3. Transformer Layers
    for (std::size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
        const auto& layer = layers_[layer_idx];

        // 3a. Pre-Attention RMSNorm
        rms_norm_batched(hidden, layer.rms_attn, norm_buf, n_tokens, d);

        // 3b. Batched QKV Linear Projections
        matmul_batched(norm_buf, layer.w_q, q_buf, n_tokens, d, d, pool);
        matmul_batched(norm_buf, layer.w_k, k_buf, n_tokens, d, d, pool);
        matmul_batched(norm_buf, layer.w_v, v_buf, n_tokens, d, d, pool);

        // 3c. Write newly projected K and V vectors into PagedKVCache physical blocks
        for (std::size_t t = 0; t < n_tokens; ++t) {
            const auto& meta = flat_meta[t];
            const auto& seq = sequences[meta.seq_batch_idx];
            const std::size_t logical_block = meta.seq_pos / block_size;
            const std::size_t slot_in_block = meta.seq_pos % block_size;
            const BlockId phys_block = seq.block_table[logical_block];

            kv_cache.write_kv(
                phys_block,
                layer_idx,
                slot_in_block,
                std::span<const float>{k_buf.data() + t * d, d},
                std::span<const float>{v_buf.data() + t * d, d});
        }

        // 3d. Causal Paged Attention with FlashAttention Online Softmax
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        pool.parallel_for(n_tokens, [&](std::size_t t_idx) {
            const auto meta = flat_meta[t_idx];
            const auto& seq = sequences[meta.seq_batch_idx];
            const float* q_token = q_buf.data() + t_idx * d;
            float* out_token = attn_out.data() + t_idx * d;

            const std::size_t num_active_positions = meta.seq_pos + 1;
            const std::size_t num_logical_blocks =
                (num_active_positions + block_size - 1) / block_size;

            for (std::size_t h = 0; h < num_heads; ++h) {
                const float* q_head = q_token + h * head_dim;
                float* out_head = out_token + h * head_dim;
                std::fill_n(out_head, head_dim, 0.0f);

                float running_max = -std::numeric_limits<float>::infinity();
                float running_sum = 0.0f;

                // Traverse non-contiguous physical blocks via sequence's BlockTable
                for (std::size_t lb = 0; lb < num_logical_blocks; ++lb) {
                    const BlockId phys_block = seq.block_table[lb];
                    const std::size_t slots_in_block =
                        (lb + 1 == num_logical_blocks)
                            ? (num_active_positions - lb * block_size)
                            : block_size;

                    for (std::size_t slot = 0; slot < slots_in_block; ++slot) {
                        const auto [k_head, v_head] =
                            kv_cache.read_head_kv(phys_block, layer_idx, slot, h);

                        float dot = 0.0f;
                        for (std::size_t i = 0; i < head_dim; ++i) {
                            dot += q_head[i] * k_head[i];
                        }
                        const float score = dot * scale;

                        // Online softmax rescaling (zero heap allocations)
                        const float new_max = std::max(running_max, score);
                        const float exp_prev = std::exp(running_max - new_max);
                        const float exp_curr = std::exp(score - new_max);

                        running_sum = running_sum * exp_prev + exp_curr;
                        for (std::size_t i = 0; i < head_dim; ++i) {
                            out_head[i] = out_head[i] * exp_prev + exp_curr * v_head[i];
                        }
                        running_max = new_max;
                    }
                }

                const float inv_sum = 1.0f / std::max(running_sum, 1e-12f);
                for (std::size_t i = 0; i < head_dim; ++i) {
                    out_head[i] *= inv_sum;
                }
            }
        });

        // 3e. Output Projection + Residual Add
        matmul_batched(attn_out, layer.w_o, proj_buf, n_tokens, d, d, pool);
        for (std::size_t i = 0; i < n_tokens * d; ++i) {
            hidden[i] += proj_buf[i];
        }

        // 3f. FFN Block (RMSNorm -> Up Proj -> SiLU -> Down Proj -> Residual Add)
        rms_norm_batched(hidden, layer.rms_ffn, norm_buf, n_tokens, d);
        matmul_batched(norm_buf, layer.w_up, ffn_up_buf, n_tokens, d, ffn, pool);
        for (float& x : ffn_up_buf) {
            x = x / (1.0f + std::exp(-x));
        }
        matmul_batched(ffn_up_buf, layer.w_down, proj_buf, n_tokens, ffn, d, pool);
        for (std::size_t i = 0; i < n_tokens * d; ++i) {
            hidden[i] += proj_buf[i];
        }
    }

    // 4. Gather only the final token hidden state of each sequence [num_seqs, d]
    const std::size_t num_seqs = sequences.size();
    std::vector<float> last_hidden(num_seqs * d, 0.0f);
    for (std::size_t seq_i = 0; seq_i < num_seqs; ++seq_i) {
        const std::size_t flat_idx = last_token_indices[seq_i];
        std::copy_n(hidden.data() + flat_idx * d, d, last_hidden.data() + seq_i * d);
    }

    // 5. LM Head Logits Projection & Greedy/Repetition-Penalized Sampling
    const std::size_t v = config_.vocab_size;
    std::vector<float> logits(num_seqs * v, 0.0f);
    matmul_batched(last_hidden, lm_head_, logits, num_seqs, d, v, pool);

    std::vector<std::size_t> sampled(num_seqs, 0);
    for (std::size_t seq_i = 0; seq_i < num_seqs; ++seq_i) {
        const float* seq_logits = logits.data() + seq_i * v;
        const std::size_t prev_tok =
            sequences[seq_i].tokens.empty() ? 0 : sequences[seq_i].tokens.back();
        std::size_t best_id = 0;
        float best_val = -std::numeric_limits<float>::infinity();
        for (std::size_t tok_id = 0; tok_id < v; ++tok_id) {
            const float adjusted =
                (tok_id == prev_tok) ? (seq_logits[tok_id] - 1.5f) : seq_logits[tok_id];
            if (adjusted > best_val) {
                best_val = adjusted;
                best_id = tok_id;
            }
        }
        sampled[seq_i] = best_id;
    }

    return sampled;
}

} // namespace nano_vllm
