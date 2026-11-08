#pragma once

#include <cstddef>
#include <deque>
#include <expected>
#include <span>
#include <utility>
#include <vector>

namespace nano_vllm {

using BlockId = std::size_t;

enum class KVError {
    OutOfPhysicalBlocks,
};

struct KVCacheConfig {
    std::size_t num_blocks{1024};
    std::size_t block_size{16};
    std::size_t num_layers{2};
    std::size_t num_heads{4};
    std::size_t head_dim{32};

    [[nodiscard]] constexpr std::size_t hidden_dim() const noexcept {
        return num_heads * head_dim;
    }
};

/// Paged KV Cache managing a single contiguous pre-allocated memory arena.
///
/// Arena layout for `k_arena_` and `v_arena_`:
/// `[num_blocks, num_layers, block_size, hidden_dim]`
class PagedKVCache {
public:
    explicit PagedKVCache(KVCacheConfig config);

    [[nodiscard]] constexpr const KVCacheConfig& config() const noexcept { return config_; }
    [[nodiscard]] constexpr std::size_t block_size() const noexcept { return config_.block_size; }
    [[nodiscard]] std::size_t num_free_blocks() const noexcept { return free_blocks_.size(); }
    [[nodiscard]] constexpr std::size_t num_total_blocks() const noexcept { return config_.num_blocks; }

    [[nodiscard]] double utilization() const noexcept {
        const std::size_t used = config_.num_blocks - free_blocks_.size();
        return static_cast<double>(used) / static_cast<double>(config_.num_blocks);
    }

    /// Allocates a single physical block in O(1) time.
    [[nodiscard]] std::expected<BlockId, KVError> allocate_block() noexcept;

    /// Allocates `count` physical blocks atomically (all or nothing).
    [[nodiscard]] std::expected<std::vector<BlockId>, KVError> allocate_blocks(std::size_t count);

    /// Frees a span of physical blocks back to the pool in O(k) time.
    void free_blocks(std::span<const BlockId> blocks) noexcept;

    /// Writes Key and Value vectors (`[hidden_dim]`) into a physical block slot.
    void write_kv(
        BlockId block_id,
        std::size_t layer_idx,
        std::size_t slot_idx,
        std::span<const float> k_vec,
        std::span<const float> v_vec) noexcept;

    /// Zero-copy read of Key and Value vectors (`[head_dim]`) for a specific attention head.
    [[nodiscard]] std::pair<std::span<const float>, std::span<const float>> read_head_kv(
        BlockId block_id,
        std::size_t layer_idx,
        std::size_t slot_idx,
        std::size_t head_idx) const noexcept;

private:
    [[nodiscard]] constexpr std::size_t offset(
        BlockId block_id,
        std::size_t layer_idx,
        std::size_t slot_idx) const noexcept {
        return block_id * stride_block_ + layer_idx * stride_layer_ + slot_idx * stride_slot_;
    }

    KVCacheConfig config_;
    std::deque<BlockId> free_blocks_;
    std::vector<bool> allocated_mask_;
    std::vector<float> k_arena_;
    std::vector<float> v_arena_;
    std::size_t stride_block_{0};
    std::size_t stride_layer_{0};
    std::size_t stride_slot_{0};
};

} // namespace nano_vllm
