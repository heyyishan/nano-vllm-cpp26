#include "nano_vllm/kv_cache.hpp"

#include <algorithm>
#include <cassert>

namespace nano_vllm {

PagedKVCache::PagedKVCache(KVCacheConfig config)
    : config_(config),
      allocated_mask_(config.num_blocks, false),
      stride_slot_(config.hidden_dim()),
      stride_layer_(config.block_size * stride_slot_),
      stride_block_(config.num_layers * stride_layer_) {
    const std::size_t total_floats = config_.num_blocks * stride_block_;
    k_arena_.assign(total_floats, 0.0f);
    v_arena_.assign(total_floats, 0.0f);

    for (BlockId id = 0; id < config_.num_blocks; ++id) {
        free_blocks_.push_back(id);
    }
}

std::expected<BlockId, KVError> PagedKVCache::allocate_block() noexcept {
    if (free_blocks_.empty()) {
        return std::unexpected(KVError::OutOfPhysicalBlocks);
    }
    const BlockId id = free_blocks_.front();
    free_blocks_.pop_front();
    allocated_mask_[id] = true;
    return id;
}

std::expected<std::vector<BlockId>, KVError> PagedKVCache::allocate_blocks(std::size_t count) {
    if (free_blocks_.size() < count) {
        return std::unexpected(KVError::OutOfPhysicalBlocks);
    }
    std::vector<BlockId> result;
    result.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const BlockId id = free_blocks_.front();
        free_blocks_.pop_front();
        allocated_mask_[id] = true;
        result.push_back(id);
    }
    return result;
}

void PagedKVCache::free_blocks(std::span<const BlockId> blocks) noexcept {
    for (const BlockId id : blocks) {
        assert(id < config_.num_blocks && "BlockId out of bounds");
        assert(allocated_mask_[id] && "Double-free detected on KV cache physical block");
        allocated_mask_[id] = false;
        free_blocks_.push_back(id);
    }
}

void PagedKVCache::write_kv(
    BlockId block_id,
    std::size_t layer_idx,
    std::size_t slot_idx,
    std::span<const float> k_vec,
    std::span<const float> v_vec) noexcept {
    assert(k_vec.size() == stride_slot_);
    assert(v_vec.size() == stride_slot_);
    const std::size_t base = offset(block_id, layer_idx, slot_idx);
    std::copy_n(k_vec.data(), stride_slot_, k_arena_.data() + base);
    std::copy_n(v_vec.data(), stride_slot_, v_arena_.data() + base);
}

std::pair<std::span<const float>, std::span<const float>> PagedKVCache::read_head_kv(
    BlockId block_id,
    std::size_t layer_idx,
    std::size_t slot_idx,
    std::size_t head_idx) const noexcept {
    const std::size_t head_dim = config_.head_dim;
    const std::size_t base = offset(block_id, layer_idx, slot_idx) + head_idx * head_dim;
    return {
        std::span<const float>{k_arena_.data() + base, head_dim},
        std::span<const float>{v_arena_.data() + base, head_dim},
    };
}

} // namespace nano_vllm
