#pragma once

/**
 * @file cuda_mempool.hpp
 * @brief High-performance CUDA memory pool with caching and stream-ordered allocation
 *
 * This memory pool provides:
 * - Sub-millisecond allocation latency (vs ~1ms for cudaMalloc)
 * - Memory reuse to reduce fragmentation
 * - Stream-ordered semantics for async safety
 * - Size-class binning for efficient matching
 *
 * Usage:
 * @code
 *   CudaMemPool pool(1ULL << 30);  // 1GB pool
 *   void* ptr = pool.allocate(1 << 20);  // 1MB allocation
 *   pool.deallocate(ptr);
 * @endcode
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <algorithm>

namespace mempool {

/**
 * @brief CUDA error checking macro
 */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            throw CudaError(cudaGetErrorString(err), __FILE__, __LINE__);     \
        }                                                                      \
    } while (0)

/**
 * @brief Exception for CUDA errors
 */
class CudaError : public std::runtime_error {
public:
    CudaError(const char* msg, const char* file, int line)
        : std::runtime_error(std::string(msg) + " at " + file + ":" + std::to_string(line)) {}
};

/**
 * @brief Configuration for the memory pool
 */
struct PoolConfig {
    /// Maximum memory to allocate from CUDA
    size_t max_pool_size = 1ULL << 32;  // 4GB default

    /// Minimum allocation size (smaller requests rounded up)
    size_t min_block_size = 512;

    /// Maximum cached blocks per size class
    size_t max_cached_blocks = 64;

    /// Growth factor for size classes (powers of 2)
    size_t num_size_classes = 32;

    /// Enable stream-ordered allocation
    bool stream_ordered = true;

    /// Device ID
    int device_id = 0;
};

/**
 * @brief Metadata for an allocated block
 */
struct Block {
    void* ptr;           ///< Device pointer
    size_t size;         ///< Actual allocation size
    size_t requested;    ///< User-requested size
    cudaStream_t stream; ///< Associated stream
    bool in_use;         ///< Currently allocated
};

/**
 * @brief Size class for binning allocations
 *
 * Each size class holds free blocks of a specific size range.
 * Uses power-of-2 sizing for efficient matching.
 */
class SizeClass {
public:
    explicit SizeClass(size_t size) : size_(size) {}

    /// Get a free block, or nullptr if none available
    Block* get() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_blocks_.empty()) {
            return nullptr;
        }
        Block* block = free_blocks_.back();
        free_blocks_.pop_back();
        block->in_use = true;
        return block;
    }

    /// Return a block to the free list
    void put(Block* block) {
        std::lock_guard<std::mutex> lock(mutex_);
        block->in_use = false;
        free_blocks_.push_back(block);
    }

    /// Number of free blocks
    size_t free_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return free_blocks_.size();
    }

    size_t size() const { return size_; }

private:
    size_t size_;
    std::vector<Block*> free_blocks_;
    mutable std::mutex mutex_;
};

/**
 * @brief High-performance CUDA memory pool
 *
 * Maintains a pool of pre-allocated GPU memory organized by size classes.
 * Allocations are satisfied from the pool when possible, falling back
 * to cudaMalloc for new blocks.
 *
 * Thread-safe for concurrent allocations from multiple CPU threads.
 */
class CudaMemPool {
public:
    /**
     * @brief Construct a memory pool
     * @param config Pool configuration
     */
    explicit CudaMemPool(const PoolConfig& config = PoolConfig())
        : config_(config), total_allocated_(0), peak_allocated_(0) {

        CUDA_CHECK(cudaSetDevice(config_.device_id));

        // Initialize size classes (powers of 2)
        size_t size = config_.min_block_size;
        for (size_t i = 0; i < config_.num_size_classes; ++i) {
            size_classes_.push_back(std::make_unique<SizeClass>(size));
            size *= 2;
        }
    }

    /**
     * @brief Destructor - frees all GPU memory
     */
    ~CudaMemPool() {
        std::lock_guard<std::mutex> lock(global_mutex_);
        for (auto& [ptr, block] : all_blocks_) {
            cudaFree(block.ptr);
        }
    }

    // Non-copyable
    CudaMemPool(const CudaMemPool&) = delete;
    CudaMemPool& operator=(const CudaMemPool&) = delete;

    /**
     * @brief Allocate GPU memory
     * @param size Requested size in bytes
     * @param stream CUDA stream for stream-ordered allocation
     * @return Device pointer
     * @throws CudaError if allocation fails
     */
    void* allocate(size_t size, cudaStream_t stream = nullptr) {
        if (size == 0) {
            return nullptr;
        }

        // Round up to size class
        size_t alloc_size = round_up_to_size_class(size);
        SizeClass* sc = get_size_class(alloc_size);

        // Try to get from free list
        Block* block = sc ? sc->get() : nullptr;

        if (block) {
            block->requested = size;
            block->stream = stream;
            return block->ptr;
        }

        // Need new allocation
        return allocate_new_block(size, alloc_size, stream);
    }

    /**
     * @brief Deallocate GPU memory
     * @param ptr Device pointer to free
     *
     * Returns memory to the pool for reuse rather than freeing.
     */
    void deallocate(void* ptr) {
        if (!ptr) {
            return;
        }

        std::lock_guard<std::mutex> lock(global_mutex_);

        auto it = all_blocks_.find(ptr);
        if (it == all_blocks_.end()) {
            throw std::runtime_error("Attempt to free unknown pointer");
        }

        Block& block = it->second;
        block.in_use = false;

        // Return to appropriate size class
        SizeClass* sc = get_size_class(block.size);
        if (sc) {
            sc->put(&block);
        }
    }

    /**
     * @brief Allocate with stream-ordered semantics
     * @param size Requested size
     * @param stream CUDA stream
     * @return Device pointer
     *
     * The allocation is guaranteed to be safe to use after all prior
     * operations on the stream complete.
     */
    void* allocate_async(size_t size, cudaStream_t stream) {
        void* ptr = allocate(size, stream);

        if (config_.stream_ordered && stream) {
            // Record event to track when allocation is safe
            cudaEvent_t event;
            CUDA_CHECK(cudaEventCreate(&event));
            CUDA_CHECK(cudaEventRecord(event, stream));

            std::lock_guard<std::mutex> lock(global_mutex_);
            auto it = all_blocks_.find(ptr);
            if (it != all_blocks_.end()) {
                pending_events_[ptr] = event;
            }
        }

        return ptr;
    }

    /**
     * @brief Deallocate with stream-ordered semantics
     * @param ptr Device pointer
     * @param stream CUDA stream
     *
     * Memory becomes available for reuse after operations on stream complete.
     */
    void deallocate_async(void* ptr, cudaStream_t stream) {
        if (!ptr) return;

        if (config_.stream_ordered && stream) {
            cudaEvent_t event;
            CUDA_CHECK(cudaEventCreate(&event));
            CUDA_CHECK(cudaEventRecord(event, stream));

            // Schedule deferred deallocation
            std::lock_guard<std::mutex> lock(global_mutex_);
            deferred_frees_.push_back({ptr, event});
        } else {
            deallocate(ptr);
        }
    }

    /**
     * @brief Process deferred deallocations
     *
     * Call periodically to return memory from completed async operations.
     */
    void process_deferred() {
        std::lock_guard<std::mutex> lock(global_mutex_);

        auto it = deferred_frees_.begin();
        while (it != deferred_frees_.end()) {
            cudaError_t status = cudaEventQuery(it->event);
            if (status == cudaSuccess) {
                // Event completed, safe to free
                cudaEventDestroy(it->event);

                auto block_it = all_blocks_.find(it->ptr);
                if (block_it != all_blocks_.end()) {
                    Block& block = block_it->second;
                    block.in_use = false;
                    SizeClass* sc = get_size_class(block.size);
                    if (sc) {
                        sc->put(&block);
                    }
                }

                it = deferred_frees_.erase(it);
            } else if (status == cudaErrorNotReady) {
                ++it;
            } else {
                CUDA_CHECK(status);
            }
        }
    }

    /**
     * @brief Release all unused cached memory back to CUDA
     */
    void trim() {
        std::lock_guard<std::mutex> lock(global_mutex_);

        for (auto it = all_blocks_.begin(); it != all_blocks_.end();) {
            if (!it->second.in_use) {
                CUDA_CHECK(cudaFree(it->second.ptr));
                total_allocated_ -= it->second.size;
                it = all_blocks_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Statistics
    size_t total_allocated() const { return total_allocated_; }
    size_t peak_allocated() const { return peak_allocated_; }
    size_t num_allocations() const { return all_blocks_.size(); }

    /**
     * @brief Get cache statistics per size class
     */
    std::vector<std::pair<size_t, size_t>> cache_stats() const {
        std::vector<std::pair<size_t, size_t>> stats;
        for (const auto& sc : size_classes_) {
            stats.push_back({sc->size(), sc->free_count()});
        }
        return stats;
    }

private:
    PoolConfig config_;
    std::vector<std::unique_ptr<SizeClass>> size_classes_;
    std::unordered_map<void*, Block> all_blocks_;
    std::unordered_map<void*, cudaEvent_t> pending_events_;

    struct DeferredFree {
        void* ptr;
        cudaEvent_t event;
    };
    std::vector<DeferredFree> deferred_frees_;

    std::mutex global_mutex_;
    size_t total_allocated_;
    size_t peak_allocated_;

    /// Round size up to nearest size class
    size_t round_up_to_size_class(size_t size) const {
        if (size <= config_.min_block_size) {
            return config_.min_block_size;
        }
        // Round up to next power of 2
        size_t power = 1;
        while (power < size) {
            power *= 2;
        }
        return power;
    }

    /// Get size class for a given size, or nullptr if too large
    SizeClass* get_size_class(size_t size) {
        if (size < config_.min_block_size) {
            size = config_.min_block_size;
        }

        // Find the index: log2(size / min_block_size)
        size_t index = 0;
        size_t s = config_.min_block_size;
        while (s < size && index < size_classes_.size()) {
            s *= 2;
            ++index;
        }

        if (index < size_classes_.size()) {
            return size_classes_[index].get();
        }
        return nullptr;
    }

    /// Allocate a new block from CUDA
    void* allocate_new_block(size_t requested, size_t alloc_size, cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(global_mutex_);

        // Check pool limit
        if (total_allocated_ + alloc_size > config_.max_pool_size) {
            // Try to trim unused memory first
            for (auto it = all_blocks_.begin(); it != all_blocks_.end();) {
                if (!it->second.in_use) {
                    CUDA_CHECK(cudaFree(it->second.ptr));
                    total_allocated_ -= it->second.size;
                    it = all_blocks_.erase(it);
                } else {
                    ++it;
                }
            }

            if (total_allocated_ + alloc_size > config_.max_pool_size) {
                throw CudaError("Memory pool exhausted", __FILE__, __LINE__);
            }
        }

        void* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, alloc_size));

        Block block{ptr, alloc_size, requested, stream, true};
        all_blocks_[ptr] = block;

        total_allocated_ += alloc_size;
        peak_allocated_ = std::max(peak_allocated_, total_allocated_);

        return ptr;
    }
};

/**
 * @brief RAII wrapper for pool allocations
 */
template<typename T>
class PoolPtr {
public:
    PoolPtr(CudaMemPool& pool, size_t count, cudaStream_t stream = nullptr)
        : pool_(pool), ptr_(nullptr), count_(count) {
        ptr_ = static_cast<T*>(pool_.allocate(count * sizeof(T), stream));
    }

    ~PoolPtr() {
        if (ptr_) {
            pool_.deallocate(ptr_);
        }
    }

    // Move only
    PoolPtr(PoolPtr&& other) noexcept
        : pool_(other.pool_), ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
    }

    PoolPtr& operator=(PoolPtr&& other) noexcept {
        if (this != &other) {
            if (ptr_) pool_.deallocate(ptr_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    PoolPtr(const PoolPtr&) = delete;
    PoolPtr& operator=(const PoolPtr&) = delete;

    T* get() const { return ptr_; }
    T* operator->() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    size_t count() const { return count_; }
    size_t size_bytes() const { return count_ * sizeof(T); }

    T* release() {
        T* tmp = ptr_;
        ptr_ = nullptr;
        return tmp;
    }

private:
    CudaMemPool& pool_;
    T* ptr_;
    size_t count_;
};

} // namespace mempool
