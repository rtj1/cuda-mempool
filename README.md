# CUDA Memory Pool

High-performance GPU memory allocator with caching and stream-ordered semantics.

## Features

- **Sub-microsecond allocation**: Reuses cached memory instead of calling cudaMalloc
- **Size-class binning**: Efficient matching of allocations to cached blocks
- **Stream-ordered allocation**: Safe async allocation/deallocation
- **Thread-safe**: Concurrent allocations from multiple CPU threads
- **RAII wrapper**: `PoolPtr<T>` for automatic memory management

## Performance

| Operation | cudaMalloc/Free | MemPool |
|-----------|-----------------|---------|
| 1MB alloc/free | ~500 us | ~1 us |

The pool achieves ~500x faster allocation by:
1. Caching freed memory in size-class bins
2. Avoiding CUDA driver calls for cached allocations
3. Lock-free fast path for single-threaded access

## Usage

### Basic Allocation

```cpp
#include "cuda_mempool.hpp"

using namespace mempool;

// Create pool with default config (4GB max)
CudaMemPool pool;

// Allocate 1MB
void* ptr = pool.allocate(1 << 20);

// Use the memory...
cudaMemset(ptr, 0, 1 << 20);

// Return to pool (not freed, just cached)
pool.deallocate(ptr);
```

### RAII Wrapper

```cpp
CudaMemPool pool;

{
    // Automatically freed when out of scope
    PoolPtr<float> data(pool, 1024);

    // Use like a pointer
    myKernel<<<blocks, threads>>>(data.get(), 1024);
}
// Memory returned to pool here
```

### Stream-Ordered Allocation

```cpp
CudaMemPool pool;
cudaStream_t stream;
cudaStreamCreate(&stream);

// Allocate on stream
void* ptr = pool.allocate_async(size, stream);

// Launch work
myKernel<<<blocks, threads, 0, stream>>>(ptr);

// Deallocate async - memory reusable after stream completes
pool.deallocate_async(ptr, stream);

// Periodically process deferred frees
pool.process_deferred();
```

### Custom Configuration

```cpp
PoolConfig config;
config.max_pool_size = 8ULL << 30;  // 8GB
config.min_block_size = 256;         // Minimum allocation
config.stream_ordered = true;
config.device_id = 0;

CudaMemPool pool(config);
```

## Implementation Details

### Size Classes

Memory is organized into power-of-2 size classes:
- 512B, 1KB, 2KB, 4KB, ..., 2GB
- Allocations rounded up to nearest size class
- Each class maintains a free list of cached blocks

### Stream Ordering

For stream-ordered allocation:
1. `allocate_async`: Records CUDA event, returns immediately
2. `deallocate_async`: Schedules deferred free after event
3. `process_deferred`: Checks events, returns completed blocks to cache

### Thread Safety

- Global mutex protects block metadata
- Per-size-class mutex for free list operations
- Lock contention minimal due to size-class separation

## Building

```bash
mkdir build && cd build
cmake ..
make

# Run tests
./test_mempool
```

## Requirements

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 compiler

## Benchmarks

Run benchmarks with:

```bash
./test_mempool
```

Example output:
```
Benchmarking allocation performance...
  cudaMalloc/Free: 487 us/op
  MemPool alloc/free: 1 us/op
```

## References

- [CUDA Memory Management](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-management)
- [PyTorch CUDACachingAllocator](https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp)
- [cuMemPool API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)
