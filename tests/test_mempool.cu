/**
 * @file test_mempool.cu
 * @brief Tests for CUDA memory pool
 */

#include "../include/cuda_mempool.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <thread>
#include <cassert>

using namespace mempool;

// Simple test kernel
__global__ void fill_kernel(float* data, int n, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

void test_basic_allocation() {
    std::cout << "Testing basic allocation..." << std::endl;

    CudaMemPool pool;

    // Allocate various sizes
    void* ptr1 = pool.allocate(1024);
    void* ptr2 = pool.allocate(4096);
    void* ptr3 = pool.allocate(1 << 20);  // 1MB

    assert(ptr1 != nullptr);
    assert(ptr2 != nullptr);
    assert(ptr3 != nullptr);

    // Deallocate
    pool.deallocate(ptr1);
    pool.deallocate(ptr2);
    pool.deallocate(ptr3);

    // Re-allocate same sizes (should reuse)
    void* ptr4 = pool.allocate(1024);
    void* ptr5 = pool.allocate(4096);

    // Should get same pointers back (from cache)
    assert(ptr4 == ptr1 || ptr4 == ptr2 || ptr4 == ptr3);

    pool.deallocate(ptr4);
    pool.deallocate(ptr5);

    std::cout << "  PASSED" << std::endl;
}

void test_pool_ptr() {
    std::cout << "Testing PoolPtr RAII wrapper..." << std::endl;

    CudaMemPool pool;

    {
        PoolPtr<float> data(pool, 1024);
        assert(data.get() != nullptr);
        assert(data.count() == 1024);
        assert(data.size_bytes() == 1024 * sizeof(float));

        // Use the memory
        fill_kernel<<<4, 256>>>(data.get(), 1024, 3.14f);
        cudaDeviceSynchronize();
    }
    // Memory automatically returned to pool

    std::cout << "  PASSED" << std::endl;
}

void test_stream_ordered() {
    std::cout << "Testing stream-ordered allocation..." << std::endl;

    PoolConfig config;
    config.stream_ordered = true;
    CudaMemPool pool(config);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    void* ptr = pool.allocate_async(1 << 20, stream);
    assert(ptr != nullptr);

    // Launch work on stream
    float* fptr = static_cast<float*>(ptr);
    fill_kernel<<<256, 256, 0, stream>>>(fptr, 1 << 18, 2.71f);

    // Deallocate async
    pool.deallocate_async(ptr, stream);

    // Process deferred frees
    cudaStreamSynchronize(stream);
    pool.process_deferred();

    cudaStreamDestroy(stream);

    std::cout << "  PASSED" << std::endl;
}

void benchmark_allocation() {
    std::cout << "\nBenchmarking allocation performance..." << std::endl;

    const int num_iterations = 1000;
    const size_t alloc_size = 1 << 20;  // 1MB

    // Benchmark cudaMalloc
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            void* ptr;
            cudaMalloc(&ptr, alloc_size);
            cudaFree(ptr);
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "  cudaMalloc/Free: " << duration.count() / num_iterations << " us/op" << std::endl;
    }

    // Benchmark pool
    {
        CudaMemPool pool;

        // Warm up
        void* warmup = pool.allocate(alloc_size);
        pool.deallocate(warmup);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            void* ptr = pool.allocate(alloc_size);
            pool.deallocate(ptr);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "  MemPool alloc/free: " << duration.count() / num_iterations << " us/op" << std::endl;
    }
}

void test_concurrent_allocation() {
    std::cout << "Testing concurrent allocation..." << std::endl;

    CudaMemPool pool;
    const int num_threads = 4;
    const int allocs_per_thread = 100;

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&pool, allocs_per_thread]() {
            std::vector<void*> ptrs;
            for (int i = 0; i < allocs_per_thread; ++i) {
                size_t size = (rand() % 10 + 1) * 1024;
                ptrs.push_back(pool.allocate(size));
            }
            for (void* ptr : ptrs) {
                pool.deallocate(ptr);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "  PASSED" << std::endl;
}

void print_stats(const CudaMemPool& pool) {
    std::cout << "\nPool Statistics:" << std::endl;
    std::cout << "  Total allocated: " << pool.total_allocated() / (1 << 20) << " MB" << std::endl;
    std::cout << "  Peak allocated: " << pool.peak_allocated() / (1 << 20) << " MB" << std::endl;
    std::cout << "  Num allocations: " << pool.num_allocations() << std::endl;

    std::cout << "\nCache by size class:" << std::endl;
    for (auto [size, count] : pool.cache_stats()) {
        if (count > 0) {
            std::cout << "  " << size << " bytes: " << count << " free blocks" << std::endl;
        }
    }
}

int main() {
    std::cout << "CUDA Memory Pool Tests" << std::endl;
    std::cout << "======================" << std::endl;

    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using device: " << prop.name << std::endl;
    std::cout << "Total memory: " << prop.totalGlobalMem / (1 << 30) << " GB" << std::endl;
    std::cout << std::endl;

    try {
        test_basic_allocation();
        test_pool_ptr();
        test_stream_ordered();
        test_concurrent_allocation();
        benchmark_allocation();

        std::cout << "\nAll tests passed!" << std::endl;
    } catch (const CudaError& e) {
        std::cerr << "CUDA Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
