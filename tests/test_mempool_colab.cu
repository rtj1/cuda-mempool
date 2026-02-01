/**
 * CUDA Memory Pool Test Suite for Google Colab
 * =============================================
 *
 * Compile and run in Colab:
 *   !nvcc -std=c++17 -O2 -I include tests/test_mempool_colab.cu -o test_mempool
 *   !./test_mempool
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cassert>
#include <iomanip>
#include <cuda_runtime.h>
#include "cuda_mempool.hpp"

using namespace mempool;
using namespace std::chrono;

#define CUDA_CHECK_TEST(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            return false; \
        } \
    } while(0)

// =============================================================================
// Test Utilities
// =============================================================================

struct TestResult {
    std::string name;
    bool passed;
    std::string error;
};

std::vector<TestResult> results;

void run_test(const std::string& name, bool (*test_fn)()) {
    std::cout << "Testing: " << name << "... " << std::flush;
    try {
        bool passed = test_fn();
        if (passed) {
            std::cout << "[PASS]" << std::endl;
            results.push_back({name, true, ""});
        } else {
            std::cout << "[FAIL]" << std::endl;
            results.push_back({name, false, "Test returned false"});
        }
    } catch (const std::exception& e) {
        std::cout << "[FAIL] " << e.what() << std::endl;
        results.push_back({name, false, e.what()});
    }
}

// =============================================================================
// Basic Allocation Tests
// =============================================================================

bool test_basic_allocation() {
    CudaMemPool pool;

    void* ptr = pool.allocate(1024);
    if (!ptr) return false;

    pool.deallocate(ptr);
    return true;
}

bool test_multiple_allocations() {
    CudaMemPool pool;
    std::vector<void*> ptrs;

    // Allocate 100 blocks
    for (int i = 0; i < 100; ++i) {
        void* ptr = pool.allocate(1024);
        if (!ptr) return false;
        ptrs.push_back(ptr);
    }

    // Deallocate all
    for (void* ptr : ptrs) {
        pool.deallocate(ptr);
    }

    return true;
}

bool test_size_classes() {
    CudaMemPool pool;

    // Test various sizes
    std::vector<size_t> sizes = {512, 1024, 2048, 4096, 8192, 16384, 65536, 1 << 20};

    for (size_t size : sizes) {
        void* ptr = pool.allocate(size);
        if (!ptr) {
            std::cerr << "Failed to allocate " << size << " bytes" << std::endl;
            return false;
        }
        pool.deallocate(ptr);
    }

    return true;
}

bool test_memory_reuse() {
    CudaMemPool pool;

    // Allocate and free
    void* ptr1 = pool.allocate(1024);
    pool.deallocate(ptr1);

    // Should reuse the same block
    void* ptr2 = pool.allocate(1024);

    bool reused = (ptr1 == ptr2);
    pool.deallocate(ptr2);

    return reused;
}

bool test_pool_ptr_raii() {
    CudaMemPool pool;

    {
        PoolPtr<float> ptr(pool, 1000);
        if (!ptr.get()) return false;

        // Write some data
        float* d_ptr = ptr.get();
        CUDA_CHECK_TEST(cudaMemset(d_ptr, 0, 1000 * sizeof(float)));
    }  // ptr goes out of scope, memory returned to pool

    // Pool should have the block available
    auto stats = pool.cache_stats();
    bool has_free_block = false;
    for (const auto& [size, count] : stats) {
        if (count > 0) has_free_block = true;
    }

    return has_free_block;
}

// =============================================================================
// Correctness Tests
// =============================================================================

__global__ void write_kernel(int* data, int value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

__global__ void verify_kernel(int* data, int expected, int n, int* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && data[idx] != expected) {
        atomicExch(result, 0);  // Set to 0 if mismatch
    }
}

bool test_memory_correctness() {
    CudaMemPool pool;

    const int N = 10000;
    void* ptr = pool.allocate(N * sizeof(int));
    int* d_data = static_cast<int*>(ptr);

    // Write pattern
    int blocks = (N + 255) / 256;
    write_kernel<<<blocks, 256>>>(d_data, 42, N);
    CUDA_CHECK_TEST(cudaDeviceSynchronize());

    // Verify pattern
    int* d_result;
    CUDA_CHECK_TEST(cudaMalloc(&d_result, sizeof(int)));
    int one = 1;
    CUDA_CHECK_TEST(cudaMemcpy(d_result, &one, sizeof(int), cudaMemcpyHostToDevice));

    verify_kernel<<<blocks, 256>>>(d_data, 42, N, d_result);
    CUDA_CHECK_TEST(cudaDeviceSynchronize());

    int h_result;
    CUDA_CHECK_TEST(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_result);
    pool.deallocate(ptr);

    return h_result == 1;
}

bool test_no_memory_corruption() {
    CudaMemPool pool;

    const int N = 1000;
    std::vector<void*> ptrs;
    std::vector<int> values;

    // Allocate multiple blocks and write different values
    for (int i = 0; i < 10; ++i) {
        void* ptr = pool.allocate(N * sizeof(int));
        int* d_data = static_cast<int*>(ptr);

        int blocks = (N + 255) / 256;
        write_kernel<<<blocks, 256>>>(d_data, i * 100, N);

        ptrs.push_back(ptr);
        values.push_back(i * 100);
    }

    CUDA_CHECK_TEST(cudaDeviceSynchronize());

    // Verify each block has correct value
    int* d_result;
    CUDA_CHECK_TEST(cudaMalloc(&d_result, sizeof(int)));

    bool all_correct = true;
    for (size_t i = 0; i < ptrs.size(); ++i) {
        int* d_data = static_cast<int*>(ptrs[i]);
        int one = 1;
        CUDA_CHECK_TEST(cudaMemcpy(d_result, &one, sizeof(int), cudaMemcpyHostToDevice));

        int blocks = (N + 255) / 256;
        verify_kernel<<<blocks, 256>>>(d_data, values[i], N, d_result);
        CUDA_CHECK_TEST(cudaDeviceSynchronize());

        int h_result;
        CUDA_CHECK_TEST(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

        if (h_result != 1) {
            std::cerr << "Corruption in block " << i << std::endl;
            all_correct = false;
        }
    }

    cudaFree(d_result);
    for (void* ptr : ptrs) {
        pool.deallocate(ptr);
    }

    return all_correct;
}

// =============================================================================
// Stream-Ordered Allocation Tests
// =============================================================================

bool test_stream_ordered_allocation() {
    PoolConfig config;
    config.stream_ordered = true;
    CudaMemPool pool(config);

    cudaStream_t stream;
    CUDA_CHECK_TEST(cudaStreamCreate(&stream));

    void* ptr = pool.allocate_async(1024, stream);
    if (!ptr) {
        cudaStreamDestroy(stream);
        return false;
    }

    pool.deallocate_async(ptr, stream);

    CUDA_CHECK_TEST(cudaStreamSynchronize(stream));
    pool.process_deferred();

    cudaStreamDestroy(stream);
    return true;
}

bool test_multi_stream() {
    CudaMemPool pool;
    const int NUM_STREAMS = 4;

    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK_TEST(cudaStreamCreate(&streams[i]));
    }

    // Allocate on different streams
    std::vector<void*> ptrs;
    for (int i = 0; i < NUM_STREAMS * 10; ++i) {
        cudaStream_t stream = streams[i % NUM_STREAMS];
        void* ptr = pool.allocate(1024 * (i + 1), stream);
        if (!ptr) return false;
        ptrs.push_back(ptr);
    }

    // Deallocate
    for (void* ptr : ptrs) {
        pool.deallocate(ptr);
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    return true;
}

// =============================================================================
// Performance Benchmarks
// =============================================================================

void benchmark_allocation() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "ALLOCATION LATENCY BENCHMARK" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    const int ITERATIONS = 10000;
    const size_t ALLOC_SIZE = 1 << 20;  // 1MB

    // Warmup
    CudaMemPool pool;
    for (int i = 0; i < 100; ++i) {
        void* ptr = pool.allocate(ALLOC_SIZE);
        pool.deallocate(ptr);
    }

    // Benchmark pool allocator
    auto start = high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) {
        void* ptr = pool.allocate(ALLOC_SIZE);
        pool.deallocate(ptr);
    }
    auto end = high_resolution_clock::now();
    double pool_ns = duration_cast<nanoseconds>(end - start).count() / double(ITERATIONS);

    // Benchmark cudaMalloc/cudaFree
    void* warmup_ptr;
    cudaMalloc(&warmup_ptr, ALLOC_SIZE);
    cudaFree(warmup_ptr);

    start = high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) {
        void* ptr;
        cudaMalloc(&ptr, ALLOC_SIZE);
        cudaFree(ptr);
    }
    end = high_resolution_clock::now();
    double cuda_ns = duration_cast<nanoseconds>(end - start).count() / double(ITERATIONS);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nAllocation size: 1 MB" << std::endl;
    std::cout << "Iterations: " << ITERATIONS << std::endl;
    std::cout << "\n" << std::setw(20) << "Allocator" << std::setw(15) << "Latency (ns)" << std::setw(15) << "Speedup" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    std::cout << std::setw(20) << "cudaMalloc/Free" << std::setw(15) << cuda_ns << std::setw(15) << "1.00x" << std::endl;
    std::cout << std::setw(20) << "CudaMemPool" << std::setw(15) << pool_ns << std::setw(15) << (cuda_ns / pool_ns) << "x" << std::endl;
}

void benchmark_throughput() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "THROUGHPUT BENCHMARK (mixed sizes)" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    CudaMemPool pool;
    std::vector<size_t> sizes = {512, 1024, 4096, 16384, 65536, 262144};
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, sizes.size() - 1);

    const int ITERATIONS = 100000;
    std::vector<void*> ptrs;
    ptrs.reserve(100);

    // Pool benchmark
    auto start = high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) {
        size_t size = sizes[dist(rng)];
        void* ptr = pool.allocate(size);
        ptrs.push_back(ptr);

        if (ptrs.size() > 50) {
            pool.deallocate(ptrs.back());
            ptrs.pop_back();
        }
    }
    for (void* ptr : ptrs) pool.deallocate(ptr);
    auto end = high_resolution_clock::now();
    double pool_ms = duration_cast<milliseconds>(end - start).count();
    ptrs.clear();

    // cudaMalloc benchmark
    rng.seed(42);
    start = high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) {
        size_t size = sizes[dist(rng)];
        void* ptr;
        cudaMalloc(&ptr, size);
        ptrs.push_back(ptr);

        if (ptrs.size() > 50) {
            cudaFree(ptrs.back());
            ptrs.pop_back();
        }
    }
    for (void* ptr : ptrs) cudaFree(ptr);
    end = high_resolution_clock::now();
    double cuda_ms = duration_cast<milliseconds>(end - start).count();

    double pool_ops = ITERATIONS / (pool_ms / 1000.0);
    double cuda_ops = ITERATIONS / (cuda_ms / 1000.0);

    std::cout << "\nIterations: " << ITERATIONS << std::endl;
    std::cout << "\n" << std::setw(20) << "Allocator" << std::setw(15) << "Time (ms)" << std::setw(18) << "ops/sec" << std::endl;
    std::cout << std::string(53, '-') << std::endl;
    std::cout << std::setw(20) << "cudaMalloc/Free" << std::setw(15) << cuda_ms << std::setw(18) << std::fixed << cuda_ops << std::endl;
    std::cout << std::setw(20) << "CudaMemPool" << std::setw(15) << pool_ms << std::setw(18) << std::fixed << pool_ops << std::endl;
    std::cout << "\nSpeedup: " << std::setprecision(2) << (cuda_ms / pool_ms) << "x" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    // Print device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << std::string(70, '=') << std::endl;
    std::cout << "CUDA MEMORY POOL TEST SUITE" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Run tests
    std::cout << "\nBASIC TESTS" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    run_test("Basic allocation", test_basic_allocation);
    run_test("Multiple allocations", test_multiple_allocations);
    run_test("Size classes", test_size_classes);
    run_test("Memory reuse", test_memory_reuse);
    run_test("PoolPtr RAII", test_pool_ptr_raii);

    std::cout << "\nCORRECTNESS TESTS" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    run_test("Memory correctness", test_memory_correctness);
    run_test("No memory corruption", test_no_memory_corruption);

    std::cout << "\nSTREAM-ORDERED TESTS" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    run_test("Stream-ordered allocation", test_stream_ordered_allocation);
    run_test("Multi-stream", test_multi_stream);

    // Run benchmarks
    benchmark_allocation();
    benchmark_throughput();

    // Summary
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "TEST SUMMARY" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    int passed = 0, failed = 0;
    for (const auto& r : results) {
        if (r.passed) passed++;
        else failed++;
    }

    std::cout << "\nTotal: " << results.size() << " tests" << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;

    if (failed > 0) {
        std::cout << "\nFailed tests:" << std::endl;
        for (const auto& r : results) {
            if (!r.passed) {
                std::cout << "  - " << r.name << ": " << r.error << std::endl;
            }
        }
    }

    std::cout << "\n" << std::string(70, '=') << std::endl;
    if (failed == 0) {
        std::cout << "ALL TESTS PASSED!" << std::endl;
    } else {
        std::cout << "SOME TESTS FAILED (" << failed << "/" << results.size() << ")" << std::endl;
    }
    std::cout << std::string(70, '=') << std::endl;

    return failed > 0 ? 1 : 0;
}
