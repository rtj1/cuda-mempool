cmake_minimum_required(VERSION 3.18)
project(cuda_mempool LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Header-only library
add_library(cuda_mempool INTERFACE)
target_include_directories(cuda_mempool INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cuda_mempool INTERFACE CUDA::cudart)

# Tests
enable_testing()
add_executable(test_mempool tests/test_mempool.cu)
target_link_libraries(test_mempool PRIVATE cuda_mempool)
target_compile_options(test_mempool PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

add_test(NAME test_mempool COMMAND test_mempool)

# Install
install(DIRECTORY include/ DESTINATION include)
install(TARGETS cuda_mempool EXPORT cuda_mempoolTargets)
install(EXPORT cuda_mempoolTargets
    FILE cuda_mempoolConfig.cmake
    NAMESPACE mempool::
    DESTINATION lib/cmake/cuda_mempool
)
