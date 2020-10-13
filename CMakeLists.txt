### CMakeLists.txt for CUDA
project(MY_PROJECT LANGUAGES CUDA CXX)
cmake_minimum_required(VERSION 3.10)
find_package(CUDA REQUIRED)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    "${CUDA_NVCC_FLAGS} -std=c++11";
    )

# Specify include directories
include_directories(
    kernels
    utility
    /home/beomsik/cuda/include
    /home/beomsik/cuda/lib64
    )

# Specify library paths
link_directories(
    /home/beomsik/cuda/lib64
    )

CUDA_INCLUDE_DIRECTORIES(
    cudnn
    cublas
    cnmem
)

add_compile_options(-lcublas -lcudnn)

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    vdnn_sample
    vdnn_sample.cpp
    )

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(
    ${CUDA_LIBRARIES}
    )