cmake_minimum_required(VERSION 3.0.2)
project(cuda_ray_caster)

set(CMAKE_INCLUDE_PATH "")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../lib)

find_package(CUDA QUIET REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_20,code=sm_20)

include_directories("../../../NVIDIA_CUDA-7.0_Samples/common/inc")

set(SOURCE_FILES
    cuda_system.h
    system.cuh
    system.cu
    )


cuda_add_library(cuda_ray_caster STATIC ${SOURCE_FILES})

