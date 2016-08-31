cmake_minimum_required(VERSION 3.0.2)
project(perf_test)

set(CMAKE_INCLUDE_PATH "..")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fpermissive")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

find_package(CUDA REQUIRED)

include_directories("..")
link_directories("../lib")

include_directories("/usr/local/cuda/samples/common/inc")

set(SOURCE_FILES
    perfTest.cpp
   )

add_executable(perf_test ${SOURCE_FILES})

target_link_libraries(perf_test math import_export subject thermal_equation thermal_solution emission ray_caster cuda_ray_caster form_factors cpu_form_factors)
target_link_libraries(perf_test ${CUDA_LIBRARIES})
