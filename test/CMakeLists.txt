cmake_minimum_required(VERSION 3.0.2)
project(tests)

set(CMAKE_INCLUDE_PATH "..")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fpermissive")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
include_directories("..")

find_package(CUDA REQUIRED)

link_directories("../lib")
#link_directories($CUDA_LIBRARIES{})

set(SOURCE_FILES
    form_factors.cpp
    export.cpp
    import.cpp
    ray_casting.cpp
    )

add_executable(tests ${SOURCE_FILES})

target_link_libraries(tests ${GTEST_BOTH_LIBRARIES})
target_link_libraries(tests ray_caster cpu_ray_caster cuda_ray_caster form_factors cpu_form_factors math import_export)
target_link_libraries(tests ${CUDA_LIBRARIES})
