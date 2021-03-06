cmake_minimum_required(VERSION 3.0.2)
project(controller)

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
    controller.cpp
   )

add_executable(controller ${SOURCE_FILES})

target_link_libraries(controller ray_caster cuda_ray_caster import_export thermal_equation thermal_solution form_factors cpu_form_factors subject emission math)
target_link_libraries(controller ${CUDA_LIBRARIES})
