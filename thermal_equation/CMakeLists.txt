cmake_minimum_required(VERSION 3.0.2)
project(thermal_equation)

set(CMAKE_INCLUDE_PATH "")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../lib)

set(SOURCE_FILES
  conductive_cpu.cpp
  conductive_cpu.h
  factory.cpp
  form_factors.cpp
  form_factors.h
  heat_source_cpu.cpp
  heat_source_cpu.h
  parallel_rays_cpu.cpp
  parallel_rays_cpu.h
  radiance_cpu.cpp
  radiance_cpu.h
  system.cpp
  system.h
)

add_library(thermal_equation STATIC ${SOURCE_FILES})
