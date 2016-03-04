cmake_minimum_required(VERSION 3.0.2)
project(emission)

set(CMAKE_INCLUDE_PATH "")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../lib)

set(SOURCE_FILES
  malley_cpu.cpp
  malley_emission.cpp
  parallel_rays_cpu.cpp
  system.cpp
  malley_cpu.h
  malley_emission.h
  parallel_rays_cpu.h
  system.h
)

add_library(emission STATIC ${SOURCE_FILES})
