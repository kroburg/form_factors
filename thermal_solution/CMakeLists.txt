cmake_minimum_required(VERSION 3.0.2)
project(thermal_solution)

set(CMAKE_INCLUDE_PATH "")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../lib)

set(SOURCE_FILES
  cpu_adams.cpp
  cpu_adams.h
  system.cpp
  system.h
)

add_library(thermal_solution STATIC ${SOURCE_FILES})
