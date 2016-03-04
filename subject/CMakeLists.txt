cmake_minimum_required(VERSION 3.0.2)
project(subject)

set(CMAKE_INCLUDE_PATH "")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../lib)

set(SOURCE_FILES
  generator.h
  objects.cpp
  spherial_generator.cpp
  system.h
  cube_generator.cpp
  mesh_graph.cpp
  objects.h
  system.cpp
  unify_normals.cpp)

add_library(subject STATIC ${SOURCE_FILES})
