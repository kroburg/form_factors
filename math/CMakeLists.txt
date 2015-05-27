cmake_minimum_required(VERSION 3.0.2)
project(math)

set(CMAKE_INCLUDE_PATH "")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../lib)

set(SOURCE_FILES
    types.h
    mat.h
    operations.h    
    triangle.h
    mat.cpp
    triangle.cpp
    )

add_library(math STATIC ${SOURCE_FILES})
