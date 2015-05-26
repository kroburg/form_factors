cmake_minimum_required(VERSION 3.0.2)
get_filename_component(ProjectId ${CMAKE_SOURCE_DIR} NAME)
string(REPLACE " " "_" ProjectId ${ProjectId})
project(${ProjectId})

set(CMAKE_INCLUDE_PATH "")

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
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
    operations.cpp
    triangle.cpp
    )

add_library(${ProjectId} STATIC ${SOURCE_FILES})

