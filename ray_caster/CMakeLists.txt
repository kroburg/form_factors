cmake_minimum_required(VERSION 3.0.2)
project(ray_caster)

set(CMAKE_INCLUDE_PATH "")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../lib)

set(SOURCE_FILES
naive_cpu.cpp
naive_cpu.h
system.cpp
system.h
zgrid_cpu.cpp
zgrid_cpu.h
    )


add_library(ray_caster STATIC ${SOURCE_FILES})

