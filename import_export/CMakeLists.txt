cmake_minimum_required(VERSION 3.0.2)
project(import_export)

set(CMAKE_INCLUDE_PATH "")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../lib)

set(SOURCE_FILES
csv_export.cpp
csv_export.h
obj_export.cpp
obj_export.h
obj_import.cpp
obj_import.h
    )

add_library(import_export STATIC ${SOURCE_FILES})

