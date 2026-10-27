cmake_minimum_required(VERSION 3.10)
project(MatrixMulLabs VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(RELEASE)
    add_compile_options(-O3 -ffast-math)
endif()

add_executable(MatrixMulLabs
    src/inverse.cpp
    src/main.cpp
)

include_directories("..")
