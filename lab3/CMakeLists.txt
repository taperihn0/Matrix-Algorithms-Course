cmake_minimum_required(VERSION 3.10)
project(MatrixMulLabs VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(RELEASE)
    add_compile_options(-Ofast -mtune=native -march=native)
endif()

add_library(lab1MatrixLib STATIC
    ../lab1/src/matrix.cpp
    ../lab1/src/matrix_utils.cpp
)

target_include_directories(lab1MatrixLib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../lab1/src
)

add_library(lab3MatrixLib STATIC
    src/compress.cpp
    src/picture.cpp
)

target_include_directories(lab3MatrixLib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(lab3MatrixLib PUBLIC lab1MatrixLib)

add_executable(MatrixCompressLab
    src/main.cpp
)

target_link_libraries(MatrixCompressLab PRIVATE lab3MatrixLib)

# Fixed svd test cases read by src/tests.hpp (Debug builds only).
target_compile_definitions(MatrixCompressLab PRIVATE
    LAB3_SVD_TEST_DATA_FILE="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/svd_cases.txt"
)