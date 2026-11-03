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

add_library(lab1_matrix STATIC
    ../lab1/src/matrix.cpp
    ../lab1/src/matrix_utils.cpp
)

target_include_directories(lab1_matrix PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../lab1/src
)

add_library(lab2_algorithms STATIC
    src/inverse.cpp
    src/lu.cpp
)

target_include_directories(lab2_algorithms PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(lab2_algorithms PUBLIC lab1_matrix)

add_executable(MatrixMulLabs
    src/main.cpp
)

target_link_libraries(MatrixMulLabs PRIVATE lab2_algorithms)

enable_testing()

add_executable(TestInverse tests/inverse.cpp)
target_link_libraries(TestInverse PRIVATE lab2_algorithms)
add_test(NAME inverse COMMAND TestInverse)

add_executable(TestLU tests/lu.cpp)
target_link_libraries(TestLU PRIVATE lab2_algorithms)
add_test(NAME lu COMMAND TestLU)
