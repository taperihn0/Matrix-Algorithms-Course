cmake_minimum_required(VERSION 3.10)
project(MatrixMulLabs VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(RELEASE)
endif()

add_executable(MatrixMulLabs
    src/main.cpp
    src/matrix.cpp
)

add_executable(Test
    tests/inv.cpp
    src/matrix.cpp
)

target_include_directories(Test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)