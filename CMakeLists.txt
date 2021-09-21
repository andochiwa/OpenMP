cmake_minimum_required(VERSION 3.20)
project(OpenMP)

set(CMAKE_CXX_STANDARD 14)

add_executable(hello_world hw/hello_world.cpp)
add_executable(test_for hw/test_for.cpp)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()