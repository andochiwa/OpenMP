cmake_minimum_required(VERSION 3.20)
project(OpenMP)

set(CMAKE_CXX_STANDARD 14)

add_executable(multi_matrix_parallel hw/multi_matrix_parallel.cpp)
add_executable(multi_vector_parallel hw/multi_vector_parallel.cpp)
add_executable(multi_vector_transpose hw/multi_vector_transpose.cpp)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()