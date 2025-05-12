cmake_minimum_required(VERSION 3.20)
project(cuda_notes VERSION 0.1.0 LANGUAGES CUDA CXX C)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
find_package(CUDAToolkit REQUIRED)

add_subdirectory(reduce)
add_subdirectory(ncuTest)
add_subdirectory(Sgemm)