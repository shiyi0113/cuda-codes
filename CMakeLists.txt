cmake_minimum_required(VERSION 3.20)
project(cuda_notes VERSION 0.1.0 LANGUAGES CUDA CXX C)


find_package(CUDAToolkit REQUIRED)

add_subdirectory(reduce)