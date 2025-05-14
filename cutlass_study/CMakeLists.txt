set(CUTLASS_PATH "/root/cuda-codes/third_party/cutlass/include")
set(CUTLASS_UTIL_PATH "/root/cuda-codes/third_party/cutlass/tools/util/include")
set(CUTLASS_NVCC_ARCHS 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

add_executable(v1_print_half v1_print_half.cu)
target_link_libraries(v1_print_half PRIVATE 
    CUDA::cudart 
    CUDA::cublas
    )

target_include_directories(v1_print_half PRIVATE ${CUTLASS_PATH})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(v1_print_half PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G>>)
else()
    target_compile_options(v1_print_half PRIVATE -lineinfo)
endif()