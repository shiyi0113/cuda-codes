add_executable(sgemm_v0_global_memory sgemm_v0_global_memory.cu)
target_link_libraries(sgemm_v0_global_memory PRIVATE 
    CUDA::cudart 
    CUDA::cublas
    )
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(sgemm_v0_global_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()