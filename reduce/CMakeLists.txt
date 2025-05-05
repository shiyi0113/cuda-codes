add_executable(reduce_v0_global_memory reduce_v0_global_memory.cu)
target_link_libraries(reduce_v0_global_memory PRIVATE 
    CUDA::cudart 
    CUDA::cublas
    #${CUDA_cublas_LIBRARY}
    )

add_executable(reduce_v1_shared_memory reduce_v1_shared_memory.cu)
target_link_libraries(reduce_v1_shared_memory PRIVATE 
    CUDA::cudart 
    CUDA::cublas
    #${CUDA_cublas_LIBRARY}
    )