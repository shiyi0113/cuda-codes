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

    )
add_executable(reduce_v2_no_divergence_branch reduce_v2_no_divergence_branch.cu)
target_link_libraries(reduce_v2_no_divergence_branch PRIVATE 
    CUDA::cudart 
    CUDA::cublas
    )
add_executable(reduce_v3_no_bank_conflict reduce_v3_no_bank_conflict.cu)
target_link_libraries(reduce_v3_no_bank_conflict PRIVATE 
    CUDA::cudart 
    CUDA::cublas
    )
add_executable(reduce_v4_thread_coarsening_A reduce_v4_thread_coarsening_A.cu)
target_link_libraries(reduce_v4_thread_coarsening_A PRIVATE 
    CUDA::cudart 
    CUDA::cublas
    )