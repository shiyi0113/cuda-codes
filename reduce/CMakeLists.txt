# add_executable(reduce_v0_global_memory reduce_v0_global_memory.cu)
# target_link_libraries(reduce_v0_global_memory PRIVATE 
#     CUDA::cudart 
#     CUDA::cublas
#     #${CUDA_cublas_LIBRARY}
#     )

# add_executable(reduce_v1_shared_memory reduce_v1_shared_memory.cu)
# target_link_libraries(reduce_v1_shared_memory PRIVATE 
#     CUDA::cudart 
#     CUDA::cublas
#     )
# add_executable(reduce_v2_no_divergence_branch reduce_v2_no_divergence_branch.cu)
# target_link_libraries(reduce_v2_no_divergence_branch PRIVATE 
#     CUDA::cudart 
#     CUDA::cublas
#     )
# add_executable(reduce_v3_no_bank_conflict reduce_v3_no_bank_conflict.cu)
# target_link_libraries(reduce_v3_no_bank_conflict PRIVATE 
#     CUDA::cudart 
#     CUDA::cublas
#     )
# add_executable(reduce_v4_thread_coarsening_A reduce_v4_thread_coarsening_A.cu)
# target_link_libraries(reduce_v4_thread_coarsening_A PRIVATE 
#     CUDA::cudart 
#     CUDA::cublas
#     )
# add_executable(reduce_v4_thread_coarsening_B reduce_v4_thread_coarsening_B.cu)
# target_link_libraries(reduce_v4_thread_coarsening_B PRIVATE 
#     CUDA::cudart 
#     CUDA::cublas
#     )
# add_executable(reduce_v5_unroll_last_warp reduce_v5_unroll_last_warp.cu)
# target_link_libraries(reduce_v5_unroll_last_warp PRIVATE 
#     CUDA::cudart 
#     CUDA::cublas
#     )
# add_executable(reduce_v6_completely_unroll reduce_v6_completely_unroll.cu)
# target_link_libraries(reduce_v6_completely_unroll PRIVATE 
#     CUDA::cudart 
#     CUDA::cublas
#     )
# add_executable(reduce_v7_multi_add reduce_v7_multi_add.cu)
# target_link_libraries(reduce_v7_multi_add PRIVATE 
#     CUDA::cudart 
#     CUDA::cublas
#     )
# add_executable(reduce_v8_warp_shuffle reduce_v8_warp_shuffle.cu)
# target_link_libraries(reduce_v8_warp_shuffle PRIVATE 
#     CUDA::cudart 
#     CUDA::cublas
#     )

add_executable(reduce_main main.cu)
target_link_libraries(reduce_main PRIVATE 
    CUDA::cudart 
    CUDA::cublas
    )
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()