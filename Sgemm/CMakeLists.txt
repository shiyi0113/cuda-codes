add_executable(main main.cu)
target_link_libraries(main PRIVATE 
    CUDA::cudart 
    CUDA::cublas
    )
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()