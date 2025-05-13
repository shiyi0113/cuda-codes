add_executable(transpose main.cu)
target_link_libraries(transpose PRIVATE 
    CUDA::cudart 
    CUDA::cublas
    )
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(transpose PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
target_compile_options(transpose PRIVATE -lineinfo)