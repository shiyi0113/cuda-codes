add_executable(combined_access combined_access.cu)
target_link_libraries(combined_access PRIVATE 
    CUDA::cudart 
    CUDA::cublas
    )
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(combined_access PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
target_compile_options(combined_access PRIVATE -lineinfo)