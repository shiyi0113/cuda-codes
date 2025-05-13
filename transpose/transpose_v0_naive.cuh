#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <unsigned int MATRIX_M, unsigned int MATRIX_N>
__global__ void transpose_naive_kernel(float *d_input, float *d_output)
{
    int N_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int M_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (N_idx < MATRIX_N && M_idx < MATRIX_M)
    {
        int idx = M_idx * MATRIX_N + N_idx;
        int trans_idx = N_idx * MATRIX_M + M_idx;

        d_output[trans_idx] = d_input[idx];
    }
}

template <unsigned int MATRIX_M, unsigned int MATRIX_N>
void transpose_naive(float *h_input, float *h_output)
{
    const size_t size = MATRIX_M * MATRIX_N;
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, size * sizeof(float));
    cudaMalloc((void **)&d_output, size * sizeof(float));
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, size * sizeof(float));

    {
        dim3 Block(32, 8);
        dim3 Grid((MATRIX_N + Block.x - 1) / Block.x, (MATRIX_M + Block.y - 1) / Block.y);
        transpose_naive_kernel<MATRIX_M, MATRIX_N><<<Grid, Block>>>(d_input, d_output);
        cudaDeviceSynchronize();
    }

    {
        dim3 Block(16, 16);
        dim3 Grid((MATRIX_N + Block.x - 1) / Block.x, (MATRIX_M + Block.y - 1) / Block.y);
        transpose_naive_kernel<MATRIX_M, MATRIX_N><<<Grid, Block>>>(d_input, d_output);
        cudaDeviceSynchronize();
    }

    {
        dim3 Block(8, 32);
        dim3 Grid((MATRIX_N + Block.x - 1) / Block.x, (MATRIX_M + Block.y - 1) / Block.y);
        transpose_naive_kernel<MATRIX_M, MATRIX_N><<<Grid, Block>>>(d_input, d_output);
        cudaDeviceSynchronize();
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "cuda Error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}