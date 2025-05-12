#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <unsigned int BLOCK>
__global__ void sgemm_sharedMemory_kernel(float *d_A, float *d_B, float *d_C, const int M, const int N, const int K)
{
    __shared__ float s_A[BLOCK][BLOCK];
    __shared__ float s_B[BLOCK][BLOCK];

    float *A_begin = d_A + blockIdx.x * blockDim.x * K;
    float *B_begin = d_B + blockIdx.y * blockDim.y;

    float sum = 0.0f;
    for (int step = 0; step < (K + BLOCK - 1) / BLOCK; step++)
    {
        s_A[threadIdx.x][threadIdx.y] = A_begin[threadIdx.x * K + threadIdx.y + step * BLOCK];
        s_B[threadIdx.x][threadIdx.y] = B_begin[(threadIdx.x + step * BLOCK) * N + threadIdx.y];
        __syncthreads();
        for (int i = 0; i < BLOCK; i++)
            sum += s_A[threadIdx.x][i] * s_B[i][threadIdx.y];
        __syncthreads();
    }

    const int C_m = threadIdx.x + blockIdx.x * blockDim.x;
    const int C_n = threadIdx.y + blockIdx.y * blockDim.y;
    d_C[C_m * N + C_n] = sum;
}

void sgemm_sharedMemory(float *h_A, float *h_B, float *h_C, const int M, const int N, const int K)
{
    const size_t mem_size_A = M * K * sizeof(float);
    const size_t mem_size_B = K * N * sizeof(float);
    const size_t mem_size_C = M * N * sizeof(float);

    float *d_matrix_A, *d_matrix_B, *d_matrix_C;
    cudaMalloc((void **)&d_matrix_A, mem_size_A);
    cudaMalloc((void **)&d_matrix_B, mem_size_B);
    cudaMalloc((void **)&d_matrix_C, mem_size_C);
    cudaMemcpy(d_matrix_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
    cudaMemset(d_matrix_C, 0, mem_size_C);
    const int BLOCK = 16;
    dim3 Grid((M + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);
    dim3 Block(BLOCK, BLOCK);
    sgemm_sharedMemory_kernel<BLOCK><<<Grid, Block>>>(d_matrix_A, d_matrix_B, d_matrix_C, M, N, K);
    cudaMemcpy(h_C, d_matrix_C, mem_size_C, cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "cuda Error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaFree(d_matrix_A);
    cudaFree(d_matrix_B);
    cudaFree(d_matrix_C);
}