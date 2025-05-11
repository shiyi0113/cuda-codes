#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <unsigned int BLOCK, unsigned int COARSENINGFACTOR>
__global__ void sgemm_threadCoarsening_kernel(float *d_A, float *d_B, float *d_C, const int M, const int N, const int K)
{
    constexpr int BLOCKNUM = BLOCK * COARSENINGFACTOR;
    __shared__ float s_A[BLOCKNUM][BLOCKNUM];
    __shared__ float s_B[BLOCKNUM][BLOCKNUM];
    float *A_begin = d_A + blockIdx.x * BLOCKNUM * K;
    float *B_begin = d_B + blockIdx.y * BLOCKNUM;

    float sum[COARSENINGFACTOR][COARSENINGFACTOR];
    for (int i = 0; i < COARSENINGFACTOR; i++)
        for (int j = 0; j < COARSENINGFACTOR; j++)
            sum[i][j] = 0.0f;
    for (int step = 0; step < (K + BLOCKNUM - 1) / BLOCKNUM; step++)
    {
        s_A[threadIdx.x][threadIdx.y] = A_begin[threadIdx.x * K + threadIdx.y + step * BLOCKNUM];
        s_A[threadIdx.x + BLOCK][threadIdx.y] = A_begin[(threadIdx.x + BLOCK) * K + threadIdx.y + step * BLOCKNUM];
        s_A[threadIdx.x][threadIdx.y + BLOCK] = A_begin[threadIdx.x * K + threadIdx.y + step * BLOCKNUM + BLOCK];
        s_A[threadIdx.x + BLOCK][threadIdx.y + BLOCK] = A_begin[(threadIdx.x + BLOCK) * K + threadIdx.y + step * BLOCKNUM + BLOCK];
        s_B[threadIdx.x][threadIdx.y] = B_begin[(threadIdx.x + step * BLOCKNUM) * N + threadIdx.y];
        s_B[threadIdx.x + BLOCK][threadIdx.y] = B_begin[(threadIdx.x + step * BLOCKNUM + BLOCK) * N + threadIdx.y];
        s_B[threadIdx.x][threadIdx.y + BLOCK] = B_begin[(threadIdx.x + step * BLOCKNUM) * N + threadIdx.y + BLOCK];
        s_B[threadIdx.x + BLOCK][threadIdx.y + BLOCK] = B_begin[(threadIdx.x + step * BLOCKNUM + BLOCK) * N + threadIdx.y + BLOCK];
        __syncthreads();
        for (int i = 0; i < COARSENINGFACTOR; i++)
        {
            for (int j = 0; j < COARSENINGFACTOR; j++)
            {
                int tx = threadIdx.x + i * BLOCK;
                int ty = threadIdx.y + j * BLOCK;
                #pragma unroll
                for (int k = 0; k < BLOCKNUM; k++)
                {
                    sum[i][j] += s_A[tx][k] * s_B[k][ty];
                }
            }
        }
        __syncthreads();
    }

    const int C_m = threadIdx.x + blockIdx.x * BLOCKNUM;
    const int C_n = threadIdx.y + blockIdx.y * BLOCKNUM;
    d_C[C_m * N + C_n] = sum[0][0];
    d_C[C_m * N + C_n + BLOCK] = sum[0][1];
    d_C[(C_m + BLOCK) * N + C_n] = sum[1][0];
    d_C[(C_m + BLOCK) * N + C_n + BLOCK] = sum[1][1];
}

void sgemm_threadCoarsening(float *h_A, float *h_B, float *h_C, const int M, const int N, const int K)
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

    const int BLOCK = 16;
    const int COARSENINGFACTOR = 2;
    dim3 Grid((M + BLOCK - 1) / (BLOCK * COARSENINGFACTOR), (N + BLOCK - 1) / (BLOCK * COARSENINGFACTOR));
    dim3 Block(BLOCK, BLOCK);
    sgemm_threadCoarsening_kernel<BLOCK, COARSENINGFACTOR><<<Grid, Block>>>(d_matrix_A, d_matrix_B, d_matrix_C, M, N, K);
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