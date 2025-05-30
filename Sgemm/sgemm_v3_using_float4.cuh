#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
template <unsigned int BLOCK, unsigned int COARSENINGFACTOR>
__global__ void sgemm_usingFloat4_kernel(float *d_A, float *d_B, float *d_C, const int M, const int N, const int K)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    constexpr int BLOCKNUM = BLOCK * COARSENINGFACTOR;
    constexpr int NUM_PER_THREAD = COARSENINGFACTOR * COARSENINGFACTOR;
    __shared__ float s_A[BLOCKNUM][BLOCKNUM];
    __shared__ float s_B[BLOCKNUM][BLOCKNUM];
    float *A_begin = d_A + blockIdx.x * BLOCKNUM * K;
    float *B_begin = d_B + blockIdx.y * BLOCKNUM;
    float sum[NUM_PER_THREAD] = {0.0f};
    for (int step = 0; step < (K + BLOCKNUM - 1) / BLOCKNUM; step++)
    {
        /*
        s_A[tx][ty * NUM_PER_THREAD] = A_begin[tx * K + ty * NUM_PER_THREAD + step * BLOCKNUM];
        s_A[tx][ty * NUM_PER_THREAD + 1] = A_begin[tx * K + ty * NUM_PER_THREAD + step * BLOCKNUM + 1];
        s_A[tx][ty * NUM_PER_THREAD + 2] = A_begin[tx * K + ty * NUM_PER_THREAD + step * BLOCKNUM + 2];
        s_A[tx][ty * NUM_PER_THREAD + 3] = A_begin[tx * K + ty * NUM_PER_THREAD + step * BLOCKNUM + 3];
        s_B[tx][ty * NUM_PER_THREAD] = B_begin[(tx + step * BLOCKNUM) * N + ty * NUM_PER_THREAD];
        s_B[tx][ty * NUM_PER_THREAD + 1] = B_begin[(tx + step * BLOCKNUM) * N + ty * NUM_PER_THREAD + 1];
        s_B[tx][ty * NUM_PER_THREAD + 2] = B_begin[(tx + step * BLOCKNUM) * N + ty * NUM_PER_THREAD + 2];
        s_B[tx][ty * NUM_PER_THREAD + 3] = B_begin[(tx + step * BLOCKNUM) * N + ty * NUM_PER_THREAD + 3];
        */
        FETCH_FLOAT4(s_A[tx][ty * NUM_PER_THREAD]) = FETCH_FLOAT4(A_begin[tx * K + ty * NUM_PER_THREAD + step * BLOCKNUM]);
        FETCH_FLOAT4(s_B[tx][ty * NUM_PER_THREAD]) = FETCH_FLOAT4(B_begin[(tx + step * BLOCKNUM) * N + ty * NUM_PER_THREAD]);
        __syncthreads();
        for (int i = 0; i < NUM_PER_THREAD; i++)
        {
            for (int k = 0; k < BLOCKNUM; k++)
            {
                sum[i] += s_A[tx][k] * s_B[k][ty * NUM_PER_THREAD + i];
            }
        }
        __syncthreads();
    }
    const int C_m = blockIdx.x * BLOCKNUM + threadIdx.x;
    const int C_n = blockIdx.y * BLOCKNUM + threadIdx.y * NUM_PER_THREAD;
    for (int i = 0; i < NUM_PER_THREAD; i++)
    {
        d_C[C_m * N + C_n + i] = sum[i];
    }
}

void sgemm_usingFloat4(float *h_A, float *h_B, float *h_C, const int M, const int N, const int K)
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
    const int COARSENINGFACTOR = 2;
    dim3 Grid((M + (BLOCK * COARSENINGFACTOR) - 1) / (BLOCK * COARSENINGFACTOR), (N + (BLOCK * COARSENINGFACTOR) - 1) / (BLOCK * COARSENINGFACTOR));
    dim3 Block(BLOCK * COARSENINGFACTOR, BLOCK / COARSENINGFACTOR);
    sgemm_usingFloat4_kernel<BLOCK, COARSENINGFACTOR><<<Grid, Block>>>(d_matrix_A, d_matrix_B, d_matrix_C, M, N, K);
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