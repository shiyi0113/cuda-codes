#include <iostream>
#include <string.h>
#include "sgemm_v0_global_memory.cuh"
#include "sgemm_v1_shared_memory.cuh"
#include "sgemm_v2_thread_coarsening.cuh"
#include "sgemm_v3_using_float4.cuh"

#define A(i, j) a[(i) * N + (j)]
#define B(i, j) b[(i) * N + (j)]
void random_matirx(const int M, const int N, float *a)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
#if 1
            A(i, j) = 2.0 * (float)drand48() - 1.0;
#else
            A(i, j) = (j - i) % 3;
#endif
        }
    }
}

float compare_matrices(const int M, const int N, float *a, float *b)
{
    float max_diff = 0.0f;
    float diff;
    int printed = 0;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            diff = abs(A(i, j) - B(i, j));
            max_diff = (diff > max_diff ? diff : max_diff);
            if (0 == printed)
            {
                if (max_diff > 0.5f)
                {
                    printf("error:i %d j %d diff %f got %f expect %f ", i, j, max_diff, A(i, j), B(i, j));
                    printed = 1;
                }
            }
        }
    }
    return max_diff;
}

void cpu_sgemm(float *A, float *B, float *C, const int M, const int N, const int K)
{
    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            float sum = 0;
            for (int k = 0; k < K; k++)
            {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}
void check(const int M, const int N, float *a, float *b)
{
    float diff = compare_matrices(M, N, a, b);
    if (diff > 0.5f)
    {
        printf("diff too big !\n");
        exit(-1);
    }
    else
    {
        printf("right!\n");
    }
}

int main()
{
    const int M = 512;
    const int N = 512;
    const int K = 512;
    const size_t mem_size_A = M * K * sizeof(float);
    const size_t mem_size_B = K * N * sizeof(float);
    const size_t mem_size_C = M * N * sizeof(float);

    float *h_matrix_A = (float *)malloc(mem_size_A);
    float *h_matrix_B = (float *)malloc(mem_size_B);
    float *h_matrix_C_cpu = (float *)malloc(mem_size_C);

    random_matirx(M, K, h_matrix_A);
    random_matirx(K, N, h_matrix_B);
    memset(h_matrix_C_cpu, 0, mem_size_C);

    /* cpu_calc */
    cpu_sgemm(h_matrix_A, h_matrix_B, h_matrix_C_cpu, M, N, K);

    /* gpu_calc */
    // v0_globalMemory
    float *h_matrix_C_globalMemory = (float *)malloc(mem_size_C);
    memset(h_matrix_C_globalMemory, 0, mem_size_C);
    sgemm_globalMemory(h_matrix_A, h_matrix_B, h_matrix_C_globalMemory, M, N, K);
    check(M, N, h_matrix_C_cpu, h_matrix_C_globalMemory);
    free(h_matrix_C_globalMemory);

    // v1_sharedMemory
    float *h_matrix_C_sharedMemory = (float *)malloc(mem_size_C);
    memset(h_matrix_C_sharedMemory, 0, mem_size_C);
    sgemm_sharedMemory(h_matrix_A, h_matrix_B, h_matrix_C_sharedMemory, M, N, K);
    check(M, N, h_matrix_C_cpu, h_matrix_C_sharedMemory);
    free(h_matrix_C_sharedMemory);

    // v2_threadCoarsening
    float *h_matrix_C_threadCoarsening = (float *)malloc(mem_size_C);
    memset(h_matrix_C_threadCoarsening, 0, mem_size_C);
    sgemm_threadCoarsening(h_matrix_A, h_matrix_B, h_matrix_C_threadCoarsening, M, N, K);
    check(M, N, h_matrix_C_cpu, h_matrix_C_threadCoarsening);
    free(h_matrix_C_threadCoarsening);

    // v3_usingFloat4
    float *h_matrix_C_usingFloat4 = (float *)malloc(mem_size_C);
    memset(h_matrix_C_usingFloat4, 0, mem_size_C);
    sgemm_usingFloat4(h_matrix_A, h_matrix_B, h_matrix_C_usingFloat4, M, N, K);
    check(M, N, h_matrix_C_cpu, h_matrix_C_usingFloat4);
    free(h_matrix_C_usingFloat4);
    /*------------------------------------------------------*/
    // free
    free(h_matrix_A);
    free(h_matrix_B);
    free(h_matrix_C_cpu);
}