#include <iostream>
#include "transpose_v0_naive.cuh"

class Perf
{
public:
    Perf(const std::string &name)
    {
        m_name = name;
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_end);
        cudaEventRecord(m_start);
        cudaEventSynchronize(m_start);
    }
    ~Perf()
    {
        cudaEventRecord(m_end);
        cudaEventSynchronize(m_end);
        float elapsed_time = 0.0f;
        cudaEventElapsedTime(&elapsed_time, m_start, m_end);
        std::cout << m_name << "elapse:" << elapsed_time << " ms" << std::endl;
    }

private:
    std::string m_name;
    cudaEvent_t m_start, m_end;
};

void transpose(float *input, float *output, const int M, const int N)
{
    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            const int input_index = m * N + n;
            const int output_index = n * M + m;
            output[output_index] = input[input_index];
        }
    }
}

bool check(float *cpu_result, float *gpu_result, const int M, const int N)
{
    const int size = M * N;
    for (int i = 0; i < size; i++)
    {
        if (cpu_result[i] != gpu_result[i])
        {
            return false;
        }
    }
    return true;
}

int main()
{
    const int MATRIX_M = 2048;
    const int MATRIX_N = 1024;
    const size_t size = MATRIX_M * MATRIX_N;

    float *h_input = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++)
    {
        h_input[i] = 2.0 * (float)drand48() - 1.0f;
    }
    // cpu_cal
    float *h_cpu_output = (float *)malloc(size * sizeof(float));
    transpose(h_input, h_cpu_output, MATRIX_M, MATRIX_N);

    /*gpu_cal*/

    // naive
    float *h_gpu_output_naive = (float *)malloc(size * sizeof(float));
    transpose_naive<MATRIX_M, MATRIX_N>(h_input, h_gpu_output_naive);
    if (check(h_cpu_output, h_gpu_output_naive, MATRIX_M, MATRIX_N))
    {
        std::cout << "right!" << std::endl;
    }
    else
    {
        std::cout << "error!" << std::endl;
        exit(-1);
    }
    free(h_gpu_output_naive);

    //----------------------------------

    free(h_input);
    free(h_cpu_output);
}