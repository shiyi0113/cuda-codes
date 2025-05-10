#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define A(i,j) a[(i)*N + (j)]
#define B(i,j) b[(i)*N + (j)]
#define FETCH_FLOAT4(pointer)(reinterpret_cast<float4*>(&(pointer))[0])

void random_matirx(const int M,const int N,float *a){
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
#if 1
            A(i,j) = 2.0*(float)drand48()-1.0;
#else
            A(i,j) = (j-i)%3;
#endif
        }
    }
}

float compare_matrices(const int M,const int N,float *a,float *b){
    float max_diff = 0.0f;
    float diff;
    int printed = 0;
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            diff = abs(A(i,j)-B(i,j));
            max_diff = (diff>max_diff?diff:max_diff);
            if(0 == printed){
                if(max_diff>0.5f){
                    printf("\n error:i %d j %d diff %f got %f expect %f ",i,j,max_diff,A(i,j),B(i,j));
                    printed = 1;
                }
            }
        }
    }
    return max_diff;
}

void cpu_sgemm(float* A,float* B,float* C,const int M,const int N,const int K){
    for(int m=0;m<M;m++){
        for(int n=0;n<N;n++){
            float sum = 0;
            for(int k=0;k<K;k++){
                sum += A[m*K + k]*B[k*N + n];
            }
            C[m*N+n] = sum;
        }
    }
}

template<unsigned int BLOCK_M,unsigned int BLOCK_N,unsigned int BLOCK_K,unsigned int NUM_PER_THREAD>
__global__ void sgemm_usingFloat4(float* d_A,float* d_B,float* d_C,const int M,const int N,const int K){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float* A_begin = d_A + blockIdx.x * BLOCK_M * K;
    float* B_begin = d_B + blockIdx.y * BLOCK_N;
    __shared__ float s_A[BLOCK_M][BLOCK_K];
    __shared__ float s_B[BLOCK_K][BLOCK_N];
    float temp[NUM_PER_THREAD]={0.0};
    for(int step = 0;step<(K+BLOCK_K-1)/BLOCK_K;step++){
        FETCH_FLOAT4(s_A[tx][ty * NUM_PER_THREAD]) = FETCH_FLOAT4(A_begin[tx * K + ty * NUM_PER_THREAD + step * BLOCK_K]);
        FETCH_FLOAT4(s_B[tx][ty * NUM_PER_THREAD]) = FETCH_FLOAT4(B_begin[(tx + step * BLOCK_K)*N + ty * NUM_PER_THREAD]);
        __syncthreads();
        for(int i=0;i<NUM_PER_THREAD;i++){
            for(int k = 0;k < BLOCK_K;k++){
                temp[i] += s_A[tx][k]*s_B[k][ty*NUM_PER_THREAD + i];
            }
        }
        __syncthreads();
    }
    const int C_m = blockIdx.x * BLOCK_M + threadIdx.x;
    const int C_n = blockIdx.y * BLOCK_N + threadIdx.y * NUM_PER_THREAD;
    for(int i = 0;i < NUM_PER_THREAD;i++)
        d_C[C_m * N + C_n + i] = temp[i];
}

int main(){
    const int M = 512;
    const int N = 512;
    const int K = 512;
    const size_t mem_size_A = M * K *sizeof(float);
    const size_t mem_size_B = K * N *sizeof(float);
    const size_t mem_size_C = M * N *sizeof(float);

    float* h_matrix_A = (float*)malloc(mem_size_A);
    float* h_matrix_B = (float*)malloc(mem_size_B);
    float* h_matrix_C = (float*)malloc(mem_size_C);
    float* h_matrix_C_cpu = (float*)malloc(mem_size_C);

    random_matirx(M,K,h_matrix_A);
    random_matirx(K,N,h_matrix_B);
    memset(h_matrix_C,0,mem_size_C);
    memset(h_matrix_C_cpu,0,mem_size_C);

    // cpu_calc
    cpu_sgemm(h_matrix_A,h_matrix_B,h_matrix_C_cpu,M,N,K);

    // gpu_calc
    float *d_matrix_A,*d_matrix_B,*d_matrix_C;
    cudaMalloc((void**)&d_matrix_A,mem_size_A);
    cudaMalloc((void**)&d_matrix_B,mem_size_B);
    cudaMalloc((void**)&d_matrix_C,mem_size_C);
    cudaMemcpy(d_matrix_A,h_matrix_A,mem_size_A,cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_B,h_matrix_B,mem_size_B,cudaMemcpyHostToDevice);

    const int BLOCK_M = 32;
    const int BLOCK_N = 32;
    const int BLOCK_K = 32;
    const int NUM_PER_THREAD = 4;
    dim3 Grid((M + BLOCK_M - 1)/BLOCK_M,(N + BLOCK_N - 1)/BLOCK_N);
    dim3 Block(BLOCK_M,BLOCK_N/NUM_PER_THREAD);
    sgemm_usingFloat4<BLOCK_M,BLOCK_N,BLOCK_K,NUM_PER_THREAD><<<Grid,Block>>>(d_matrix_A,d_matrix_B,d_matrix_C,M,N,K);
    cudaMemcpy(h_matrix_C,d_matrix_C,mem_size_C,cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();
    if(err !=cudaSuccess){
        std::cout<<"cuda Error: "<< cudaGetErrorString(err)<<std::endl;
    }
    // check
    float diff = compare_matrices(M,N,h_matrix_C,h_matrix_C_cpu);
    if(diff > 0.5f){
        printf("diff too big !\n");
        exit(-1);
    }else{
        printf("right!\n");
    }
    // free
    free(h_matrix_A);
    free(h_matrix_B);
    free(h_matrix_C);
    free(h_matrix_C_cpu);
    cudaFree(d_matrix_A);
    cudaFree(d_matrix_B);
    cudaFree(d_matrix_C);
}