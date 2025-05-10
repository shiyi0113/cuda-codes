#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define A(i,j) a[(i)*N + (j)]
#define B(i,j) b[(i)*N + (j)]
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

template<unsigned int BLOCK,unsigned int COARSEFACTOR>
__global__ void sgemm_threadCoarsening(float* d_A,float* d_B,float* d_C,const int M,const int N,const int K){
    constexpr int BLOCKNUM = BLOCK * COARSEFACTOR;
    __shared__ float s_A[BLOCKNUM][BLOCKNUM];
    __shared__ float s_B[BLOCKNUM][BLOCKNUM];
    float r_C[COARSEFACTOR][COARSEFACTOR];
    for(int i = 0;i < COARSEFACTOR;i++){
        for(int j = 0;j < COARSEFACTOR;j++){
            r_C[i][j] = 0.0f;
        }
    }
    float* A_begin = d_A + blockIdx.x * BLOCKNUM * K;
    float* B_begin = d_B + blockIdx.y * BLOCKNUM;
    const int C_m = blockIdx.x * BLOCKNUM + threadIdx.x;
    const int C_n = blockIdx.y * BLOCKNUM + threadIdx.y;

    for(size_t step = 0;step < (K + BLOCKNUM - 1)/BLOCKNUM;step++){
        for(int i = 0;i < COARSEFACTOR;i++){
            for(int j = 0;j < COARSEFACTOR;j++){
                int tx = threadIdx.x + i * BLOCK;
                int ty = threadIdx.y + j * BLOCK;
                s_A[tx][ty] = A_begin[tx * K + ty + step*BLOCKNUM];
                s_B[tx][ty] = B_begin[(tx + step*BLOCKNUM)*N + ty];
            }
        }
        __syncthreads();
        for(int i = 0;i < COARSEFACTOR;i++){
            for(int j = 0;j < COARSEFACTOR;j++){
                int tx = threadIdx.x + i * BLOCKNUM/2;
                int ty = threadIdx.y + j * BLOCKNUM/2;
                for(int k = 0;k < BLOCKNUM;k++)
                    r_C[i][j] += s_A[tx][k] * s_B[k][ty];
            }
        }
        __syncthreads();
    }
    for(int i = 0;i < COARSEFACTOR;i++){
        for(int j = 0;j < COARSEFACTOR;j++){
            d_C[(C_m + i * BLOCK) * N + C_n + j * BLOCK] = r_C[i][j];
        }
    }
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

    const int BLOCK = 16;
    const int COARSEFACTOR = 2;
    dim3 Grid((M+BLOCK-1)/BLOCK/COARSEFACTOR,(N+BLOCK-1)/BLOCK/COARSEFACTOR);
    dim3 Block(BLOCK,BLOCK);
    sgemm_threadCoarsening<BLOCK,COARSEFACTOR><<<Grid,Block>>>(d_matrix_A,d_matrix_B,d_matrix_C,M,N,K);
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