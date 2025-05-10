#include <iostream>
#include <cuda_runtime.h>

__global__ void add(float *d_x,float *d_y,float *d_z){
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    d_z[n] = d_x[n] + d_y[n];
}

int main(){
    const int N = 32 * 1024 * 1024;
    float* h_input_x = (float*)malloc(N * sizeof(float));
    float* h_input_y = (float*)malloc(N * sizeof(float));

    float *d_input_x,*d_input_y;
    cudaMalloc((void**)&d_input_x,N*sizeof(float));
    cudaMalloc((void**)&d_input_y,N*sizeof(float));
    cudaMemcpy(d_input_x,h_input_x,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_y,h_input_y,N*sizeof(float),cudaMemcpyHostToDevice);

    float *h_output = (float*)malloc(N*sizeof(float));
    float *d_output;
    cudaMalloc((void**)&d_output,N*sizeof(float));

    dim3 Grid(N/256);  //只计算前1/4的数据求和  即 8*1024*1024个数据
    dim3 Block(64);
    for(int i=0;i<2;i++){
        add<<<Grid,Block>>>(d_input_x,d_input_y,d_output);
        cudaDeviceSynchronize();
    }
    cudaFree(d_input_x);
    cudaFree(d_input_y);
    cudaFree(d_output);
    free(h_input_x);
    free(h_input_y);
    free(h_output);
}