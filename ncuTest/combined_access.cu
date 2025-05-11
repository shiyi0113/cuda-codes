#include <iostream>
#include <cuda_runtime.h>

__global__ void add(float *d_x,float *d_y,float *d_z){
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    d_z[n] = d_x[n] + d_y[n];
}
__global__ void add1(float *d_x,float *d_y,float *d_z){
    int n = threadIdx.x + blockDim.x * blockIdx.x + 1;
    d_z[n] = d_x[n] + d_y[n];
}
__global__ void add2(float *d_x,float *d_y,float *d_z){
    int tid_permuted = threadIdx.x^0x1;
    int n = tid_permuted + blockDim.x * blockIdx.x;
    d_z[n] = d_x[n] + d_y[n];
}
__global__ void add3(float *d_x,float *d_y,float *d_z){
    int n = (threadIdx.x + blockDim.x * blockIdx.x) / 32;
    d_z[n] = d_x[n] + d_y[n];
}
__global__ void add4(float *d_x,float *d_y,float *d_z){
    int n = (threadIdx.x + blockDim.x * blockIdx.x) * 4;
    d_z[n] = d_x[n] + d_y[n];
}
int main(){
    const int N = 128 * 1024 * 1024;
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

    dim3 Grid(N/256); 
    dim3 Block(64);
    add<<<Grid,Block>>>(d_input_x,d_input_y,d_output);
    cudaDeviceSynchronize();
    add1<<<Grid,Block>>>(d_input_x,d_input_y,d_output);
    cudaDeviceSynchronize();
    add2<<<Grid,Block>>>(d_input_x,d_input_y,d_output);
    cudaDeviceSynchronize();
    add3<<<Grid,Block>>>(d_input_x,d_input_y,d_output);
    cudaDeviceSynchronize();
    add4<<<Grid,Block>>>(d_input_x,d_input_y,d_output);
    cudaFree(d_input_x);
    cudaFree(d_input_y);
    cudaFree(d_output);
    free(h_input_x);
    free(h_input_y);
    free(h_output);
}