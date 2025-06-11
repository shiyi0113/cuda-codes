#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define THREAD_PER_BLOCK 256
bool check(float *out,float *res,int n);
// each block proceses twice the amount of data.
__global__ void reduce4A(float *d_input,float *d_output){
    __shared__ float s_input[THREAD_PER_BLOCK];
    float* input_begin = d_input+blockIdx.x * blockDim.x * 2;

    s_input[threadIdx.x] = input_begin[threadIdx.x]+input_begin[threadIdx.x + blockDim.x];
    __syncthreads();
    for(int i = blockDim.x/2;i > 0;i /= 2){
        if(threadIdx.x < i)
            s_input[threadIdx.x] += s_input[threadIdx.x + i];
        __syncthreads();
    }

    if(threadIdx.x == 0){
        d_output[blockIdx.x] = s_input[0];
    }
}

void reduce4A_thread_coarsening(){
    const int N = 32*1024*1024;
    int NUM_PER_BLOCK = 2 * THREAD_PER_BLOCK;
    int block_num = N/NUM_PER_BLOCK;
    
    // original data vector
    float *input = (float*)malloc(N*sizeof(float));
    float *d_input;
    cudaMalloc((void**)&d_input,N *sizeof(float));
    // assign random numbers
    for(int i=0;i<N;i++){
        input[i] = 1.0;
    }

    // save gpu result vector.output[blockIdx.x] = this block sum
    float *output = (float*)malloc(block_num*sizeof(float));
    float *d_output;
    cudaMalloc((void**)&d_output,(block_num*sizeof(float)));

    // save cpu result vector
    float *h_result = (float*)malloc(block_num*sizeof(float));

    // cpu reduce
    for(int i=0;i<block_num;i++){
        float cur = 0;
        for(int j=0;j<NUM_PER_BLOCK;j++){
            cur+=input[i*NUM_PER_BLOCK+j];
        }
        h_result[i]=cur;
    }

    // gpu reduce
    cudaMemcpy(d_input,input,N*sizeof(float),cudaMemcpyHostToDevice);
    dim3 Grid(block_num,1);
    dim3 Block(THREAD_PER_BLOCK,1);
    reduce4A<<<Grid,Block>>>(d_input,d_output);
    cudaMemcpy(output,d_output,block_num *sizeof(float),cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if(err !=cudaSuccess){
        std::cout<<"cuda Error: "<< cudaGetErrorString(err)<<std::endl;
    }

    // check
    if(check(output,h_result,block_num)){
        std::cout<<"the ans is right!"<<std::endl;
    }else{
        std::cout<<"the ans is wrong!"<<std::endl;
        std::cout<<"h_result: ";
        for(int i=0;i<10;i++){
            std::cout<<h_result[i]<<" ";
        }
        std::cout<<std::endl;
        std::cout<<"output: ";
        for(int i=0;i<10;i++){
            std::cout<<output[i]<<" ";
        }
        std::cout<<std::endl;
    }

    // free
    free(input);
    free(output);
    free(h_result);
    cudaFree(d_input);
    cudaFree(d_output);
}