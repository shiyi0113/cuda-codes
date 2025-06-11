#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define THREAD_PER_BLOCK 256
bool check(float *out,float *res,int n);
template<unsigned int BLOCKSIZE>
__device__ void warpReduceC(volatile float* cache, unsigned int tid){
    if (BLOCKSIZE >= 64)cache[tid]+=cache[tid+32];
    if (BLOCKSIZE >= 32)cache[tid]+=cache[tid+16];
    if (BLOCKSIZE >= 16)cache[tid]+=cache[tid+8];
    if (BLOCKSIZE >= 8)cache[tid]+=cache[tid+4];
    if (BLOCKSIZE >= 4)cache[tid]+=cache[tid+2];
    if (BLOCKSIZE >= 2)cache[tid]+=cache[tid+1];
}

// each block proceses twice the amount of data.
template<unsigned int BLOCKSIZE,unsigned int NUM_PER_BLOCK,unsigned int NUM_PER_THREAD>
__global__ void reduce7(float *d_input,float *d_output){
    __shared__ float s_input[BLOCKSIZE];
    float* input_begin = d_input+blockIdx.x *NUM_PER_BLOCK;
    s_input[threadIdx.x] = 0;
    for(int i=0;i<NUM_PER_THREAD;i++){
        s_input[threadIdx.x] += input_begin[threadIdx.x + blockDim.x * i];
    }
    __syncthreads();
    
    if (BLOCKSIZE >= 512) {
        if (threadIdx.x < 256) { 
            s_input[threadIdx.x] += s_input[threadIdx.x + 256]; 
        } 
        __syncthreads(); 
    }
    if (BLOCKSIZE >= 256) {
        if (threadIdx.x < 128) { 
            s_input[threadIdx.x] += s_input[threadIdx.x + 128]; 
        } 
        __syncthreads(); 
    }
    if (BLOCKSIZE >= 128) {
        if (threadIdx.x < 64) { 
            s_input[threadIdx.x] += s_input[threadIdx.x + 64]; 
        } 
        __syncthreads(); 
    }
    if (threadIdx.x < 32) warpReduceC<BLOCKSIZE>(s_input, threadIdx.x);
    if(threadIdx.x == 0){
        d_output[blockIdx.x] = s_input[0];
    }
}

void reduce7_multi_Add(){
    const int N = 32*1024*1024;
    
    const int block_num = 1024;
    const int NUM_PER_BLOCK = N/block_num;
    const int NUM_PER_THREAD = NUM_PER_BLOCK/THREAD_PER_BLOCK;
    
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
    reduce7<THREAD_PER_BLOCK,NUM_PER_BLOCK,NUM_PER_THREAD><<<Grid,Block>>>(d_input,d_output);
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