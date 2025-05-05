#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define THREAD_PER_BLOCK 256

__global__ void reduce0(float *d_input,float *d_output){
    float* input_begin = d_input+blockIdx.x*blockDim.x;
    
    for(int i =1;i<blockDim.x;i*=2){
        if(threadIdx.x%(i*2) == 0){
            input_begin[threadIdx.x] += input_begin[threadIdx.x+i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        d_output[blockIdx.x] = input_begin[0];
    }
}

bool check(float *out,float *res,int n){
    for(int i=0;i<n;i++){
        if(abs(out[i]-res[i])>0.005)
            return false;
    }
    return true;
}

int main(){
    const int N = 32*1024*1024;
    int block_num = N/THREAD_PER_BLOCK;

    // original data vector
    float *input = (float*)malloc(N*sizeof(float));
    float *d_input;
    cudaMalloc((void**)&d_input,N *sizeof(float));
    // assign random numbers
    for(int i=0;i<N;i++){
        input[i] = 1.0;
    }

    // save gpu result vector.output[blockIdx.x] = this block sum
    float *output = (float*)malloc((N/THREAD_PER_BLOCK)*sizeof(float));
    float *d_output;
    cudaMalloc((void**)&d_output,(N/THREAD_PER_BLOCK) *sizeof(float));

    // save cpu result vector
    float *h_result = (float*)malloc((N/THREAD_PER_BLOCK)*sizeof(float));

    // cpu reduce
    for(int i=0;i<block_num;i++){
        float cur = 0;
        for(int j=0;j<THREAD_PER_BLOCK;j++){
            cur+=input[i*THREAD_PER_BLOCK+j];
        }
        h_result[i]=cur;
    }

    // gpu reduce
    cudaMemcpy(d_input,input,N*sizeof(float),cudaMemcpyHostToDevice);
    dim3 Grid(N/THREAD_PER_BLOCK,1);
    dim3 Block(THREAD_PER_BLOCK,1);
    reduce0<<<Grid,Block>>>(d_input,d_output);
    cudaMemcpy(output,d_output,(N/THREAD_PER_BLOCK) *sizeof(float),cudaMemcpyDeviceToHost);
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