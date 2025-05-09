#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32

template<unsigned int BLOCKSIZE>
__device__ __forceinline__ float warpReduceSum(float sum){
    sum += __shfl_down_sync(0xffffffff,sum,16);
    sum += __shfl_down_sync(0xffffffff,sum,8);
    sum += __shfl_down_sync(0xffffffff,sum,4);
    sum += __shfl_down_sync(0xffffffff,sum,2);
    sum += __shfl_down_sync(0xffffffff,sum,1);
    return sum;
}

// each block proceses twice the amount of data.
template<unsigned int BLOCKSIZE,unsigned int NUM_PER_BLOCK,unsigned int NUM_PER_THREAD>
__global__ void reduce8(float *d_input,float *d_output){
    float sum = 0.0f;
    float* input_begin = d_input+blockIdx.x *NUM_PER_BLOCK;
    for(int i=0;i<NUM_PER_THREAD;i++){
        sum += input_begin[threadIdx.x + blockDim.x * i];
    }
    sum = warpReduceSum<BLOCKSIZE>(sum);

    __shared__ float warpLevelSum[WARP_SIZE];
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    if(laneId == 0){
        warpLevelSum[warpId] = sum;
    }
    __syncthreads();

    if(warpId == 0){
        sum = (laneId < blockDim.x / WARP_SIZE) ? warpLevelSum[laneId] : 0.0f;
        sum = warpReduceSum<BLOCKSIZE>(sum);
    }

    if(threadIdx.x == 0){
        d_output[blockIdx.x] = sum;
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
    reduce8<THREAD_PER_BLOCK,NUM_PER_BLOCK,NUM_PER_THREAD><<<Grid,Block>>>(d_input,d_output);
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