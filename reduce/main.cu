#include "reduce_v0_global_memory.cuh"
#include "reduce_v1_shared_memory.cuh"
#include "reduce_v2_no_divergence_branch.cuh"
#include "reduce_v3_no_bank_conflict.cuh"
#include "reduce_v4_thread_coarsening_A.cuh"
#include "reduce_v4_thread_coarsening_B.cuh"
#include "reduce_v5_unroll_last_warp.cuh"
#include "reduce_v6_completely_unroll.cuh"
#include "reduce_v7_multi_add.cuh"
#include "reduce_v8_warp_shuffle.cuh"
bool check(float *out,float *res,int n){
    for(int i=0;i<n;i++){
        if(abs(out[i]-res[i])>0.005)
            return false;
    }
    return true;
}
int main()
{
    reduce0_naive();
    reduce1_shared_memory();
    reduce2_no_rivergence_branch();
    reduce3_no_bank_conflict();
    reduce4A_thread_coarsening();
    reduce4B_thread_cosrsening();
    reduce5_unroll_last_warp();
    reduce6_completely_unroll();
    reduce7_multi_Add();
    reduce8_warp_shuffle();
}
