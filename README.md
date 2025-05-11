# CUDA-Codes

This repository contains code examples and optimization practices from my CUDA programming learning journey. It primarily focuses on the implementation of CUDA core algorithms and performance optimization techniques.

## Project Overview

CUDA-Codes aims to deepen understanding of GPU parallel computing principles and techniques through practicing various CUDA programming patterns and optimization strategies. The code in this repository is organized by different algorithms and optimization stages.

## Current Progress

### Completed

- **Reduce Operation Optimization**:
  - Basic implementation
  - Shared memory optimization
  - Solving branch divergence issues
  - Bank conflict elimination
  - Thread coarsening techniques
  - Last warp unrolling
  - Complete loop unrolling
  - Multi-add optimization
  - Warp shuffle implementation

### In Progress

- **SGEMM (Single Precision General Matrix Multiplication) Optimization**:
  - Basic implementation
  - Shared memory optimization
  - Thread coarsening techniques
  - Using float4

## References

The code implementations in this repository are based on the following resources:

- [深入浅出GPU优化系列：reduce优化](https://zhuanlan.zhihu.com/p/426978026)
- [cuda 入门的正确姿势：how-to-optimize-gemm](https://zhuanlan.zhihu.com/p/478846788)