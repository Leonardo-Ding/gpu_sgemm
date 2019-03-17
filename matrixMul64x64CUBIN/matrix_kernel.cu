// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>


extern "C" __global__ void 
maxwell_sgemm_64x64_raggedMn_nt(float *B, float *A, float *C, int ldb, int lda, int ldc, int N, int M, int K, float *alpha, float *beta, float alpha_, float beta_, int flag)
{
	__shared__ float smem[2048];
    
    int tid = threadIdx.x;
    
    float val = (tid > 32) ? A[tid] : B[tid];
    
    smem[tid] = val;
    __syncthreads();
    
    C[tid] = smem[tid];
}

