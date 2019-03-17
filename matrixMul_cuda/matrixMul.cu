/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication as described in Chapter 3
 * of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include <helper_image.h>
#include "common.h"

#define mFetchSmem(ia, ib, ir){ \
	a2[0][ir] = smem[ia + 0];	\
	a2[1][ir] = smem[ia + 8];	\
	b2[0][ir] = smem[ib + 0];	\
	b2[1][ir] = smem[ib + 8];	\
}

#define mRank8x8(ir){ \
	c[0*8+0] += a2[0][ir].x * b2[0][ir].x;	\
	c[1*8+0] += a2[0][ir].y * b2[0][ir].x;	\
	c[2*8+0] += a2[0][ir].z * b2[0][ir].x;	\
	c[3*8+0] += a2[0][ir].w * b2[0][ir].x;	\
	c[4*8+0] += a2[1][ir].x * b2[0][ir].x;	\
	c[5*8+0] += a2[1][ir].y * b2[0][ir].x;	\
	c[6*8+0] += a2[1][ir].z * b2[0][ir].x;	\
	c[7*8+0] += a2[1][ir].w * b2[0][ir].x;	\
	c[0*8+1] += a2[0][ir].x * b2[0][ir].y;	\
	c[1*8+1] += a2[0][ir].y * b2[0][ir].y;	\
	c[2*8+1] += a2[0][ir].z * b2[0][ir].y;	\
	c[3*8+1] += a2[0][ir].w * b2[0][ir].y;	\
	c[4*8+1] += a2[1][ir].x * b2[0][ir].y;	\
	c[5*8+1] += a2[1][ir].y * b2[0][ir].y;	\
	c[6*8+1] += a2[1][ir].z * b2[0][ir].y;	\
	c[7*8+1] += a2[1][ir].w * b2[0][ir].y;	\
	c[0*8+2] += a2[0][ir].x * b2[0][ir].z;	\
	c[1*8+2] += a2[0][ir].y * b2[0][ir].z;	\
	c[2*8+2] += a2[0][ir].z * b2[0][ir].z;	\
	c[3*8+2] += a2[0][ir].w * b2[0][ir].z;	\
	c[4*8+2] += a2[1][ir].x * b2[0][ir].z;	\
	c[5*8+2] += a2[1][ir].y * b2[0][ir].z;	\
	c[6*8+2] += a2[1][ir].z * b2[0][ir].z;	\
	c[7*8+2] += a2[1][ir].w * b2[0][ir].z;	\
	c[0*8+3] += a2[0][ir].x * b2[0][ir].w;	\
	c[1*8+3] += a2[0][ir].y * b2[0][ir].w;	\
	c[2*8+3] += a2[0][ir].z * b2[0][ir].w;	\
	c[3*8+3] += a2[0][ir].w * b2[0][ir].w;	\
	c[4*8+3] += a2[1][ir].x * b2[0][ir].w;	\
	c[5*8+3] += a2[1][ir].y * b2[0][ir].w;	\
	c[6*8+3] += a2[1][ir].z * b2[0][ir].w;	\
	c[7*8+3] += a2[1][ir].w * b2[0][ir].w;	\
	c[0*8+4] += a2[0][ir].x * b2[1][ir].x;	\
	c[1*8+4] += a2[0][ir].y * b2[1][ir].x;	\
	c[2*8+4] += a2[0][ir].z * b2[1][ir].x;	\
	c[3*8+4] += a2[0][ir].w * b2[1][ir].x;	\
	c[4*8+4] += a2[1][ir].x * b2[1][ir].x;	\
	c[5*8+4] += a2[1][ir].y * b2[1][ir].x;	\
	c[6*8+4] += a2[1][ir].z * b2[1][ir].x;	\
	c[7*8+4] += a2[1][ir].w * b2[1][ir].x;	\
	c[0*8+5] += a2[0][ir].x * b2[1][ir].y;	\
	c[1*8+5] += a2[0][ir].y * b2[1][ir].y;	\
	c[2*8+5] += a2[0][ir].z * b2[1][ir].y;	\
	c[3*8+5] += a2[0][ir].w * b2[1][ir].y;	\
	c[4*8+5] += a2[1][ir].x * b2[1][ir].y;	\
	c[5*8+5] += a2[1][ir].y * b2[1][ir].y;	\
	c[6*8+5] += a2[1][ir].z * b2[1][ir].y;	\
	c[7*8+5] += a2[1][ir].w * b2[1][ir].y;	\
	c[0*8+6] += a2[0][ir].x * b2[1][ir].z;	\
	c[1*8+6] += a2[0][ir].y * b2[1][ir].z;	\
	c[2*8+6] += a2[0][ir].z * b2[1][ir].z;	\
	c[3*8+6] += a2[0][ir].w * b2[1][ir].z;	\
	c[4*8+6] += a2[1][ir].x * b2[1][ir].z;	\
	c[5*8+6] += a2[1][ir].y * b2[1][ir].z;	\
	c[6*8+6] += a2[1][ir].z * b2[1][ir].z;	\
	c[7*8+6] += a2[1][ir].w * b2[1][ir].z;	\
	c[0*8+7] += a2[0][ir].x * b2[1][ir].w;	\
	c[1*8+7] += a2[0][ir].y * b2[1][ir].w;	\
	c[2*8+7] += a2[0][ir].z * b2[1][ir].w;	\
	c[3*8+7] += a2[0][ir].w * b2[1][ir].w;	\
	c[4*8+7] += a2[1][ir].x * b2[1][ir].w;	\
	c[5*8+7] += a2[1][ir].y * b2[1][ir].w;	\
	c[6*8+7] += a2[1][ir].z * b2[1][ir].w;	\
	c[7*8+7] += a2[1][ir].w * b2[1][ir].w;	\
}

void
matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
    {
    #pragma omp parallel for num_threads(16)
        for (unsigned int j = 0; j < wB; ++j)
        {
            double sum = 0;

            for (unsigned int k = 0; k < wA; ++k)
            {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }

            C[i * wB + j] = (float)sum;
        }
    }
}

void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

__device__ __forceinline__ void d_rank8x8(float *C, const float *A, const float *B)
{
	float b;
#pragma unroll
	for (int i = 0; i < 8; i++)
	{
		b = B[i];
		C[0*8+i] += A[0]*b;
		C[1*8+i] += A[1]*b;
		C[2*8+i] += A[2]*b;
		C[3*8+i] += A[3]*b;
		C[4*8+i] += A[4]*b;
		C[5*8+i] += A[5]*b;
		C[6*8+i] += A[6]*b;
		C[7*8+i] += A[7]*b;
	}
}
/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
//__launch_bounds__(64, 8) //MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP
extern "C"
__global__ void 
sgemm_kernel_64(float *C, float *A, float *B, int hA, int wA, int wB)
{
	__shared__ float4 smem[2*64 * 2 * 2];
    float c[64] = {0.0f};//thread register initialized zero    
	float4 a1[2], b1[2];	// registers for 1st prefetch from global memory
	//float2 a1[4], b1[4];
	float4 a2[2][2], b2[2][2];	// registers for 2nd prefetch from shared memory
    
	// Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tid = threadIdx.x;
	//int tid_x = tid & 0x7;
	//int tid_y = tid / 8;
	int tid15 = (tid & 15);
	int tid4 = (tid >> 4);

    int aBegin = 64 * by;  
    //int aEnd   = aBegin + hA*(wA - 1); 
    int aStep  = 8 * hA;  

    int bBegin = 64 * bx; 
    int bStep  = 8 * wB;
    int wA8 = wA - 8;

	A += aBegin + tid4*hA + (tid15<<2);
	B += bBegin + tid4*wB + (tid15<<2);
	//AA = A + hA * 4;
	//BB = B + wB * 4; 
    
	// 1st prefetch from global memory
	a1[0] = ld_gbl_cs((const float4 *)(A       ));
	//A += hA * 4;
	a1[1] = ld_gbl_cs((const float4 *)(A + hA*4));
	b1[0] = ld_gbl_cs((const float4 *)(B       ));
	//B += wB * 4;
	b1[1] = ld_gbl_cs((const float4 *)(B + wB*4));
	
	// shared offsets
	int sh_offs = (tid4<<4) + tid15;
	int sh_a = ((tid4<<1) | (tid&1));//tid_y;
	int sh_b = ((tid>>1) & 7) + 256;//tid_x + 256;
    
    // shared memory double buffer
	smem[sh_offs           ] = a1[0];
	smem[sh_offs + 64      ] = a1[1];
	smem[sh_offs      + 256] = b1[0];
	smem[sh_offs + 64 + 256] = b1[1];
    __syncthreads();
        
    // 2nd prefetch from shared memory
	mFetchSmem(sh_a+0*16, sh_b+0*16, 0);//shared memory-->register, memory access

    // main loop
    for (int k = 0; k < wA; k += 8)
    //for (int a = aBegin + aStep, b = bBegin + bStep; a <= aEnd; a += aStep, b += bStep)
    {
		A += aStep;
		B += bStep;
    	sh_offs ^= 128;
		
		// 1st prefetch from global memory
		if (k < wA8)
		{
			a1[0] = ld_gbl_cs((const float4 *)(A       ));
			a1[1] = ld_gbl_cs((const float4 *)(A + hA*4));
			b1[0] = ld_gbl_cs((const float4 *)(B       ));
			b1[1] = ld_gbl_cs((const float4 *)(B + wB*4));
		}
		
		// compute sub matrix
		mFetchSmem(sh_a+1*16, sh_b+1*16, 1);//shared memory-->register, memory access
		mRank8x8(0);							  //register->register, compute
		mFetchSmem(sh_a+2*16, sh_b+2*16, 0);//shared memory-->register, memory access
		mRank8x8(1);							  //register->register, compute
		mFetchSmem(sh_a+3*16, sh_b+3*16, 1);//shared memory-->register, memory access
		mRank8x8(0);							  //register->register, compute
		mFetchSmem(sh_a+4*16, sh_b+4*16, 0);//shared memory-->register, memory access
		mRank8x8(1);							  //register->register, compute
		mFetchSmem(sh_a+5*16, sh_b+5*16, 1);//shared memory-->register, memory access
		mRank8x8(0);							  //register->register, compute
		mFetchSmem(sh_a+6*16, sh_b+6*16, 0);//shared memory-->register, memory access
		mRank8x8(1);							  //register->register, compute
		mFetchSmem(sh_a+7*16, sh_b+7*16, 1);//shared memory-->register, memory access
		
		// shift read index
    	sh_a ^= 128;
    	sh_b ^= 128;
        
        // compute the last sub matrix
		mRank8x8(0);							  //register->register, compute
		
		// shared memory double buffer
		if (k < wA8)
		{
			smem[sh_offs           ] = a1[0];
			smem[sh_offs + 64      ] = a1[1];
			smem[sh_offs      + 256] = b1[0];
			smem[sh_offs + 64 + 256] = b1[1];
		}
		
		mRank8x8(1);							  //register->register, compute
    	__syncthreads();
		
		// 2nd prefetch from shared memory
		if (k < wA8)
		{
			mFetchSmem(sh_a+0*16, sh_b+0*16, 0);//shared memory-->register, memory access
		}
    }
    //__syncthreads();
    
    //if (tid_x == 0 && tid_y == 2 && bx == 0 && by == 0)
    //{
    //	printf("c[0] : %f\n", c[0]);
    //}

	// store the 8*8 result to shared
#if 1
	int C_index = wB*64*by + 64*bx + tid4*8*wB + tid15*4;
	int tid31 = tid & 31;
	int cs = ((tid31/2)&7) + (tid31&1)*16 + tid4*32;
#pragma unroll
	for (int i = 0; i < 8; i++)
	{
		// reform C to shared memory
		smem[cs + 0] = make_float4(c[i*8+0], c[i*8+1], c[i*8+2], c[i*8+3]);
		smem[cs + 8] = make_float4(c[i*8+4], c[i*8+5], c[i*8+6], c[i*8+7]);
		
		//if (i == 0 && tid_x == 0 && tid_y == 2 && bx == 0 && by == 0)
		//	printf("%d: %f\n", cs, smem[cs + 0].x);
		
		//if (C_index + (i + (i/4)*32 + 4)*wB >= 1024*1024 - 1)
		//	printf("%d %d %d %d\n", bx, by, tid, (i + (i/4)*32 + 4)*wB);
		
		// coalesing access
		st_gbl_cs((const float4 *)(C + C_index + (i + (i/4)*28 + 0)*wB), smem[tid15 + (tid4<<5)+ 0] );
		st_gbl_cs((const float4 *)(C + C_index + (i + (i/4)*28 + 4)*wB), smem[tid15 + (tid4<<5)+16] );
	}
#else
    int C_index = wB * 64 * by + tid_y * 8 * wB + 64 * bx + tid_x * 8;    
#pragma unroll
	for (int i = 0; i < 8; i++)
	{
		C[C_index + i * wB + 0] = c[i * 8 + 0];
		C[C_index + i * wB + 1] = c[i * 8 + 1];
		C[C_index + i * wB + 2] = c[i * 8 + 2];
		C[C_index + i * wB + 3] = c[i * 8 + 3];
		C[C_index + i * wB + 4] = c[i * 8 + 4];
		C[C_index + i * wB + 5] = c[i * 8 + 5];
		C[C_index + i * wB + 6] = c[i * 8 + 6];
		C[C_index + i * wB + 7] = c[i * 8 + 7];
	}
#endif
}

void constantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}


/**
 * Program main
 */
int main(int argc, char **argv)
{
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    int devID = 0;
	cudaSetDevice(devID);

    int N = 1024;
    int NITER = 10000;
    if (argc == 2)
    {
    	N = atoi(argv[1]);
    }
    else if (argc == 3)
    {
    	N = atoi(argv[1]);
    	NITER = atoi(argv[2]);
    }
	
	dim3 dimsA(N, N, 1);
	dim3 dimsB(N, N, 1);
    
    //if (dimsA.x != dimsB.y)
    //{
    //    printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
    //           dimsA.x, dimsB.y);
    //    exit(EXIT_FAILURE);
    //}

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);
	
	// set seed for rand()
    srand(2006);
	// Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    // Initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    
    //checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C = (float *) malloc(mem_size_C);

    checkCudaErrors(cudaMalloc((void **)&d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void **)&d_B, mem_size_B));
    checkCudaErrors(cudaMalloc((void **)&d_C, mem_size_C));

    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

    // Setup execution parameters
    dim3 threads(64);//32x32 
    dim3 grid(dimsB.x/64, dimsA.y/64);

    //Performs warmup operation using matrixMul CUDA kernel
    sgemm_kernel_64<<< grid, threads >>>(d_C, d_A, d_B, dimsA.y, dimsA.x, dimsB.x);

    printf("done\n");

    cudaDeviceSynchronize();

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    // Execute the kernel
    int nIter = NITER;

    for (int j = 0; j < nIter; j++)
    {
        sgemm_kernel_64<<< grid, threads >>>(d_C, d_A, d_B, dimsA.y, dimsA.x, dimsB.x);
    }

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        threads.x * threads.y);

    // Copy result from device to host
    checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

    printf("Computing result using host CPU...\n");
	float *A_T = (float *)malloc(mem_size_A);
	for (int i = 0; i < dimsA.x; i++){
		for (int j = 0; j < dimsA.y; j++){
			A_T[i*dimsA.y + j] = h_A[j*dimsA.x+i];
		}
	}
    float *reference = (float *)malloc(mem_size_C);
    matrixMulCPU(reference, A_T, h_B, (unsigned int)dimsA.y, (unsigned int)dimsA.x, (unsigned int)dimsB.x);
    
	//float c08 = 0;
	//for (int i = 0; i < dimsB.x; i++)
	//{
	//	c08 += A_T[i] * h_B[i * dimsB.x + 32];
	//}
	//printf("c08: %f\n", c08);
	
	//bool resCUBLAS = sdkCompareL2fe(reference, h_C, mem_size_C, 1.0e-4f);
    
	//if (resCUBLAS != true)
    //{
	//	printf("COMPARE ERROR!\n");
	//}
	for (int i = 0; i < dimsA.y; i++)
	{
		for (int j = 0; j < dimsB.x; j++)
		{
			if (fabs(reference[i * dimsB.x + j] - h_C[i * dimsB.x + j]) > 1e-3)
			{
				printf("index (%d, %d)---- reference %f, gpu_data: %f\n", i, j, reference[i * dimsB.x + j], h_C[i * dimsB.x + j]);
				return 0; //exit(1);
			}
		}
	}


	//printf("done.\n");
	//for (int i = 0; i < 100; i++)
    //    printf("h_C is %f \t", h_C[i]);
    //printf("\n");

    //printf("Checking computed result for correctness: ");
    //bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    //double eps = 1.e-6 ; // machine zero

    //for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++)
    //{
    //    double abs_err = fabs(h_C[i] - (dimsA.x * valB));
    //    double dot_length = dimsA.x;
    //    double abs_val = fabs(h_C[i]);
    //    double rel_err = abs_err/abs_val/dot_length ;

    //    if (rel_err > eps)
    //    {
    //        printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], dimsA.x*valB, eps);
    //        correct = false;
    //    }
    //}

    //printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
