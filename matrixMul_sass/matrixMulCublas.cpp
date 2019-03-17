// Includes
#include <stdio.h>

// includes, CUDA
#include <cuda.h>
#include <cublas_v2.h>

// local
#include "matrixMulCublas.h"

int matrixMulCublas(float *d_C, const float *d_A, const float *d_B, unsigned int hA, unsigned int wA, unsigned int wB, int nIter)
{
	CUevent start, stop;
	cuEventCreate(&start, CU_EVENT_DEFAULT);
	cuEventCreate(&stop, CU_EVENT_DEFAULT);
	
	// compute reference solution
    printf("\nComputing result using CUBLAS...");
    
    //int nIter = 100;
    
    // CUBLAS version 2.0
    {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        cublasHandle_t handle;

        cublasCreate(&handle);

        //Perform warmup operation with cublas
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, wB, hA, wA, &alpha, (float *)d_B, wB, (float *)d_A, wA, &beta, (float *)d_C, wB);

        // Record the start event
        cuEventRecord(start, NULL);

        for (int j = 0; j < nIter; j++)
        {
            //note cublas is column primary!
            //need to transpose the order
        	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, wB, hA, wA, &alpha, (float *)d_B, wB, (float *)d_A, wA, &beta, (float *)d_C, wB);

        }

        // Record the stop event
		cuEventRecord(stop, NULL);
		cuEventSynchronize(stop);
		
		float msecTotal = 0;
		cuEventElapsedTime(&msecTotal, start, stop);

        // Compute and print the performance
        float msecPerMatrixMul = msecTotal / nIter;
        double flopsPerMatrixMul = 2.0 * (double)hA * (double)wA * (double)wB;
        double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
        printf(
            "CUBLAS Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
            gigaFlops,
            msecPerMatrixMul,
            flopsPerMatrixMul);

        // Destroy the handle
        cublasDestroy(handle);
    }
}
