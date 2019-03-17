// Includes
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cstring>

// includes, project
#include <helper_functions.h>
#include <helper_cuda.h>

// includes, CUDA
#include <cuda.h>
#include <cublas_v2.h>
#include <builtin_types.h>
#include <drvapi_error_string.h>

// includes. local
#include "matrixMulCublas.h"
#include "matrixMulDrvLoadCubin.h"

using namespace std;

#define M (512*2)
#define N (512*2)
#define K (512*2)
#define ITERS 10000

// Variables
CUdevice cuDevice;
CUcontext cuContext;
CUresult error;
float *h_A;
float *h_B;
CUdeviceptr d_A;
CUdeviceptr d_B;
CUdeviceptr d_C;

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;

    for (j = 0; j < height; j++)
    {
        if (error_count < iListLength)
        {
            printf("\n  Row %d:\n", j);
        }

        for (i = 0; i < width; i++)
        {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);

            if (fDiff > fListTol)
            {
                if (error_count < iListLength)
                {
                    printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }

                error_count++;
            }
        }
    }

    printf(" \n  Total Errors = %d\n", error_count);
}

void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
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
        
        if (i % 500 == 0) printf("-----------CPU COMPUTE LINE: %d\n", i);
    }
}

/**
 * Program main
 */
int main(int argc, char **argv)
{
    printf("[Matrix Multiply Using CUDA Driver API] - Starting...\n");

    int devID = 0;
	//cudaSetDevice(devID);
	dim3 dimsA(M, K, 1);
	dim3 dimsB(K, N, 1);
	int nIters = ITERS;
	
	printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);
	
	// set seed for rand()
    srand(2006);
    cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1);

	// Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    h_B = (float *)malloc(mem_size_B);
	// Allocate host matrix C
    dim3 dimsC(dimsA.y, dimsB.x, 1);
    unsigned int size_C = dimsC.y * dimsC.x;
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);

	// Initialize
    cuInit(0);
	// pick up device with zero ordinal (default, or devID)
    error = cuDeviceGet(&cuDevice, devID);
    if (error != CUDA_SUCCESS) exit(-1);

	// Create context
    error = cuCtxCreate(&cuContext, 0, cuDevice);
    if (error != CUDA_SUCCESS) exit(-1);

	// Initialize input vectors
	randomInit(h_A, size_A);
    randomInit(h_B, size_B);

	// Allocate vectors in device memory
    error = cuMemAlloc(&d_A, mem_size_A);
    if (error != CUDA_SUCCESS) exit(-1);

	error = cuMemAlloc(&d_B, mem_size_B);
    if (error != CUDA_SUCCESS) exit(-1);

    error = cuMemAlloc(&d_C, mem_size_C);
    if (error != CUDA_SUCCESS) exit(-1);

	// Copy vectors from host memory to device memory
    error = cuMemcpyHtoD(d_A, h_A, mem_size_A);
    if (error != CUDA_SUCCESS) exit(-1);
    error = cuMemcpyHtoD(d_B, h_B, mem_size_B);
    if (error != CUDA_SUCCESS) exit(-1);

	// DrvLoadCubin
    float *h_Cubin = (float *) malloc(mem_size_C);
    matrixMulDrvLoadCubin((float *)d_C, (float *)d_A, (float *)d_B, dimsA.y, dimsA.x, dimsB.x, nIters);
    cuMemcpyDtoH(h_Cubin, d_C, mem_size_C);

    // CUBLAS version 2.0
    float *h_CUBLAS = (float *)malloc(mem_size_C);
    matrixMulCublas((float *)d_C, (float *)d_A, (float *)d_B, dimsA.y, dimsA.x, dimsB.x, nIters);
    cuMemcpyDtoH(h_CUBLAS, d_C, mem_size_C);
	
	// CPU
	printf("\nComputing result using host CPU 16 threads...\n");
	float *A_T = (float *)malloc(mem_size_A);
	for (int i = 0; i < dimsA.x; i++)
		for (int j = 0; j < dimsA.y; j++)
			A_T[i*dimsA.y + j] = h_A[j*dimsA.x+i];
    float *reference = (float *)malloc(mem_size_C);
    matrixMulCPU(reference, A_T, h_B, (unsigned int)dimsA.y, (unsigned int)dimsA.x, (unsigned int)dimsB.x);
    
    
	// check result (CUBIN)
	printf("\n----------Check Cubin---------\n");
    bool resCUBIN = sdkCompareL2fe(reference, h_Cubin, size_C, 1.0e-6f);
    if (resCUBIN != true)
    {
        printDiff(reference, h_Cubin, dimsC.x, dimsC.y, 100, 1.0e-5f);
    }
    printf("Comparing Cubin Matrix Multiply with CPU results: %s\n", (true == resCUBIN) ? "PASS" : "FAIL");
    
    // check result (CUBLAS)
	printf("\n----------Check Cublas---------\n");
    bool resCUBLAS = sdkCompareL2fe(reference, h_CUBLAS, size_C, 1.0e-6f);
    if (resCUBLAS != true)
    {
        printDiff(reference, h_CUBLAS, dimsC.x, dimsC.y, 100, 1.0e-5f);
    }
    printf("Comparing CUBLAS Matrix Multiply with CPU results: %s\n", (true == resCUBLAS) ? "PASS" : "FAIL");

	// Clean up memory
    free(h_A);
    free(h_B);
    free(h_Cubin);
    free(h_CUBLAS);
    free(reference);
    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);

	cuCtxDestroy(cuContext);
    return 0;
}
