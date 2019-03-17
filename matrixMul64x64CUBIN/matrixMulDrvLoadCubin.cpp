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
#include <builtin_types.h>
#include <drvapi_error_string.h>

using namespace std;

// Variables
CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction matrixMul_kernel;
CUresult error;
float *h_A;
float *h_B;
float *h_C;
CUdeviceptr d_A;
CUdeviceptr d_B;
CUdeviceptr d_C;
//bool noprompt = false;
#define CUBIN_FILE "../cublas64x64.cubin"
#define PTX_FILE "matrixMul.ptx"
#define KERNEL_NAME "maxwell_sgemm_64x64_raggedMn_nt"

bool inline
findModulePath(const char *module_file, string &module_path, char **argv, string &ptx_source)
{
    char *actual_path = sdkFindFilePath(module_file, argv[0]);

    if (actual_path)
    {
        module_path = actual_path;
    }
    else
    {
        printf("> findModulePath file not found: <%s> \n", module_file);
        return false;
    }

    if (module_path.empty())
    {
        printf("> findModulePath could not find file: <%s> \n", module_file);
        return false;
    }
    else
    {
        printf("> findModulePath found file at <%s>\n", module_path.c_str());

        if (module_path.rfind(".ptx") != string::npos)
        {
            FILE *fp = fopen(module_path.c_str(), "rb");
            fseek(fp, 0, SEEK_END);
            int file_size = ftell(fp);
            char *buf = new char[file_size+1];
            fseek(fp, 0, SEEK_SET);
            fread(buf, sizeof(char), file_size, fp);
            fclose(fp);
            buf[file_size] = '\0';
            ptx_source = buf;
            delete[] buf;
        }

        return true;
    }
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

/**
 * Program main
 */
int main(int argc, char **argv)
{
    printf("[Matrix Multiply Using CUDA Driver API] - Starting...\n");

    int devID = 0;
	//cudaSetDevice(devID);
	dim3 dimsA(1024, 1024, 1);
	dim3 dimsB(1024, 1024, 1);
	
	printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);
	
	// set seed for rand()
    srand(2006);

	// Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    h_B = (float *)malloc(mem_size_B);
	// Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    h_C = (float *) malloc(mem_size_C);

	// Initialize
    cuInit(0);
	// pick up device with zero ordinal (default, or devID)
    error = cuDeviceGet(&cuDevice, devID);

    if (error != CUDA_SUCCESS)
    {
        exit(-1);
    }

	// Create context
    error = cuCtxCreate(&cuContext, 0, cuDevice);

    if (error != CUDA_SUCCESS)
    {
        exit(-1);
    }

	// first search for the module path before we load the results
    string module_path, ptx_source;

    if (!findModulePath(CUBIN_FILE, module_path, argv, ptx_source))
    {
    	printf("> findModulePath could not find <matrixMul> ptx or cubin\n");
        exit(-1);
    }
    else
    {
        printf("> initCUDA loading module: <%s>\n", module_path.c_str());
    }

	error = cuModuleLoad(&cuModule, module_path.c_str());
	
	if (error != CUDA_SUCCESS)
    {
        exit(-1);
	}

	// Get function handle from module
    error = cuModuleGetFunction(&matrixMul_kernel, cuModule, KERNEL_NAME);
	if (error != CUDA_SUCCESS)
	{
		printf("cuModuleGetFunction failed!\n");
	}

	// Initialize input vectors
	randomInit(h_A, size_A);
    randomInit(h_B, size_B);

	// Allocate vectors in device memory
    error = cuMemAlloc(&d_A, mem_size_A);
    if (error != CUDA_SUCCESS)
    {
        exit(-1);
    }

	error = cuMemAlloc(&d_B, mem_size_B);
    if (error != CUDA_SUCCESS)
    {
        exit(-1);
    }

    error = cuMemAlloc(&d_C, mem_size_C);
    if (error != CUDA_SUCCESS)
    {
        exit(-1);
    }

	// Copy vectors from host memory to device memory
    error = cuMemcpyHtoD(d_A, h_A, mem_size_A);
    if (error != CUDA_SUCCESS)
    {
        exit(-1);
    }

    error = cuMemcpyHtoD(d_B, h_B, mem_size_B);
    if (error != CUDA_SUCCESS)
    {
        exit(-1);
    }

	// Grid/Block configuration
    int threadsPerBlock_x = 64;
	int threadsPerBlock_y = 1;
    int blocksPerGrid_x   = (dimsB.x) / (64);
	int blocksPerGrid_y   = (dimsA.y) / (64);
	int lda = dimsA.y;
	int ldb = dimsB.x;
	int ldc = dimsC.x;
	int N = dimsA.y;
	int M = dimsB.x;
	int K = dimsA.x;
	float *d_alpha = NULL;
	float *d_beta = NULL;
	float alpha = 1.0f;
	float beta = 0.0f;
	int flag = 0;
	
    void *args[] = { &d_B, &d_A, &d_C, &ldb, &lda, &ldc, &N, &M, &K, &d_alpha, &d_beta, &alpha, &beta, &flag };

    // Launch the CUDA kernel
    error = cuLaunchKernel(matrixMul_kernel,  blocksPerGrid_x, blocksPerGrid_y, 1,
                           threadsPerBlock_x, threadsPerBlock_y, 1,
                           0,
                           NULL, args, NULL);
    if (error != CUDA_SUCCESS)
    {
        exit(-1);
    }

	printf("done\n");

	error = cuCtxSynchronize();

    if (error != CUDA_SUCCESS)
    {
        exit(-1);
    }

	CUevent start, stop;
	cuEventCreate(&start, CU_EVENT_DEFAULT);
	cuEventCreate(&stop, CU_EVENT_DEFAULT);

	cuEventRecord(start, NULL);

	// Execute the kernel
    int nIter = 10;

    for (int j = 0; j < nIter; j++) {
	// Launch the CUDA kernel
    error = cuLaunchKernel(matrixMul_kernel,  blocksPerGrid_x, blocksPerGrid_y, 1,
                           threadsPerBlock_x, threadsPerBlock_y, 1,
                           0,
                           NULL, args, NULL);

	}

	// stop and destroy timer
    cuEventRecord(stop, NULL);
	cuEventSynchronize(stop);
	
	float msecTotal = 0.0f;
	cuEventElapsedTime(&msecTotal, start, stop);

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        threadsPerBlock_x*threadsPerBlock_y );

    // Copy result from device to host
    cuMemcpyDtoH(h_C, d_C, mem_size_C);

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


	// Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);

	cuCtxDestroy(cuContext);
    return 0;
}
