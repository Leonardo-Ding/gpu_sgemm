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

// local
#include "matrixMulDrvLoadCubin.h"

using namespace std;

//bool noprompt = false;
#define CUBIN_FILE "../matrixMul_64x64.cubin"
#define PTX_FILE "matrixMul.ptx"
#define KERNEL_NAME "sgemm_kernel_64"

bool inline
findModulePath(const char *module_file, string &module_path, string &ptx_source)
{
    const char *actual_path = module_file;

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

/**
 * Program main
 */
int matrixMulDrvLoadCubin(float *d_C_, const float *d_A_, const float *d_B_, unsigned int hA, unsigned int wA, unsigned int wB, int nIter)
{
	CUresult error;
	
	CUmodule cuModule;
	CUfunction matrixMul_kernel;
	
	// cudevptr
	CUdeviceptr d_A = (CUdeviceptr)d_A_;
	CUdeviceptr d_B = (CUdeviceptr)d_B_;
	CUdeviceptr d_C = (CUdeviceptr)d_C_;

    printf("\n[Matrix Multiply Using CUDA Driver API Load CUBIN] - Starting...\n");

	// first search for the module path before we load the results
    string module_path, ptx_source;

    if (!findModulePath(CUBIN_FILE, module_path, ptx_source))
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

	// Grid/Block configuration
    int threadsPerBlock_x = 64;
	int threadsPerBlock_y = 1;
    int blocksPerGrid_x   = (wB) / (64);
	int blocksPerGrid_y   = (hA) / (64);
    void *args[] = { &d_C, &d_A, &d_B, &hA, &wA, &wB };

    // Launch the CUDA kernel
    error = cuLaunchKernel(matrixMul_kernel,  blocksPerGrid_x, blocksPerGrid_y, 1,
                           threadsPerBlock_x, threadsPerBlock_y, 1,
                           0,
                           NULL, args, NULL);
    if (error != CUDA_SUCCESS)
    {
        exit(-1);
    }

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
    //int nIter = 100;

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
    double flopsPerMatrixMul = 2.0 * (double)wA * (double)hA * (double)wB;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        threadsPerBlock_x*threadsPerBlock_y );
    return 0;
}
