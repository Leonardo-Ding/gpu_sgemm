This is a simple demo about how to optimize the gemm on Nvidia gpu platform with using sass level optimization trick.
Enviroment: Ubuntu 16.04/14.04(other Linux os may be OK)

Software required: 
    cuda 8.0 
    cmake 2.8 or above
    perl

We first write a cuda-c level code in matrixMul_cuda dir. Firstly, this is a 'NT'(C = A*B, A is a column-majored matrix, B is a row-majored matrix) format doing gemm operation. Secondly, ouput C is 64x64 tile format, input A and B is 64x8 format, and we use 64 cuda threads to do it.
some small tricks are listed as followed:   
1. 8x8 loop unroll technique to bring Instruction Level Parallel gain in our code,
2. wide 128 load/store instructions to enhance load/store efficiency, 
3. double buffer trick to remove one unneccesary __syncthreads() in for loop, 
4. data prefetch to improve pipeline efficiency,
5. reform C to shared memory before storing data to global memory.

However, the cuda-c code is not optimized very well in TLP through nvcc compilier as we must control gpu register usage below 128 to keep theoretical occupancy 25%. So we decide to optimize the code in sass level code.
Detailed register distribution is not listed here and this process is not so easy. Luckily, we have a good assembler to help us. We use NervanaSystems/maxas for reference and modify it to run code successfully on Nvidia pascal architecture gpu and cuda 8.0 environment.

Note: consider this is only a simple demo to show how to optimize gpu code in sass, some restriction in code is listed here:
1. A and B matrix row and column size must be 64 or 64 multiples,
2. This is only a version of matrix multiply for C = AxB and not a complete gemm version for C = alpha*A*B+beta*C,
3. You must write 128x64, 128x128 or others to have better performance beyond Cublas for large input matrix size, though our code has a better performance than cublas when doing 1024x1024 matrix multiply tested on GTX1080.
4. Sorry to say our document is not detailed, anybody who interested in this project can contact me.

Finally, thank Scott Gray very much for your wonderful code.