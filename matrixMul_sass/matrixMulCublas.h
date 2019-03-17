#ifndef _MATRIXMUL_CUBLAS_H_
#define _MATRIXMUL_CUBLAS_H_

int matrixMulCublas(float *d_C, const float *d_A, const float *d_B, unsigned int hA, unsigned int wA, unsigned int wB, int nIter);

#endif
