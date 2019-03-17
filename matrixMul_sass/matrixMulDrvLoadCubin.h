#ifndef _MATRIXMUL_DRV_LOAD_CUBIN_H_
#define _MATRIXMUL_DRV_LOAD_CUBIN_H_

int matrixMulDrvLoadCubin(float *d_C_, const float *d_A_, const float *d_B_, unsigned int hA, unsigned int wA, unsigned int wB, int nIter);

#endif
