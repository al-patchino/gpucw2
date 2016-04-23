#include <stdio.h>
#include <cuda_runtime.h>

void M1_Controller(float* d_Matrix, float* h_Matrix, int height, int width);
__global__ void normaliseRow(int pivotPos, float *d_Matrix);
__global__ void scaleAndSubtract(int row2_ID, int pivotPos, int width, float *d_Matrix);
__global__ void scaleAndSubtract2(int row2_ID, int pivotPos, int width, float *d_Matrix);

