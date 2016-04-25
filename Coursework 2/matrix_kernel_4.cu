#include "matrix_kernel_4.h"
#include <cuda_runtime.h>
#include <math.h>

static __global__ void normaliseRow(int pivotPos, float *d_Matrix);
static __global__ void scaleAndSubtract(int pivotPos, int width, float *d_Matrix);
static __global__ void scaleAndSubtract2(int pivotPos, int width, float *d_Matrix);



/*

This is the second improvement of the GPU implementation.

* Uses shared memory whenever possible

* Give each thread more work by dividing into segments

*/

// -- Controller function for device function
// -- Max matrix width = 1024
void M4_Controller(float* d_Matrix, float* h_Matrix, int height, int width){

	// -- Iterate through all rows
	for (int row_ID = 0; row_ID < height; row_ID++){

		// Each row normalisation has fewer non-zero elements to normalise
		dim3 blocksPerGrid(1);
		dim3 threadsPerBlock(width - row_ID);

		// -- Define pivot position and value
		int pivotPos = row_ID * (width + 1);

		// -- Normalise row
		normaliseRow << <blocksPerGrid, threadsPerBlock, sizeof(float)* threadsPerBlock.x >> >(pivotPos, d_Matrix);

		// -- Wait for kernel to finish normalising row
		cudaDeviceSynchronize();

		// -- Each row will have few non-zero elements to remove
		dim3 rows(height - row_ID);			// Number of rows
		dim3 elements(width - row_ID);		// Number of elements in row

		// -- If pivot is on last row then there's nothing else to eliminate
		//if (rows.x == 1) return;

		//-- Call kernel to scale and subtract rows
		scaleAndSubtract << < rows, elements, sizeof(float)* elements.x >> >(pivotPos, width, d_Matrix);

		// -- Wait for kernel to finish eliminating rows
		cudaDeviceSynchronize();
	}

	// -- Go through all rows starting from second
	for (int row_ID = 1; row_ID < height; row_ID++){

		// -- Define pivot position and value
		int pivotPos = row_ID * (width + 1);

		// -- Each row will have few non-zero elements to remove
		dim3 blocksPerGrid(row_ID + 1);
		dim3 threadsPerBlock(width - row_ID);

		//-- Call kernel to scale and subtract rows
		scaleAndSubtract2 << < blocksPerGrid, threadsPerBlock, sizeof(float)* threadsPerBlock.x >> >(pivotPos, width, d_Matrix);

		// -- Wait for kernel to finish eliminating rows
		cudaDeviceSynchronize();
	}
}

// -- Normalise row relative to pivot value
__global__ void normaliseRow(int pivotPos, float *d_Matrix){

	// -- Get threadID
	int tid = threadIdx.x;

	// -- Declare shared memory dynamically
	extern __shared__ float s_Row[];

	// -- Move from global to shared memory
	s_Row[tid] = d_Matrix[pivotPos + tid];

	// -- Make sure all threads have loaded to shared memory
	__syncthreads();

	// -- Normalise element relative to pivot value
	d_Matrix[pivotPos + tid] = s_Row[tid] / s_Row[0];

	// -- To guard against the next case, but not sure if completely needed *******
	__syncthreads();
}

// -- Normalise row relative to pivot value downwards
__global__ void scaleAndSubtract(int pivotPos, int width, float *d_Matrix){

	// -- Get threadID
	int tid = threadIdx.x;

	// -- Get blockID
	int bid = blockIdx.x;

	// -- Declare shared memory dynamically
	extern __shared__ float s_pivotRow[];

	// -- Move from global to shared memory
	s_pivotRow[tid] = d_Matrix[pivotPos + tid];

	// -- Make sure all threads have loaded to shared memory.
	__syncthreads();

	// -- Discard bid 0 threads so it doesnt overwrite pivotRow
	if (bid == 0) return;

	// -- Find coefficient to scale row with
	float coeff = d_Matrix[pivotPos + (width * bid)];

	// -- Update elements in row by subtracting elements with coeff
	d_Matrix[pivotPos + (width * bid) + tid] = d_Matrix[pivotPos + (width * bid) + tid]
		- (coeff * s_pivotRow[tid]);
}

// -- Normalise row relative to pivot value upwards
__global__ void scaleAndSubtract2(int pivotPos, int width, float *d_Matrix){

	// -- Don't overwrite pivot row
	if (blockIdx.x == 0) return;

	// -- Get blockID
	int bid = blockIdx.x;

	// -- Get threadID
	int tid = threadIdx.x;

	// -- Declare shared memory dynamically
	extern __shared__ float s_pivotRow[];

	// -- Move from global to shared memory
	s_pivotRow[tid] = d_Matrix[pivotPos + tid];

	// -- Make sure all threads have loaded to shared memory.
	__syncthreads();

	// -- Find coefficient to scale row with
	float coeff = d_Matrix[pivotPos - (width * bid)];

	// -- Update elements in row by subtracting elements with coeff
	d_Matrix[pivotPos - (width * bid) + tid] = d_Matrix[pivotPos - (width * bid) + tid]
		- (coeff * s_pivotRow[tid]);
}