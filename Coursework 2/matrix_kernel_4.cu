#include "matrix_kernel_4.h"
#include <cuda_runtime.h>
#include <math.h>

static __global__ void normaliseRow(int pivotPos, float *um_Matrix, float pivotVal, int width);
static __global__ void scaleAndSubtract(int pivotPos, int width, float *um_Matrix);
static __global__ void scaleAndSubtract2(int pivotPos, int width, float *um_Matrix);


#define SEGMENT_LENGTH 128

/*

This is the second improvement of the GPU implementation.

	*	Trying out Unified Memory (had to re-compile for 64 bit architecture instead)
	
	*	This means I can pass coeff & pivotVal by value to reduce reads to GM since memory
		is visible to both device and host
	
	*	Shared Memory usage removed from normaliseRow since pivotVal is being passed in function.
		In this case, shared memory would offer no advantage, since its 1 GM read and 1 GM load.

	*	Added checks to prevent threads accessing illegal memory

*/

// -- Controller function for device function
// -- Max matrix width = 1024
void M4_Controller(float* d_Matrix, float* um_Matrix, int height, int width){

	// -- Iterate through all rows
	for (int row_ID = 0; row_ID < height; row_ID++){

		// Each row normalisation has fewer non-zero elements to normalise
		dim3 blocksPerGrid(1);
		dim3 threadsPerBlock(width - row_ID);

		int pivotPos = row_ID * (width + 1);

		// -- Normalise row
		normaliseRow << <blocksPerGrid, threadsPerBlock>> >(pivotPos, um_Matrix, um_Matrix[pivotPos], width);
		
		// -- Wait for kernel to finish normalising row
		cudaDeviceSynchronize();


		// -- Each row will have few non-zero elements to remove
		dim3 rows(height - row_ID);			// Number of rows
		dim3 elements(width - row_ID);		// Number of elements in row

		// -- If pivot is on last row then there's nothing else to eliminate
		//if (rows.x == 1) return;

		// -- Call kernel to scale and subtract rows
		scaleAndSubtract << < rows, elements >> >(pivotPos, width, um_Matrix);

		// -- Wait for kernel to finish eliminating rows
		cudaDeviceSynchronize();

		return;
	}

	// -- Go through all rows starting from second
	for (int row_ID = 1; row_ID < height; row_ID++){

		// -- Define pivot position and value
		int pivotPos = row_ID * (width + 1);

		// -- Each row will have few non-zero elements to remove
		dim3 blocksPerGrid(row_ID + 1);
		dim3 threadsPerBlock(width - row_ID);

		//-- Call kernel to scale and subtract rows
		scaleAndSubtract2 << < blocksPerGrid, threadsPerBlock, sizeof(float)* threadsPerBlock.x >> >(pivotPos, width, um_Matrix);

		// -- Wait for kernel to finish eliminating rows
		cudaDeviceSynchronize();
	}
}

// -- Normalise row relative to pivot value
__global__ void normaliseRow(int pivotPos, float *um_Matrix, float pivotVal, int width){

	// -- Get global threadID
	int g_tid = (blockIdx.x * SEGMENT_LENGTH) + threadIdx.x;

	// -- Kill threads that will access memory outside of their row
	if ((g_tid + pivotPos) > (width - 1)) return;

	// -- Normalise element relative to pivot value
	um_Matrix[pivotPos + g_tid] = um_Matrix[pivotPos + g_tid] / pivotVal;
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
__global__ void scaleAndSubtract2(int pivotPos, int width, float *um_Matrix){

	// -- Don't overwrite pivot row
	if (blockIdx.x == 0) return;

	// -- Get blockID
	int bid = blockIdx.x;

	// -- Get threadID
	int tid = threadIdx.x;

	// -- Declare shared memory dynamically
	extern __shared__ float s_pivotRow[];

	// -- Move from global to shared memory
	s_pivotRow[tid] = um_Matrix[pivotPos + tid];

	// -- Make sure all threads have loaded to shared memory.
	__syncthreads();

	// -- Find coefficient to scale row with
	float coeff = um_Matrix[pivotPos - (width * bid)];

	// -- Update elements in row by subtracting elements with coeff
	um_Matrix[pivotPos - (width * bid) + tid] = um_Matrix[pivotPos - (width * bid) + tid]
		- (coeff * s_pivotRow[tid]);
}