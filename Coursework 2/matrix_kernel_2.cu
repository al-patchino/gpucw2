#include "matrix_kernel_2.h"
#include <cuda_runtime.h>

static __global__ void normaliseRow(int pivotPos, float *d_Matrix, int row_ID, int width);
static __global__ void scaleAndSubtract(int row_ID, int row2_ID, int pivotPos, int width, float *d_Matrix);
static __global__ void scaleAndSubtract2(int row_ID, int row2_ID, int pivotPos, int width, float *d_Matrix);

#define THREAD_BLOCK 4
#define CEIL(a, b) (((a) / (b)) + (((a) % (b)) > 0 ? 1 : 0))
/*

This is the first improvement of the GPU implementation.

	
	* 1 thread updates one element. Same as before. Matrix limit is bounded by
	  max number of threads. This needs changing.
	
	* Slightly different scaleAndSubtract kernel. Block ID should never be 0,
	  otherwise it'll overwrite the pivot row

	* Second scaleAndSubtract loop is more difficult to remove, so it's still there

*/

// -- Controller function for device function
void M2_Controller(float* d_Matrix, float* h_Matrix, int height, int width){

	// -- Iterate through all rows
	for (int row_ID = 0; row_ID < height; row_ID++){

		// -- Each row normalisation has fewer non-zero elements to normalise
		dim3 blocksPerGrid(CEIL((width - row_ID), THREAD_BLOCK));
		dim3 threadsPerBlock(THREAD_BLOCK);

		// -- Define pivot position and value
		int pivotPos = row_ID * (width + 1);

		// -- Normalise row
		normaliseRow << <blocksPerGrid, threadsPerBlock >> >(pivotPos, d_Matrix, row_ID, width);

		// -- Wait for kernel to finish normalising row
		cudaDeviceSynchronize();

		// -- Loop through j-th column and remove suitable multiples
		for (int row2_ID = 1; row2_ID < (height - row_ID); row2_ID++){

			// -- Each row will have few non-zero elements to remove
			dim3 blocksPerGrid(CEIL((width - row_ID), THREAD_BLOCK));
			dim3 threadsPerBlock(THREAD_BLOCK);

			//-- Call kernel to scale and subtract rows
			scaleAndSubtract << < blocksPerGrid, threadsPerBlock >> >(row_ID, row2_ID, pivotPos, width, d_Matrix);

		}
		
		// -- Wait for kernel to finish eliminating rows
		cudaDeviceSynchronize();
	}
	
	// -- Go through all rows starting from second
	for (int row_ID = 1; row_ID < height; row_ID++){

		// -- Define pivot position and value
		int pivotPos = row_ID * (width + 1);

		// -- Loop through j-th column and remove suitable multiples
		for (int row2_ID = 1; row2_ID < (row_ID + 1); row2_ID++){

			// -- Each row will have few non-zero elements to remove
			dim3 blocksPerGrid(CEIL((width - row_ID), THREAD_BLOCK));
			dim3 threadsPerBlock(THREAD_BLOCK);

			//-- Call kernel to scale and subtract rows
			scaleAndSubtract2 << < blocksPerGrid, threadsPerBlock >> >(row_ID, row2_ID, pivotPos, width, d_Matrix);

			// -- Wait for kernel to finish eliminating rows
			cudaDeviceSynchronize();

		}

		return;
	}
}

// -- Normalise row relative to pivot value
__global__ void normaliseRow(int pivotPos, float *d_Matrix, int row_ID, int width){

	// -- Get threadID
	int g_tid = (blockIdx.x * THREAD_BLOCK) + threadIdx.x;

	// -- If tid tries to access illegal memory then return
	if (g_tid > (width - row_ID)) return;

	// -- Normalise element relative to pivot value
	d_Matrix[pivotPos + g_tid] = d_Matrix[pivotPos + g_tid] / d_Matrix[pivotPos];
}

// -- Eliminate downwards
__global__ void scaleAndSubtract(int row_ID, int row2_ID, int pivotPos, int width, float *d_Matrix){

	// -- Get global threadID
	int g_tid = (blockIdx.x * THREAD_BLOCK) + threadIdx.x;

	// -- If tid tries to access illegal memory then return
	if (g_tid > (width - row_ID)) return;

	// -- Find coefficient to scale row with
	float coeff = d_Matrix[pivotPos + (width * row2_ID)];

	// -- Update elements in row by subtracting elements with coeff
	d_Matrix[pivotPos + (width * row2_ID) + g_tid] = d_Matrix[pivotPos + (width * row2_ID) + g_tid]
												- (coeff * d_Matrix[pivotPos + g_tid]);
}

// -- Eliminate upwards
__global__ void scaleAndSubtract2(int row_ID, int row2_ID, int pivotPos, int width, float *d_Matrix){

	// -- Get global threadID
	int g_tid = (blockIdx.x * THREAD_BLOCK) + threadIdx.x;
	
	// -- If tid tries to access illegal memory then return
	if (g_tid > (width - row_ID)) return;

	// -- Find coefficient to scale row with
	float coeff = d_Matrix[pivotPos - (width * row2_ID)];

	// -- Update elements in row by subtracting elements with coeff
	d_Matrix[pivotPos - (width * row2_ID) + g_tid] = d_Matrix[pivotPos - (width * row2_ID) + g_tid]
		- (coeff * d_Matrix[pivotPos + g_tid]);
}


