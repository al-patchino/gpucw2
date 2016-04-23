#include "matrix_kernel_2.h"
#include <cuda_runtime.h>

static __global__ void normaliseRow(int pivotPos, float *d_Matrix);
static __global__ void scaleAndSubtract(int pivotPos, int width, float *d_Matrix);
static __global__ void scaleAndSubtract2(int row2_ID, int pivotPos, int width, float *d_Matrix);

/*

This is the first improvement of the GPU implementation.

	* First scaleAndSubtract loop has been removed. Each block is responsible
	  for each row.
	
	* 1 thread updates one element. Same as before. Matrix limit is bounded by
	  max number of threads. This needs changing.
	
	* Slightly different scaleAndSubtract kernel. Block ID should never be 0,
	  otherwise it'll overwrite the pivot row

	* Second scaleAndSubtract loop is more difficult to remove, so it's still there

*/

// -- Controller function for device function
// -- Max matrix width = 1024
void M2_Controller(float* d_Matrix, float* h_Matrix, int height, int width){

	// Iterate through all rows
	for (int row_ID = 0; row_ID < height; row_ID++){

		// Each row normalisation has fewer non-zero elements to normalise
		dim3 blocksPerGrid(1);
		dim3 threadsPerBlock(width - row_ID);

		// -- Define pivot position and value
		int pivotPos = row_ID * (width + 1);

		//printf("pivotPos %d\n", pivotPos);

		// -- Normalise row
		//printf("Lauching normaliseRow<<<%d, %d>>>...\n", blocksPerGrid.x, threadsPerBlock.x);
		normaliseRow << <blocksPerGrid, threadsPerBlock >> >(pivotPos, d_Matrix);

		// -- Wait for kernel to finish normalising row
		cudaDeviceSynchronize();

		// -- Each row will have few non-zero elements to remove
		dim3 rows(height - row_ID);	// Number of rows
		dim3 elements(width - row_ID);		// Number of elements in row

		//-- Call kernel to scale and subtract rows
		scaleAndSubtract <<< rows, elements >>>(pivotPos, width, d_Matrix);
		
	}

	

	// -- Go through all rows starting from second
	for (int row_ID = 1; row_ID < height; row_ID++){

		// -- Define pivot position and value
		int pivotPos = row_ID * (width + 1);

		// -- Loop through j-th column and remove suitable multiples
		for (int row2_ID = 1; row2_ID < (row_ID + 1); row2_ID++){

			// -- Each row will have few non-zero elements to remove
			dim3 blocksPerGrid(1);
			dim3 threadsPerBlock(width - row_ID);

			//-- Call kernel to scale and subtract rows
			scaleAndSubtract2 <<< blocksPerGrid, threadsPerBlock >> >(row2_ID, pivotPos, width, d_Matrix);

			// -- Print matrix
			//printMatrix(height, width);
		}

		//if(row_ID == 3) return;
	}
}

// -- Normalise row relative to pivot value
__global__ void normaliseRow(int pivotPos, float *d_Matrix){

	// -- Get threadID
	int tid = threadIdx.x;

	// -- Normalise element relative to pivot value
	d_Matrix[pivotPos + tid] = d_Matrix[pivotPos + tid] / d_Matrix[pivotPos];
}

// -- Normalise row relative to pivot value downwards
__global__ void scaleAndSubtract(int pivotPos, int width, float *d_Matrix){

	// -- Don't overwrite pivot row
	if (blockIdx.x == 0) return;
	
	// -- Get threadID
	int tid = threadIdx.x;
	
	// -- Get blockID
	int bid = blockIdx.x;

	// -- Find coefficient to scale row with
	float coeff = d_Matrix[pivotPos + (width * bid)];

	// -- Update elements in row by subtracting elements with coeff
	d_Matrix[pivotPos + (width * bid) + tid] = d_Matrix[pivotPos + (width * bid) + tid]
		- (coeff * d_Matrix[pivotPos + tid]);
}

// -- Normalise row relative to pivot value upwards
__global__ void scaleAndSubtract2(int row2_ID, int pivotPos, int width, float *d_Matrix){

	// -- Get threadID
	int tid = threadIdx.x;

	// -- Find coefficient to scale row with
	float coeff = d_Matrix[pivotPos - (width * row2_ID)];

	// -- Update elements in row by subtracting elements with coeff
	d_Matrix[pivotPos - (width * row2_ID) + tid] = d_Matrix[pivotPos - (width * row2_ID) + tid]
		- (coeff * d_Matrix[pivotPos + tid]);
}


