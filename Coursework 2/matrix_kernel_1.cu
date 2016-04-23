#include "matrix_kernel_1.h"
#include <cuda_runtime.h>


// -- Controller function for device function
// -- Max matrix width = 1024
void M1_Controller(float* d_Matrix, float* h_Matrix, int height, int width){

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
		normaliseRow <<<blocksPerGrid, threadsPerBlock >>>(pivotPos, d_Matrix);

		// -- Wait for kernel to finish normalising row
		cudaDeviceSynchronize();
		
		// -- Loop through j-th column and remove suitable multiples
		for (int row2_ID = 1; row2_ID < (height - row_ID); row2_ID++){

			// -- Each row will have few non-zero elements to remove
			dim3 blocksPerGrid(1);
			dim3 threadsPerBlock(width - row_ID);

			//-- Call kernel to scale and subtract rows
			//printf("Lauching scaleAndSubtract<<<%d, %d>>>(%d)...\n", blocksPerGrid.x, threadsPerBlock.x, pivotPos);
			scaleAndSubtract <<< blocksPerGrid, threadsPerBlock >> >(row2_ID, pivotPos, width, d_Matrix);
		

		}	
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
			//printf("Lauching scaleAndSubtract2<<<%d, %d>>>(%d)...\n", blocksPerGrid.x, threadsPerBlock.x, pivotPos);
			scaleAndSubtract2 << < blocksPerGrid, threadsPerBlock >> >(row2_ID, pivotPos, width, d_Matrix);

			// -- Print matrix
			//printMatrix(height, width);
		}
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
__global__ void scaleAndSubtract(int row2_ID, int pivotPos, int width, float *d_Matrix){

	// -- Get threadID
	int tid = threadIdx.x;

	// -- Find coefficient to scale row with
	float coeff = d_Matrix[pivotPos + (width * row2_ID)];

	// -- Update elements in row by subtracting elements with coeff
	d_Matrix[pivotPos + (width * row2_ID) + tid] = d_Matrix[pivotPos + (width * row2_ID) + tid]
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


