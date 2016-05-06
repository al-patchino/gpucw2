#include "matrix_kernel_2.h"
#include <cuda_runtime.h>

static __global__ void normaliseRow(int pivotPos, float *d_Matrix, int row_ID, int width);
static __global__ void scaleAndSubtract(int row_ID, int row2_ID, int pivotPos, int width, float *d_Matrix);
static __global__ void scaleAndSubtract2(int row_ID, int row2_ID, int pivotPos, int width, float *d_Matrix);

#define THREAD_BLOCK 32
#define CEIL(a, b) (((a) / (b)) + (((a) % (b)) > 0 ? 1 : 0))
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}

/*

This is the first improvement of the GPU implementation.

	* 1 thread updates one element. Same as before.
	
	* Uses multiple blocks instead of single block

	* Uses shared memory for coeff and pivot since it is reused

	* Cons: This is very inefficient. cudaSynchronise calls are very costly
	  in execution time and thus needs to be removed

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
		HANDLE_ERROR(cudaDeviceSynchronize());

		// -- Loop through j-th column and remove suitable multiples
		for (int row2_ID = 1; row2_ID < (height - row_ID); row2_ID++){

			// -- Each row will have few non-zero elements to remove
			dim3 blocksPerGrid(CEIL((width - row_ID), THREAD_BLOCK));
			dim3 threadsPerBlock(THREAD_BLOCK);

			//-- Call kernel to scale and subtract rows
			scaleAndSubtract << < blocksPerGrid, threadsPerBlock >> >(row_ID, row2_ID, pivotPos, width, d_Matrix);

			// -- Wait for kernel to finish eliminating rows
			HANDLE_ERROR(cudaDeviceSynchronize());
		}
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
			HANDLE_ERROR(cudaDeviceSynchronize());
		}

	}
}

// -- Normalise row relative to pivot value
__global__ void normaliseRow(int pivotPos, float *d_Matrix, int row_ID, int width){

	// -- Get threadID
	int g_tid = (blockIdx.x * THREAD_BLOCK) + threadIdx.x;

	// -- Declared pivot as shared mem to be shared by the block
	__shared__ float sh_pivot;

	// -- Only the first of each block loads to shared mem
	if (threadIdx.x == 0) sh_pivot = d_Matrix[pivotPos];

	// -- Wait until pivot has been loaded
	__syncthreads();

	// -- If tid tries to access illegal memory then return
	if (g_tid > (width - row_ID)) return;

	// -- Normalise element relative to pivot value
	d_Matrix[pivotPos + g_tid] = d_Matrix[pivotPos + g_tid] / sh_pivot;

	__syncthreads();
	
}

// -- Eliminate downwards
__global__ void scaleAndSubtract(int row_ID, int row2_ID, int pivotPos, int width, float *d_Matrix){

	// -- Get global threadID
	int g_tid = (blockIdx.x * THREAD_BLOCK) + threadIdx.x;

	// -- Declared coeff as shared mem to be shared by the block
	__shared__ float sh_coeff;

	// -- Only the first of each block loads to shared mem
	if (threadIdx.x == 0) sh_coeff = d_Matrix[pivotPos + (width * row2_ID)];

	// -- Wait until coeff has been loaded
	__syncthreads();
	
	// -- If tid tries to access illegal memory then return
	if (g_tid > (width - row_ID)) return;


	// -- Update elements in row by subtracting elements with coeff
	d_Matrix[pivotPos + (width * row2_ID) + g_tid] = d_Matrix[pivotPos + (width * row2_ID) + g_tid]
												- (sh_coeff * d_Matrix[pivotPos + g_tid]);

	__syncthreads();
	
}

// -- Eliminate upwards
__global__ void scaleAndSubtract2(int row_ID, int row2_ID, int pivotPos, int width, float *d_Matrix){

	// -- Get global threadID
	int g_tid = (blockIdx.x * THREAD_BLOCK) + threadIdx.x;

	// -- Declared coeff as shared mem to be shared by the block
	__shared__ float sh_coeff;

	// -- Only the first of each block loads to shared mem
	if (threadIdx.x == 0) sh_coeff = d_Matrix[pivotPos - (width * row2_ID)];

	__syncthreads();
	
	// -- If tid tries to access illegal memory then return
	if (g_tid > (width - row_ID)) return;

	// -- Update elements in row by subtracting elements with coeff
	d_Matrix[pivotPos - (width * row2_ID) + g_tid] = d_Matrix[pivotPos - (width * row2_ID) + g_tid]
		- (sh_coeff * d_Matrix[pivotPos + g_tid]);

	__syncthreads();
}


