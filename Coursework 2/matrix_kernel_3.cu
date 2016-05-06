#include "matrix_kernel_3.h"
#include <cuda_runtime.h>
#include <math.h>

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

This is the third improvement of the GPU implementation.

	* Uses shared memory for entire row of size THREAD_BLOCK
	* Now we can get rid of the wasteful cudaDeviceSync()
	* Still slower than CPU implementation

*/

// -- Controller function for device function
void M3_Controller(float* d_Matrix, float* h_Matrix, int height, int width){

	// -- Iterate through all rows
	for (int row_ID = 0; row_ID < height; row_ID++){

		// -- Each row normalisation has fewer non-zero elements to normalise
		dim3 blocksPerGrid(CEIL((width - row_ID), THREAD_BLOCK));
		dim3 threadsPerBlock(THREAD_BLOCK);

		// -- Define pivot position and value
		int pivotPos = row_ID * (width + 1);

		// -- Normalise row
		normaliseRow << <blocksPerGrid, threadsPerBlock >> >(pivotPos, d_Matrix, row_ID, width);

		// -- Loop through j-th column and remove suitable multiples
		for (int row2_ID = 1; row2_ID < (height - row_ID); row2_ID++){

			// -- Each row will have few non-zero elements to remove
			dim3 blocksPerGrid(CEIL((width - row_ID), THREAD_BLOCK));
			dim3 threadsPerBlock(THREAD_BLOCK);

			//-- Call kernel to scale and subtract rows
			scaleAndSubtract << < blocksPerGrid, threadsPerBlock >> >(row_ID, row2_ID, pivotPos, width, d_Matrix);
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
		}
	}
}

// -- Normalise row relative to pivot value
__global__ void normaliseRow(int pivotPos, float *d_Matrix, int row_ID, int width){

	// -- Get threadID
	int tid = threadIdx.x;
	int g_tid = (blockIdx.x * THREAD_BLOCK) + tid;

	// -- Declare shared memory dynamically
	__shared__ float sh_Row[THREAD_BLOCK];
	__shared__ float sh_Pivot;

	// -- Use first thread to load pivot in shared variable
	if (tid == 0) sh_Pivot = d_Matrix[pivotPos];

	// -- Use other threads to load elements into shared block
	sh_Row[tid] = d_Matrix[pivotPos + g_tid];	

	// -- Make sure all threads have loaded to shared memory
	__syncthreads();

	// -- Normalise row element relative to pivot
	sh_Row[tid] = sh_Row[tid] / sh_Pivot;

	// -- To guard against the next case, but not sure if completely needed **************************************************
	__syncthreads();

	d_Matrix[pivotPos + g_tid] = sh_Row[tid];

	// -- To guard against the next case, but not sure if completely needed **************************************************
	__syncthreads();
}

// -- Normalise row relative to pivot value downwards
__global__ void scaleAndSubtract(int row_ID, int row2_ID, int pivotPos, int width, float *d_Matrix){

	// -- Get threadID
	int tid = threadIdx.x;
	int g_tid = (blockIdx.x * THREAD_BLOCK) + tid;

	// -- Declare shared memory dynamically
	__shared__ float sh_pivotRow[THREAD_BLOCK];
	__shared__ float sh_Coeff;
	__shared__ float sh_resultRow[THREAD_BLOCK];
	
	// -- Use first thread to load pivot in shared variable
	if (tid == 0) sh_Coeff = d_Matrix[pivotPos + (width * row2_ID)];

	// -- Move from global to shared memory
	sh_pivotRow[tid] = d_Matrix[pivotPos + g_tid];
	sh_resultRow[tid] = d_Matrix[pivotPos + (width*row2_ID) + g_tid];

	// -- Make sure all threads have loaded to shared memory.
	__syncthreads();

	d_Matrix[pivotPos + (width * row2_ID) + g_tid] = sh_resultRow[tid]
		- (sh_Coeff * sh_pivotRow[tid]);

	__syncthreads();
}

// -- Normalise row relative to pivot value upwards
__global__ void scaleAndSubtract2(int row_ID, int row2_ID, int pivotPos, int width, float *d_Matrix){

	// -- Get threadID
	int tid = threadIdx.x;
	int g_tid = (blockIdx.x * THREAD_BLOCK) + tid;

	// -- Declare shared memory dynamically
	__shared__ float sh_pivotRow[THREAD_BLOCK];
	__shared__ float sh_Coeff;
	__shared__ float sh_resultRow[THREAD_BLOCK];

	// -- Use first thread to load pivot in shared variable
	if (tid == 0) sh_Coeff = d_Matrix[pivotPos - (width * row2_ID)];

	// -- Move from global to shared memory
	sh_pivotRow[tid] = d_Matrix[pivotPos + g_tid];
	sh_resultRow[tid] = d_Matrix[pivotPos - (width*row2_ID) + g_tid];

	// -- Make sure all threads have loaded to shared memory.
	__syncthreads();

	// -- Update elements in row by subtracting elements with coeff
	d_Matrix[pivotPos - (width * row2_ID) + g_tid] = sh_resultRow[tid]
		- (sh_Coeff * sh_pivotRow[tid]);
	
	__syncthreads();
	
}