#include "matrix_kernel_4.h"
#include <cuda_runtime.h>
#include <math.h>

static __global__ void normaliseRow(int pivotPos, float *d_Matrix, int row_ID, int width);
static __global__ void scaleAndSubtract(int row_ID, int pivotPos, int width, float *d_Matrix);
static __global__ void scaleAndSubtract2(int row_ID, int pivotPos, int width, float *d_Matrix);

#define THREAD_BLOCK 4
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

This is the fourth improvement of the GPU implementation.

	* Multidimensional blocks helps us get rid of the unecessary loops

*/

// -- Controller function for device function
void M4_Controller(float* d_Matrix, float* h_Matrix, int height, int width){

	// -- Iterate through all rows
	for (int row_ID = 0; row_ID < height; row_ID++){

		// -- Each row normalisation has fewer non-zero elements to normalise
		dim3 blocksPerGrid(CEIL((width - row_ID), THREAD_BLOCK));
		dim3 threadsPerBlock(THREAD_BLOCK);

		// -- Define pivot position and value
		int pivotPos = row_ID * (width + 1);

		// -- Normalise row
		normaliseRow << <blocksPerGrid, threadsPerBlock >> >(pivotPos, d_Matrix, row_ID, width);

		HANDLE_ERROR(cudaThreadSynchronize());

		

		// -- Each row will have few non-zero elements to remove
		int rowsLeft = height - row_ID;
		dim3 blocks(CEIL((width - row_ID), THREAD_BLOCK), rowsLeft, 1);
		dim3 threads(THREAD_BLOCK, 1);

		scaleAndSubtract <<< blocks, threads>> >(row_ID, pivotPos, width, d_Matrix);

		HANDLE_ERROR(cudaThreadSynchronize());

	}


	// -- Go through all rows starting from second
	for (int row_ID = 1; row_ID < height; row_ID++){

		// -- Define pivot position and value
		int pivotPos = row_ID * (width + 1);

		// -- Each row will have few non-zero elements to remove
		int rowsLeft = row_ID;
		dim3 blocks(CEIL((width - row_ID), THREAD_BLOCK), rowsLeft, 1);
		dim3 threads(THREAD_BLOCK, 1);

		scaleAndSubtract2<< < blocks, threads >> >(row_ID, pivotPos, width, d_Matrix);

		HANDLE_ERROR(cudaThreadSynchronize());

		return;
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
__global__ void scaleAndSubtract(int row_ID, int pivotPos, int width, float *d_Matrix){

	// -- Get threadID
	int tid = threadIdx.x;
	int g_tid_x = (blockIdx.x * THREAD_BLOCK) + tid;
	int bid = blockIdx.x;
	int row = blockIdx.y;

	// -- Declare shared memory dynamically
	__shared__ float sh_pivotRow[THREAD_BLOCK];
	__shared__ float sh_Coeff;
	__shared__ float sh_resultRow[THREAD_BLOCK];

	// -- Use first thread to load pivot in shared variable
	if (tid == 0) sh_Coeff = d_Matrix[pivotPos + (width * row)];

	// -- Move from global to shared memory
	sh_pivotRow[tid] = d_Matrix[pivotPos + g_tid_x];
	sh_resultRow[tid] = d_Matrix[pivotPos + (width * row) + g_tid_x];

	// -- Make sure all threads have loaded to shared memory.
	__syncthreads();

	// -- Don't overwrite pivot row
	if (row == 0) return;

	// -- Update elements in row by subtracting elements with coeff
	d_Matrix[pivotPos + (width * row) + g_tid_x] = sh_resultRow[tid]
		- (sh_Coeff * sh_pivotRow[tid]);

	__syncthreads();
}

// -- Normalise row relative to pivot value upwards
__global__ void scaleAndSubtract2(int row_ID, int pivotPos, int width, float *d_Matrix){

	// -- Get threadID
	int tid = threadIdx.x;
	int g_tid_x = (blockIdx.x * THREAD_BLOCK) + tid;
	int bid = blockIdx.x;
	int row = blockIdx.y;

	// -- Declare shared memory dynamically
	__shared__ float sh_pivotRow[THREAD_BLOCK];
	__shared__ float sh_Coeff;
	__shared__ float sh_resultRow[THREAD_BLOCK];

	// -- Use first thread to load pivot in shared variable
	if (tid == 0) sh_Coeff = d_Matrix[pivotPos + (width * row)];

	// -- Move from global to shared memory
	sh_pivotRow[tid] = d_Matrix[pivotPos + g_tid_x];
	sh_resultRow[tid] = d_Matrix[pivotPos - (width * row) + g_tid_x];

	// -- Make sure all threads have loaded to shared memory.
	__syncthreads();

	// -- Don't overwrite pivot row
	if (row == 0) return;

	// -- Update elements in row by subtracting elements with coeff
	d_Matrix[pivotPos - (width * row) + g_tid_x] = sh_resultRow[tid]
		- (sh_Coeff * sh_pivotRow[tid]);

	__syncthreads();

}