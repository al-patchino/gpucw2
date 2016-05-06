#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda_runtime.h>
//#include <sys/time.h>		// -- Linux
#include <Winsock2.h>		// -- Windows
#include <time.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>

#include "matrix_kernel_1.h"
#include "matrix_kernel_2.h"
#include "matrix_kernel_3.h"
#include "matrix_kernel_4.h"

#define HANDLE_ERROR handleError();

// -- General purpose functions
void printProperties();
void generateRandomMatrix(int height, int width, float* mat);
void printMatrix(int height, int width, float* mat);
void writeToFile(int height, int width, float* mat, float* mat2);
void writeOutputToFile(int height, int width, float* mat, float* mat2);
void swapRows(int row1, int row2, int height, int width);
void solveOnHost(int height, int width, float* mat);
void handleError();
void kernelLaucher(int kernelID, float *h_Matrix, int height, int width);
void kernelLaucherUM(int kernelID, int height, int width);
float validateSolution(float* h_mat, float* d_mat, int width, int height);
float checkSum(float* mat, int height, int width);

// Does not change once it has been randomised
float *input_Matrix; 

// Matrix to be solved by the CPU. Does not change once solved
float *cpu_Matrix;	 

// Pre-UM matrices
float *h_Matrix;
float *d_Matrix;

float *um_Matrix; // Unified Memory matrix 


int choice = NULL;


int main() {

	// -- Max matrix for kernel 1 = 791x792
	// -- Max matrix for kernel 2 = 97x98
	// -- Max matrix for kernel 3 = 211x212

	int kernelID = 3;


	do{

		int height;
		int width;

		int printInfo = 0;

	/*	//Print instructions
		printf("\n####################################################\n"
			"################ -- Menu -- ########################\n"
			"####################################################\n\n"
			"1 - KERNEL#1 - Naive implementation\n"
			"2 - KERNEL#2 - Multiple blocks & shared variable\n"
			"3 - KERNEL#3 - Use of shared memory and tiles\n"
			"4 - KERNEL#4 - \n"
			"5 - KERNEL#5 - Using Unified Memory (CUDA 6.0+)\n\n"
			"0 - Quit\n\n"
			"###################################################\n\n");

		printf("Format: [Matrix width] [KernelID] [Print]\n");
		fflush(stdin);
		printf("> ");
		scanf("%d %d %d", &width, &kernelID, &printInfo);

		*/

		//printProperties();

		//printf("Starting kernel #%d on a matrix width of %d...\n", kernelID, width);
		for (int power = 5; power < 13; power++){  //// Benchmark

			width = (int)pow(2.0, power);			//// benchmark
			height = width - 1;

			for (int i = 0; i < 3; i++){           ///// benchmark

				// -- Allocate space on the host
				input_Matrix = (float*)malloc(sizeof(float)* height * width);
				cpu_Matrix = (float*)malloc(sizeof(float)* height * width);
				h_Matrix = (float*)malloc(sizeof(float)* height * width);


				// -- Generate random floats & print matrix
				generateRandomMatrix(height, width, input_Matrix);
				if (printInfo == 1 && width < 33) {
					printf("This is the input matrix: \n");
					printMatrix(height, width, input_Matrix);
				}

				// -- Copy randomised matrix to other memories to preserve original
				memcpy(h_Matrix, input_Matrix, sizeof(float)* height * width);
				memcpy(cpu_Matrix, input_Matrix, sizeof(float)* height * width);

				// -- Start Timer for CPU
				StopWatchInterface *CPUTime = NULL;
				sdkCreateTimer(&CPUTime);
				sdkResetTimer(&CPUTime);
				sdkStartTimer(&CPUTime);

				// -- Solve & print matrix on the CPU. This does not change once solved.
				solveOnHost(height, width, cpu_Matrix);

				cudaThreadSynchronize();
				sdkStopTimer(&CPUTime);
				

				if (printInfo == 1 && width < 33) {
					printf("This is the solution from the CPU: \n");
					printMatrix(height, width, cpu_Matrix);
				}

				// -- Start Timer
				StopWatchInterface *GPUTime = NULL;
				sdkCreateTimer(&GPUTime);
				sdkResetTimer(&GPUTime);
				sdkStartTimer(&GPUTime);

				// -- Launch GPU kernel
				if (kernelID < 5) kernelLaucher(kernelID, h_Matrix, height, width);
				if (kernelID >= 5) kernelLaucherUM(kernelID, height, width);

				// -- Finish timer
				cudaThreadSynchronize();
				sdkStopTimer(&GPUTime);

				if (printInfo == 1 && width < 33) {
					printf("This is the solution from the GPU using kernel #%d: \n", kernelID);
					if (kernelID < 5) printMatrix(height, width, h_Matrix);
					if (kernelID > 4) printMatrix(height, width, um_Matrix);
				}

				//printf("Time taken (GPU): %f ms\n\n", sdkGetTimerValue(&GPUTime));
				//printf("CPU checksum: %f\n", checkSum(cpu_Matrix, height, width));

				// -- For non-UM kernels
				if (kernelID < 5){
					printf("Kernel #%d: Time taken (CPU): %f ms, GPU: %f ms. \nWidth: %d, Height: %d\nMean error: %f\n",
						kernelID, sdkGetTimerValue(&CPUTime), sdkGetTimerValue(&GPUTime), width, height, validateSolution(cpu_Matrix, h_Matrix, width, height));
				}

				// -- For UM kernels
				if (kernelID > 4){
					//printf("Time taken (CPU): %f ms, GPU: %f ms. \nWidth: %d, Height: %d\nMean error: %f\n\n",
						//sdkGetTimerValue(&CPUTime), sdkGetTimerValue(&GPUTime), width, height, validateSolution(cpu_Matrix, um_Matrix, width, height));
				}



				//printf("Time taken (CPU): %f ms, GPU: %f ms. \nWidth: %d, Height: %d\n", sdkGetTimerValue(&CPUTime), sdkGetTimerValue(&GPUTime), width, height);


				//writeToFile(height, width, cpu_Matrix, h_Matrix);
				//writeOutputToFile(height, width, cpu_Matrix, h_Matrix);

				free(h_Matrix);
				free(input_Matrix);
				free(cpu_Matrix);
				HANDLE_ERROR(cudaFree(d_Matrix));
				HANDLE_ERROR(cudaFree(um_Matrix));
			}

			printf("\n");
		}

		/*



		
		*/



	} while (kernelID != 0);

	// -- Free memory
	HANDLE_ERROR(cudaFree(d_Matrix));
	HANDLE_ERROR(cudaFree(um_Matrix));
	free(h_Matrix);
	free(input_Matrix);
	free(cpu_Matrix);

	return 0;
}

void generateRandomMatrix(int height, int width, float* mat) {

	// Generates a random matrix of floats
	//srand((unsigned int)time(NULL));
	for (int i = 0; i < (height * width); i++) {
		//float value = (float)rand() / (float)(RAND_MAX/90);
		float value = (rand() % 100) + 20;
		mat[i] = (float)value;
	}
	mat[0] = 90.00;

}

void writeToFile(int height, int width, float* mat, float* mat2){

	FILE *fp;
	char output[] = "output.txt";
	int n;
	fp = fopen(output, "w");

	for (int row = 0; row < height; row++){
		for (int col = 0; col < width; col++){

			fprintf(fp, "[%.4f] ", mat[(row * width) + col]);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n\n");

	for (int row = 0; row < height; row++){
		for (int col = 0; col < width; col++){

			fprintf(fp, "[%.4f] ", mat2[(row * width) + col]);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n\n");


	fclose(fp);

}

void writeOutputToFile(int height, int width, float* mat, float* mat2){

	FILE *fp;
	char output[] = "output2.txt";
	int n;
	fp = fopen(output, "w");

	int startPos = width - 1;

	for (int row_ID = 0; row_ID < height; row_ID++){

		fprintf(fp, "[%f] ", mat[startPos + (width * row_ID)]);
		fprintf(fp, "[%f]\n", mat2[startPos + (width * row_ID)]);

	}

	fclose(fp);

}

void solveOnHost(int height, int width, float* mat){

	float pivot = 0;
	
	// -- Go through all rows
	for (int row_ID = 0; row_ID < height; row_ID++){

		// -- Define pivot position and value
		int pivotPos = row_ID * (width + 1);
		pivot = mat[pivotPos];

		// -- Normalise row relative to pivot		
		for (int col = 0; col < (width - row_ID); col++){
			mat[pivotPos + col] = mat[pivotPos + col] / pivot;
		}

		

		
		// -- Loop through j-th column and remove suitable multiples
		for (int row2_ID = 1; row2_ID < (height - row_ID); row2_ID++){

			float coeff = mat[pivotPos + (width * row2_ID)];

			for (int offset_col = 0; offset_col < (width - row_ID); offset_col++){

				// -- Update elements in row by subtracting elements with coeff
				mat[pivotPos + (width * row2_ID) + offset_col] = mat[pivotPos + (width * row2_ID) + offset_col]
					- (coeff * mat[pivotPos + offset_col]);
			}		
		}


	}



	// -- Go through all rows starting from second
	for (int row_ID = 1; row_ID < height; row_ID++){

		// -- Define pivot position and value
		int pivotPos = row_ID * (width + 1);
		pivot = mat[pivotPos];

		// -- Loop through j-th column and remove suitable multiples
		for (int row2_ID = 1; row2_ID < (row_ID + 1); row2_ID++){

			float coeff = mat[pivotPos - (width * row2_ID)];

			for (int offset_col = 0; offset_col < (width - row_ID); offset_col++){

				// -- Update elements in row by subtracting elements with coeff
				mat[pivotPos - (width * row2_ID) + offset_col] = mat[pivotPos - (width * row2_ID) + offset_col]
					- (coeff * mat[pivotPos + offset_col]);
			}
		}

	}

}

void printMatrix(int height, int width, float* mat) {

	printf("HERE'S A PRINTOUT OF THE MATRIX:\n");

	for (int i = 0; i < height; ++i){

		for (int k = 0; k < width; ++k){

			printf("[%f] ", mat[i * width + k]);
		}
		printf("\n");
	}
	printf("\n\n");

}

void printProperties() {

	int gpuNumber, gpuID;
	cudaError_t errorCode;
	struct cudaDeviceProp gpuProp;


	errorCode = cudaGetDeviceCount(&gpuNumber);

	if (errorCode) printf("Error in cudaDeviceCount\n");

	printf("Number of the available CUDA devices: %d\n", gpuNumber);

	for (gpuID = 0; gpuID < gpuNumber; gpuID++) {

		errorCode = cudaGetDeviceProperties(&gpuProp, gpuID);

		printf("\nDevice ID: %d\n", gpuID);
		printf("Name of the   GPU:                      %s\n", gpuProp.name);
		printf("Compute Capability:                  %d.%d\n", gpuProp.major, gpuProp.minor);
		printf("Global Memory                      %.2f Gb\n", (float)gpuProp.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
		printf("Shared Memory per Block:           %.2f Kb\n", (float)gpuProp.sharedMemPerBlock / 1024.0);
		printf("Constant Memory                    %.2f Kb\n", (float)gpuProp.totalConstMem / 1024.0);
		printf("Maximum Number of Threads per Block:    %d\n", gpuProp.maxThreadsPerBlock);
		printf("Number of Registers per block   :       %d\n", gpuProp.regsPerBlock);
		printf("Support Concurrent Exec.    of Kernels? %d\n", gpuProp.concurrentKernels);

		// Additional information
		printf("Clock frequency in kilohertz:						%d kHz\n", gpuProp.clockRate);
		printf("Specified whether there is a run time limit on kernels: %d\n", gpuProp.kernelExecTimeoutEnabled);
		printf("Size of L2 cache in bytes:						  %d bytes\n", gpuProp.l2CacheSize);
		printf("Maximum size of each dimension of a grid:				%d\n", gpuProp.maxGridSize);
		printf("Global memory bus width in bits:					    %d\n", gpuProp.memoryBusWidth);
		printf("Peak memory clock frequency in kilohertz:			%d kHz\n", gpuProp.memoryClockRate);
		printf("Warp size in threads:									%d\n", gpuProp.warpSize);

		printf("Device overlap:											%d\n", gpuProp.deviceOverlap);

	}
}

void swapRows(int row1, int row2, int height, int width, float* mat){

	printf("Swapping rowID #%d with rowID #%d...\n", row1, row2);

	// -- Allocate space for buffer to hold row
	float *tempRow = (float*)malloc(sizeof(float)*width);

	// -- Calculate startpos & endpos for elements to be swapped 
	int startPos_row1 = row1 * width;
	int endPos_row1 = row1 * width + width;

	// -- Calculate startpos & endpos for elements to be swapped 
	int startPos_row2 = row2 * width;
	int endPos_row2 = row2 * width + width;


	for (int i = 0; i < width; ++i){

		//tempRow[i - (width * row1)] = hostInput[i];
		// -- Save first row element to buffer memory
		tempRow[i] = cpu_Matrix[startPos_row1 + i];
		
		// -- Swap 
		cpu_Matrix[startPos_row1 + i] = cpu_Matrix[startPos_row2 + i];

		cpu_Matrix[startPos_row2 + i] = tempRow[i];
	}

	free(tempRow);


}

void handleError() {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(err));
	}
}

void kernelLaucher(int kernelID, float *h_Matrix, int height, int width){

	cudaError_t err = cudaSuccess;

	// -- Allocate device memory
	HANDLE_ERROR(cudaMalloc((void**)&d_Matrix, sizeof(float)* height * width));
	// -- Copy host matrix to device matrix
	HANDLE_ERROR(cudaMemcpy(d_Matrix, h_Matrix, height * width * sizeof(float), cudaMemcpyHostToDevice));
	
	/*
	// -- Allocate device memory
	err = cudaMalloc((void**)&d_Matrix, sizeof(float)* height * width);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device memory (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// -- Copy host matrix to device matrix
	err = cudaMemcpy(d_Matrix, h_Matrix, height * width * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy host matrix to device matrix (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}*/

	// -- Which kernel to run depending on user input
	if (kernelID == 1)	M1_Controller(d_Matrix, h_Matrix, height, width);
	if (kernelID == 2)	M2_Controller(d_Matrix, h_Matrix, height, width);
	if (kernelID == 3)	M3_Controller(d_Matrix, h_Matrix, height, width);
	if (kernelID == 4)	M4_Controller(d_Matrix, h_Matrix, height, width);
	
	// -- Copy device matrix to host matrix
	HANDLE_ERROR(cudaMemcpy(h_Matrix, d_Matrix, height * width * sizeof(float), cudaMemcpyDeviceToHost));
	/*
	// -- Copy device matrix to host matrix
	err = cudaMemcpy(h_Matrix, d_Matrix, height * width * sizeof(float), cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}*/

}

void kernelLaucherUM(int kernelID, int height, int width){

	cudaError_t err = cudaSuccess;

	// -- Allocate managed memory
	err = cudaMallocManaged(&um_Matrix, height * width * sizeof(float));

	// -- Copy randomised matrix to managed memory
	memcpy(um_Matrix, h_Matrix, height * width * sizeof(float));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "(UM) Failed to allocate device memory (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// -- Which kernel to run depending on user input
	if (kernelID == 5)	M4_Controller(d_Matrix, um_Matrix, height, width);


}

float checkSum(float* mat, int height, int width){

	float checkSum = 0.0;

	int startPos = width - 1;

	for (int row_ID = 0; row_ID < height; row_ID++){
		checkSum = checkSum + mat[startPos + (width * row_ID)];
	}

	return checkSum;

}

// -- Finds average absolute difference between both solutions
float validateSolution(float* h_mat, float* d_mat, int width, int height){

	int startPos = width - 1;
	float diff = 0;
	
	for (int row_ID = 0; row_ID < height; row_ID++){

		diff = diff + abs(h_mat[startPos + (width * row_ID)] - d_mat[startPos + (width * row_ID)]);

	}

	diff = diff / width;

	return diff;
}