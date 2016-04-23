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



#define HANDLE_ERROR handleError();

// -- General purpose functions
void writeToFile();
int* readFromFile(int *input);
void generateRandomMatrix(int height, int width);
void printMatrix(int height, int width, float* mat);
void swapRows(int row1, int row2, int height, int width);
void solveOnHost(int height, int width, float* hostInput);
void handleError();
void kernelLaucher(int kernelID, float *h_Matrix, int height, int width);
float checkSum(float* mat, int height, int width);

float *hostInput; // The input 1D vector
float *h_Matrix;
float *hostOutput; // The output vector
float *d_Matrix;

int blocksize = 0;
int numOfInputElements = 0;
int numOfOutputElements = 0;

int choice = NULL;

struct timeval t1, t2;

int main() {

	cudaError_t err = cudaSuccess;

	int *temp;
	int m = 0;

	// -- Max matrix for kernel 1 = 791x792
	// -- Max matrix for kernel 2 = 98x99

	int height = 791;
	int width = height + 1;



	int kernelID = 1;

	// -- Allocate space on the host
	hostInput = (float*)malloc(sizeof(float) * height * width);
	h_Matrix = (float*)malloc(sizeof(float)* height * width);
	

	// -- Generate random floats
	generateRandomMatrix(height, width);
	memcpy(h_Matrix, hostInput, sizeof(float)* height * width);

	// -- Print host matrix
	printf("hostInput:\n");
	//printMatrix(height, width, hostInput);


	// -- Start Timer
	StopWatchInterface *CPUTime = NULL;
	sdkCreateTimer(&CPUTime);
	sdkResetTimer(&CPUTime);
	sdkStartTimer(&CPUTime);

	// -- Solve matrix on the CPU
	solveOnHost(height, width, hostInput);

	// -- Finish timer
	cudaThreadSynchronize();
	sdkStopTimer(&CPUTime);

	// -- Print elapsed time
	printf("Time taken (CPU): %f ms\n\n", sdkGetTimerValue(&CPUTime));

	// -- Print matrix
	printf("CPU Identity Matrix:\n");
	//printMatrix(height, width, hostInput);	

	// -- Start Timer
	StopWatchInterface *GPUTime = NULL;
	sdkCreateTimer(&GPUTime);
	sdkResetTimer(&GPUTime);
	sdkStartTimer(&GPUTime);

	// -- Launch GPU kernel
	kernelLaucher(kernelID, h_Matrix, height, width);

	// -- Print matrix
	printf("GPU Identity Matrix:\n");
	//printMatrix(height, width, h_Matrix);

	// -- Finish timer
	cudaThreadSynchronize();
	sdkStopTimer(&GPUTime);

	// -- Print elapsed time
	printf("Time taken (GPU): %f ms\n\n", sdkGetTimerValue(&GPUTime));


	printf("CPU checksum: %f\n", checkSum(hostInput, height, width));
	printf("GPU checksum: %f\n", checkSum(h_Matrix, height, width));

	// -- Free memory

	cudaFree(d_Matrix);

	free(h_Matrix);


	free(hostInput);
	free(hostOutput);

	return 0;
}

void writeToFile() {

	printf("Opening text file... ");

	//srand((unsigned int)time(NULL));

	FILE *f = fopen("file.txt", "w+");

	if (f == NULL) {
		printf("Error opening file!\n");
		exit(1);
	}

	/* print integers to text file */
	int a = 10;

	for (int i = 0; i < numOfInputElements; i++) {
		int value = rand() % a;
		fprintf(f, "%d ", value);
	}

	fclose(f);

	printf("Finished writing random numbers!\n");

}

void generateRandomMatrix(int height, int width) {

	// Generates a random matrix of floats
	//srand((unsigned int)time(NULL));
	for (int i = 0; i < (height * width); i++) {
		float value = (float)rand() / (float)(RAND_MAX/90);
		hostInput[i] = value;
	}
	hostInput[0] = 90.99;

}

void solveOnHost(int height, int width, float* hostInput){

	float pivot = 0;
	
	// -- Print matrix
	//printMatrix(height, width);
	
	// -- Go through all rows
	for (int row_ID = 0; row_ID < height; row_ID++){

		// -- Define pivot position and value
		int pivotPos = row_ID * (width + 1);
		pivot = hostInput[pivotPos];

		// -- Normalise row relative to pivot		
		for (int col = 0; col < (width - row_ID); col++){
			hostInput[pivotPos + col] = hostInput[pivotPos + col] / pivot;
		}

		// -- Loop through j-th column and remove suitable multiples
		for (int row2_ID = 1; row2_ID < (height - row_ID); row2_ID++){

			float coeff = hostInput[pivotPos + (width * row2_ID)];

			for (int offset_col = 0; offset_col < (width - row_ID); offset_col++){

				// -- Update elements in row by subtracting elements with coeff
				hostInput[pivotPos + (width * row2_ID) + offset_col] = hostInput[pivotPos + (width * row2_ID) + offset_col]
					- (coeff * hostInput[pivotPos + offset_col]);
			}
			//printMatrix(height, width, hostInput);			
		}	
	}

	//return;

	// -- Go through all rows starting from second
	for (int row_ID = 1; row_ID < height; row_ID++){

		// -- Define pivot position and value
		int pivotPos = row_ID * (width + 1);
		pivot = hostInput[pivotPos];

		// -- Loop through j-th column and remove suitable multiples
		for (int row2_ID = 1; row2_ID < (row_ID + 1); row2_ID++){

			float coeff = hostInput[pivotPos - (width * row2_ID)];

			for (int offset_col = 0; offset_col < (width - row_ID); offset_col++){

				// -- Update elements in row by subtracting elements with coeff
				hostInput[pivotPos - (width * row2_ID) + offset_col] = hostInput[pivotPos - (width * row2_ID) + offset_col]
					- (coeff * hostInput[pivotPos + offset_col]);
			}

			// -- Print matrix
			//printMatrix(height, width);
		}

		//if (row_ID == 3) return;
	}

}

void printMatrix(int height, int width, float* mat) {

	printf("HERE'S A PRINTOUT OF THE MATRIX:\n");

	for (int i = 0; i < height; ++i){

		for (int k = 0; k < width; ++k){

			printf("[%.2f] ", mat[i * width + k]);
		}
		printf("\n");
	}
	printf("\n\n");

}

void swapRows(int row1, int row2, int height, int width){

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
		tempRow[i] = hostInput[startPos_row1 + i];
		
		// -- Swap 
		hostInput[startPos_row1 + i] = hostInput[startPos_row2 + i];

		hostInput[startPos_row2 + i] = tempRow[i];
	}

	free(tempRow);
	
	//int startPos = row1 *

}

int* readFromFile(int* input) {

	printf("Reading text file... \n");

	FILE *f = fopen("file.txt", "r");

	int i;

	for (i = 0; i < numOfInputElements; i++) {
		fscanf(f, "%d", &input[i]);

	}

	return input;
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
	}

	if (kernelID == 1)	M1_Controller(d_Matrix, h_Matrix, height, width);
	if (kernelID == 2)	M2_Controller(d_Matrix, h_Matrix, height, width);
	

	// -- Copy device matrix to host matrix
	err = cudaMemcpy(h_Matrix, d_Matrix, height * width * sizeof(float), cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

}

float checkSum(float* mat, int height, int width){

	float checkSum = 0.0;

	int startPos = width - 1;

	for (int row_ID = 0; row_ID < height; row_ID++){
		checkSum = checkSum + mat[startPos + (width * row_ID)];
	}

	return checkSum;

}