
/**
* Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/**
* Matrix multiplication: C = A * B.
* Host code.
*
* This sample implements matrix multiplication as described in Chapter 3
* of the programming guide.
* It has been written for clarity of exposition to illustrate various CUDA
* programming principles, not with the goal of providing the most
* performant generic kernel for matrix multiplication.
*
* See also:
* V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
* in Proc. 2008 ACM/IEEE Conf. on Superconducting (SC '08),
* Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
*/

// System includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define WIN32
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

//CONSTANTS
const int MATRIX_MAX_SIZE = 2048;
const int MATRIX_MIN_SIZE = 0;
const double eps = 1.e-4;  // machine zero


bool isNCorrect(int n);

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

/**
* Matrix multiplication (CUDA Kernel) on the device: C = A * B
* n is A's width and wB is B's width
*/
template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float *C, float *A, float *B, int n)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = n * BLOCK_SIZE * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + n - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * n;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	float Csub = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin;
		a <= aEnd;
		a += aStep, b += bStep)
	{

		// Declaration of the shared memory array As used to
		// store the sub-matrix of A
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

		// Declaration of the shared memory array Bs used to
		// store the sub-matrix of B
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		As[ty][tx] = A[a + n * ty + tx];
		Bs[ty][tx] = B[b + n * ty + tx];

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
#pragma unroll

		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
			Csub += As[ty][k] * Bs[k][tx];
		}

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	int c = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + n * ty + tx] = Csub;
}

void constantInit(float *data, int size, float val)
{
	for (int i = 0; i < size; ++i)
	{
		data[i] = val;
	}
}

/**
* Run a simple test of matrix multiplication using CUDA
*/
int matrixMultiply(int argc, char **argv, int block_size, dim3 &dim, bool async)
{
	// Allocate host memory for matrices A and B
	unsigned int size_A = dim.x * dim.y;
	unsigned int mat_mem_size = sizeof(float) * size_A;
	float *h_A = (float *)malloc(mat_mem_size);
	float *h_B = (float *)malloc(mat_mem_size);

	// Initialize host memory
	const float valB = 0.01f;
	constantInit(h_A, size_A, 1.0f);
	constantInit(h_B, size_A, valB);

	// Allocate device memory
	float *d_A, *d_B, *d_C;

	float *h_C = (float *)malloc(mat_mem_size);

	if (h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host matrix C!\n");
		exit(EXIT_FAILURE);
	}

	cudaError_t	cuda_last_operation_status = cudaMalloc((void **)&d_A, mat_mem_size);
	if (cuda_last_operation_status != cudaSuccess)
	{
		printf("cudaMalloc d_A returned error code %d, line(%d)\n", cuda_last_operation_status, __LINE__);
		exit(EXIT_FAILURE);
	}

	cuda_last_operation_status = cudaMalloc((void **)&d_B, mat_mem_size);

	if (cuda_last_operation_status != cudaSuccess)
	{
		printf("cudaMalloc d_B returned error code %d, line(%d)\n", cuda_last_operation_status, __LINE__);
		exit(EXIT_FAILURE);
	}

	cuda_last_operation_status = cudaMalloc((void **)&d_C, mat_mem_size);

	if (cuda_last_operation_status != cudaSuccess)
	{
		printf("cudaMalloc d_C returned error code %d, line(%d)\n", cuda_last_operation_status, __LINE__);
		exit(EXIT_FAILURE);
	}

	// copy host memory to device
	cuda_last_operation_status = cudaMemcpy(d_A, h_A, mat_mem_size, cudaMemcpyHostToDevice);

	if (cuda_last_operation_status != cudaSuccess)
	{
		printf("cudaMemcpy (d_A,h_A) returned error code %d, line(%d)\n", cuda_last_operation_status, __LINE__);
		exit(EXIT_FAILURE);
	}

	cuda_last_operation_status = cudaMemcpy(d_B, h_B, mat_mem_size, cudaMemcpyHostToDevice);

	if (cuda_last_operation_status != cudaSuccess)
	{
		printf("cudaMemcpy (d_B,h_B) returned error code %d, line(%d)\n", cuda_last_operation_status, __LINE__);
		exit(EXIT_FAILURE);
	}

	// Setup execution parameters
	dim3 threads(block_size, block_size);
	dim3 grid(dim.x / threads.x, dim.y / threads.y);

	// Create and start timer
	printf("Computing result using CUDA Kernel...\n");

	// Performs warmup operation using matrixMul CUDA kernel
	switch (block_size)
	{
	case 8: matrixMulCUDA<8> << < grid, threads >> > (d_C, d_A, d_B, dim.x); break;
	case 16: matrixMulCUDA<16> << < grid, threads >> > (d_C, d_A, d_B, dim.x); break;
	case 32: matrixMulCUDA<32> << < grid, threads >> > (d_C, d_A, d_B, dim.x); break;
	}
	printf("done\n");

	cudaDeviceSynchronize();

	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start;
	cuda_last_operation_status = cudaEventCreate(&start);

	if (cuda_last_operation_status != cudaSuccess)
	{
		fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(cuda_last_operation_status));
		exit(EXIT_FAILURE);
	}

	cudaEvent_t stop;
	cuda_last_operation_status = cudaEventCreate(&stop);

	if (cuda_last_operation_status != cudaSuccess)
	{
		fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(cuda_last_operation_status));
		exit(EXIT_FAILURE);
	}

	// Record the start event
	cuda_last_operation_status = cudaEventRecord(start, NULL);

	if (cuda_last_operation_status != cudaSuccess)
	{
		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(cuda_last_operation_status));
		exit(EXIT_FAILURE);
	}

	// Execute the kernel
	int nIter = 1;

	for (int j = 0; j < nIter; j++)
	{
		switch (block_size)
		{
		case 8: matrixMulCUDA<8> << < grid, threads >> > (d_C, d_A, d_B, dim.x); break;
		case 16: matrixMulCUDA<16> << < grid, threads >> > (d_C, d_A, d_B, dim.x); break;
		case 32: matrixMulCUDA<32> << < grid, threads >> > (d_C, d_A, d_B, dim.x); break;
		}
	}

	// Record the stop event
	cuda_last_operation_status = cudaEventRecord(stop, NULL);

	if (cuda_last_operation_status != cudaSuccess)
	{
		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(cuda_last_operation_status));
		exit(EXIT_FAILURE);
	}

	// Wait for the stop event to complete
	cuda_last_operation_status = cudaEventSynchronize(stop);

	if (cuda_last_operation_status != cudaSuccess)
	{
		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(cuda_last_operation_status));
		exit(EXIT_FAILURE);
	}

	float msecTotal = 0.0f;
	cuda_last_operation_status = cudaEventElapsedTime(&msecTotal, start, stop);

	if (cuda_last_operation_status != cudaSuccess)
	{
		fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(cuda_last_operation_status));
		exit(EXIT_FAILURE);
	}

	// Compute and print the performance
	float msecPerMatrixMul = msecTotal / nIter;
	double flopsPerMatrixMul = 2.0 * (double)dim.x * (double)dim.y * (double)dim.x;
	double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
	printf(
		"Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
		gigaFlops,
		msecPerMatrixMul,
		flopsPerMatrixMul,
		threads.x * threads.y);

	// Copy result from device to host
	cuda_last_operation_status = cudaMemcpy(h_C, d_C, mat_mem_size, cudaMemcpyDeviceToHost);

	if (cuda_last_operation_status != cudaSuccess)
	{
		printf("cudaMemcpy (h_C,d_C) returned error code %d, line(%d)\n", cuda_last_operation_status, __LINE__);
		exit(EXIT_FAILURE);
	}

	printf("Checking computed result for correctness: ");
	bool correct = true;
	const double dot_length = dim.x;
	for (int i = 0; i < (int)(dim.x * dim.y); i++)
	{
		const double abs_err = fabs(h_C[i] - (dim.x * valB));
		const double abs_val = fabs(h_C[i]);
		const double rel_err = abs_err / abs_val / dot_length;

		if (rel_err > eps) {
			printf("Error - too big inaccuracy! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
				i, h_C[i], dim.x * valB, eps);
			correct = false;
		}
	}

	printf("%s\n", correct ? "OK" : "FAIL");

	// Clean up memory
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaDeviceReset();
	return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}

/**
* Program main
*/
int main(int argc, char **argv)
{
	printf("[Matrix Multiply Using CUDA] - Starting...\n");
	// By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
	int devID = 0;
	cudaError_t error = cudaGetDevice(&devID);

	if (error != cudaSuccess)
	{
		printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
	}

	cudaDeviceProp device_prop;
	error = cudaGetDeviceProperties(&device_prop, devID);

	if (device_prop.computeMode == cudaComputeModeProhibited)
	{
		fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
		exit(EXIT_SUCCESS);
	}

	if (error != cudaSuccess)
	{
		printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
	}
	else
	{
		printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, device_prop.name, device_prop.major, device_prop.minor);
	}

	// Use a larger block size for Fermi and above
	const int block_size = (device_prop.major < 2) ? 16 : 32;

	dim3 dim(20 * block_size, 20 * block_size, 1);
	// width of Matrix A
	int n = 300; 
	if (!isNCorrect(n))
	{
		//try get N value from comman line arguments
		if (checkCmdLineFlag(argc, (const char **)argv, "n"))
		{
			n = getCmdLineArgumentInt(argc, (const char **)argv, "n");
			if (!isNCorrect(n))
			{
				printf("N=%d is incorrect. n should be in the range [%d, %d].\n", n, MATRIX_MIN_SIZE, MATRIX_MAX_SIZE);
				exit(0);
			}
		}
	}

	dim.x = n;
	dim.y = n;
	
	printf("Matrix(%d,%d)\n", dim.x, dim.y);

	const int matrix_result = matrixMultiply(argc, argv, block_size, dim, false);

	exit(0);
}



bool isNCorrect(int n)
{
	return (n > MATRIX_MIN_SIZE && n < MATRIX_MAX_SIZE);
}