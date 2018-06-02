
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
#include <helper_cuda.h>

#define N_ITER 10

const char *sSDKsample = "simpleStreams";

const char *sEventSyncMethod[] =
{
	"cudaEventDefault",
	"cudaEventBlockingSync",
	"cudaEventDisableTiming",
	NULL
};

const char *sDeviceSyncMethod[] =
{
	"cudaDeviceScheduleAuto",
	"cudaDeviceScheduleSpin",
	"cudaDeviceScheduleYield",
	"INVALID",
	"cudaDeviceScheduleBlockingSync",
	NULL
};


//CONSTANTS
const int MATRIX_MAX_SIZE = 4096;
const int MATRIX_MIN_SIZE = 0;
const double eps = 1.e-4;  // machine zero


bool is_n_correct(int n);

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
		data[i] = val;
}

void randomInit(float * data, int size)
{
	for (int i = 0; i < size; ++i)
		data[i] = (float)rand() / RAND_MAX;
}

void htd_copy(const unsigned mat_mem_size, float* h,  float* d, cudaStream_t stream)
{
	checkCudaErrors(cudaMemcpyAsync(d, h, mat_mem_size, cudaMemcpyHostToDevice, stream));
}

void dth_copy(const unsigned mat_mem_size, float* d_C, float* h_C, cudaStream_t stream)
{
	checkCudaErrors(cudaMemcpyAsync(h_C, d_C, mat_mem_size, cudaMemcpyDeviceToHost, stream));
}

void transpose(float * mat, int n)
{
	for (int i = 0; i < n; i++)
		for (int j = i+1; j < n; j++)
		{
			float temp = mat[i*n + j];
			mat[i*n + j] = mat[j*n + i];
			mat[j*n + i] = temp;
		}
}
void ikj(float * a, float * b, float *c, int n) {
	for (int i = 0; i < n; i++) {
		for (int k = 0; k < n; k++) {
			for (int j = 0; j < n; j++) {
				c[i*n + j] += a[i*n +k] * b[k*n +j];
			}
		}
	}
}
/**
* Run a simple test of matrix multiplication using CUDA
*/
int matrixMultiply(int block_size, dim3 &dim, int nstreams = 1)
{
	const int n = dim.x;
	if (nstreams < 1 || nstreams > n)
	{
		printf("Number of nstreams should be in the range [1, %d] in.\n", n);
		exit(0);
	}
	if (n % nstreams != 0)
	{
		printf("N should be a multiple of the number of nstreams.");
		exit(0);
	}
	// Allocate host memory for matrices A and B
	checkCudaErrors(cudaSetDeviceFlags(cudaDeviceBlockingSync | cudaDeviceMapHost));
	
	const unsigned int mat_size_in_1d = n*n;
	const unsigned int mat_mem_size = sizeof(float) * mat_size_in_1d;
	float *h_a, *h_b, *h_c;
	checkCudaErrors(cudaMallocHost(&h_a, mat_mem_size));
	checkCudaErrors(cudaMallocHost(&h_b, mat_mem_size));
	checkCudaErrors(cudaMallocHost(&h_c, mat_mem_size));

	// Initialize host memory
	const float valB = 0.01f;
	randomInit(h_a, mat_size_in_1d);
	randomInit(h_b, mat_size_in_1d);
	//constantInit(h_a, mat_size, 1.0f);
	//constantInit(h_b, mat_size, valB);
	transpose(h_b, n);
	// Allocate device memory
	float *d_A, *d_B, *d_C;


	//nstreams for async communication
	cudaStream_t *streams = (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));
	for (int i = 0; i < nstreams; i++)
	{
		checkCudaErrors(cudaStreamCreate(&streams[i]));
	}

	if (h_c == NULL)
	{
		fprintf(stderr, "Failed to allocate host matrix C!\n");
		exit(EXIT_FAILURE);
	}

	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mat_mem_size));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mat_mem_size));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mat_mem_size));

	// Setup execution parameters
	dim3 threads(block_size, block_size);
	dim3 grid(dim.x / threads.x, dim.y / threads.y);

	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
	checkCudaErrors(cudaEventCreateWithFlags(&stop, cudaEventBlockingSync));

	// Record the start event
	checkCudaErrors(cudaEventRecord(start, NULL));
	
	const int step = mat_size_in_1d / nstreams;
	const unsigned int chunk_mem_size = mat_mem_size / nstreams;
	for (int j = 0; j < N_ITER; j++)
	{
		printf("iteration %d\n", j);
		
		for (int i = 0, off = 0; i < nstreams; i++, off += step)
		{
			float* h_a_step = h_a+ off; float* d_a_step = d_A + off;
			htd_copy(chunk_mem_size, h_a_step, d_a_step, streams[i]);

			float* h_b_step = h_b + off; float* d_b_step = d_B + off;
			htd_copy(chunk_mem_size, h_b_step, d_b_step, streams[i]);

			float* d_c_step = d_C+ off; float* h_c_step = h_c+ off;
			switch (block_size)
			{
			case 8: matrixMulCUDA<8> <<< grid, threads, 0, streams[i] >>> (d_c_step, d_a_step, d_b_step, dim.x); break;
			case 16: matrixMulCUDA<16> <<< grid, threads, 0, streams[i] >>> (d_c_step, d_a_step, d_b_step, dim.x); break;
			case 32: matrixMulCUDA<32> <<< grid, threads, 0, streams[i] >>> (d_c_step, d_a_step, d_b_step, dim.x); break;
			}
			dth_copy(chunk_mem_size, d_c_step, h_c_step, streams[i]);
		}
	}

	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop, NULL));

	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
	// Compute and print the performance
	float msecPerMatrixMul = msecTotal / N_ITER;
	double flopsPerMatrixMul = 2.0 * (double)dim.x * (double)dim.y * (double)dim.x;
	double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
	printf(
		"Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
		gigaFlops,
		msecPerMatrixMul,
		flopsPerMatrixMul,
		threads.x * threads.y);

	printf("Checking computed result for correctness: ");
	bool correct = true;
	float *cres = static_cast<float*>(malloc(mat_mem_size));
	constantInit(cres, mat_size_in_1d, 0);
	//transpose(h_b, n);
	ikj(h_a, h_b, cres, n);
	for (int i = 0; i < mat_size_in_1d; i++)
	{
		const double abs_err = fabs(h_c[i] - cres[i]);
		const double abs_val = fabs(h_c[i]);
		const double rel_err = abs_err / abs_val / n;

		if (rel_err > eps) {
			printf("Error - too big inaccuracy! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
				i, h_c[i], n * valB, eps);
			correct = false;
		}
	}

	printf("%s\n", correct ? "OK" : "FAIL");

	// Clean up memory
	free(h_a);
	free(h_b);
	free(h_c);
	free(streams);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaDeviceReset();
	return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}

bool is_n_correct(int n)
{
	return (n >= MATRIX_MIN_SIZE && n <= MATRIX_MAX_SIZE);
}

int get_n(int argc, char **argv, int block_size)
{
	int n = 512;
	if (checkCmdLineFlag(argc, (const char **)argv, "n"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "n");
	}
	if (!is_n_correct(n))
	{
		printf("N=%d is incorrect. n should be in the range [%d, %d].\n", n, MATRIX_MIN_SIZE, MATRIX_MAX_SIZE);
		exit(-1);
	}
	if (n % block_size != 0)
	{
		printf("n should be multiplication of %d", block_size);
		exit(0);
	}
	return n;
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
	const int n = get_n(argc, argv, block_size);
	dim.x = dim.y = n;
	printf("Matrix(%d,%d)\n", dim.x, dim.y);
	if (checkCmdLineFlag(argc, const_cast<const char **>(argv), "a")) //async
	{
		const int streams = (n / block_size);
		matrixMultiply(block_size, dim, streams);
	} else
	{
		matrixMultiply(block_size, dim);
	}
	exit(0);
}
