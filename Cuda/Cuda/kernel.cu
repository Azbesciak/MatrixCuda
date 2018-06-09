
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
#include <iostream>
#include <fstream>
#include <utility>

using namespace std;

#define WIN32
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#define N_ITER 10

//CONSTANTS
#define MATRIX_MAX_SIZE 4096
#define MATRIX_MIN_SIZE 0
#define EPS 1.e-6

#define DEBUG false

#if DEBUG
#define DEBUG_PRINT printf
#else
#define DEBUG_PRINT(...)
#endif

#define CHECK false

const string logFilename = "logfile.txt";

/**
* Matrix multiplication (CUDA Kernel) on the device: C = A * B
* *_off - offset in indexes
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

	int aBegin = n * BLOCK_SIZE * by;
	int aEnd = aBegin + n - 1;
	int aStep = BLOCK_SIZE;
	int bBegin = BLOCK_SIZE * bx;
	int bStep = BLOCK_SIZE * n;
	float Csub = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
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


// recursive variadic function
template<typename T,typename Element>
void dumpToFile(ofstream &logStream, T && item, Element && element)
{
	logStream << item << ";" << element << endl;
}
template<typename T, typename... Elements>
void dumpToFile(ofstream &logStream,T &&first, Elements && ...args)
{
	logStream << forward<T>(first) << ";";
	dumpToFile(logStream, forward<Elements>(args)...);
}


void htd_copy(const unsigned mat_mem_size, float* h,  float* d, cudaStream_t stream)
{
	checkCudaErrors(cudaMemcpyAsync(d, h, mat_mem_size, cudaMemcpyHostToDevice, stream));
}

void dth_copy(const unsigned mat_mem_size, float* d_C, float* h_C, cudaStream_t stream)
{
	checkCudaErrors(cudaMemcpyAsync(h_C, d_C, mat_mem_size, cudaMemcpyDeviceToHost, stream));
}

void ikj(float * a, float * b, float *c, int n) {
	for (int i = 0; i < n; i++)
		for (int k = 0; k < n; k++)
			for (int j = 0; j < n; j++)
				c[i*n + j] += a[i*n +k] * b[k*n +j];
}

void do_check(const int n, const int nstreams, const unsigned mat_size_in_1d, const unsigned uber_mat_size, const unsigned uber_mat_mem_size, float* h_a, float* h_b, float* h_c)
{
	printf("Checking computed result for correctness: ");
	float *cres = static_cast<float*>(malloc(uber_mat_mem_size));

	constantInit(cres, uber_mat_size, 0);
	for (int i = 0, off = 0; i < nstreams; i++, off += mat_size_in_1d)
		ikj(h_a + off, h_b + off, cres + off, n);
	float sum_org = 0, sum_cpy = 0;
	bool is_correct = true;
	for (int i = 0; i < uber_mat_size; i++)
	{
		sum_org += h_c[i];
		sum_cpy += cres[i];
		const double abs_err = fabs(h_c[i] - cres[i]);
		const double abs_val = fabs(h_c[i]);
		const double rel_err = abs_err / abs_val;
 
		if (rel_err > EPS) {
			DEBUG_PRINT("Error - too big inaccuracy! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
				i, temp_c[i], cres[i], EPS);
			is_correct = false;
		}
	}
	
	printf("%s\n", is_correct ? "OK" : "FAIL");
	if (!is_correct)
		printf("org- %f, cpy- %f, dif: %f \n", sum_org, sum_cpy, sum_org - sum_cpy);
 
	// Clean up memory
	free(cres);
}

/**
* Run a simple test of matrix multiplication using CUDA
*/
int matrixMultiply(const int block_size, const int n, const int nmats, const int nstreams, ofstream &logStream)
{
	// Allocate host memory for matrices A and B
	checkCudaErrors(cudaSetDeviceFlags(cudaDeviceBlockingSync | cudaDeviceMapHost));
	
	const unsigned int mat_size_in_1d = n*n;
	const unsigned int uber_mat_size = nmats * mat_size_in_1d;
	const unsigned int mat_mem_size = sizeof(float) * mat_size_in_1d;
	const unsigned int uber_mat_mem_size = mat_mem_size * nmats;
	float *h_a, *h_b, *h_c;
	checkCudaErrors(cudaMallocHost(&h_a, uber_mat_mem_size));
	checkCudaErrors(cudaMallocHost(&h_b, uber_mat_mem_size));
	checkCudaErrors(cudaMallocHost(&h_c, uber_mat_mem_size));

	randomInit(h_a, uber_mat_size);
	randomInit(h_b, uber_mat_size);
	// Allocate device memory
	float *d_A, *d_B, *d_C;
	
	//nstreams for async communication
	cudaStream_t *streams = static_cast<cudaStream_t *>(malloc(nstreams * sizeof(cudaStream_t)));
	for (int i = 0; i < nstreams; i++)
		checkCudaErrors(cudaStreamCreate(&streams[i]));

	if (h_c == NULL)
	{
		fprintf(stderr, "Failed to allocate host matrix C!\n");
		exit(EXIT_FAILURE);
	}

	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), uber_mat_mem_size));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), uber_mat_mem_size));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), uber_mat_mem_size));

	// Setup execution parameters
	
	
	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
	checkCudaErrors(cudaEventCreateWithFlags(&stop, cudaEventBlockingSync));

	// Record the start event
	checkCudaErrors(cudaEventRecord(start, NULL));
	
	const int grid_size = n / block_size;
	dim3 threads(block_size, block_size); // ALWAYS EQUAL TO BLOCK SIZE
	dim3 grid(grid_size, grid_size);
	for (int iter = 0; iter < N_ITER; iter++)
	{
		DEBUG_PRINT("iteration %d\n", iter);

		for (int i = 0, off = 0; i < nmats; i++, off += mat_size_in_1d)
		{
			const int stream_index = i % nstreams;
			float* h_a_step = h_a + off;
			float* d_a_step = d_A + off;
			htd_copy(mat_mem_size, h_a_step, d_a_step, streams[stream_index]);

			float* h_b_step = h_b + off;
			float* d_b_step = d_B + off;
			htd_copy(mat_mem_size, h_b_step, d_b_step, streams[stream_index]);

			float* d_c_step = d_C + off;
			float* h_c_step = h_c + off;
			switch (block_size)
			{
			case 8: matrixMulCUDA<8> << < grid, threads, 0, streams[stream_index] >> > (d_c_step, d_a_step, d_b_step, n); break;
			case 16: matrixMulCUDA<16> << < grid, threads, 0, streams[stream_index] >> > (d_c_step, d_a_step, d_b_step, n); break;
			case 32: matrixMulCUDA<32> << < grid, threads, 0, streams[stream_index] >> > (d_c_step, d_a_step, d_b_step, n); break;
			}
			dth_copy(mat_mem_size, d_c_step, h_c_step, streams[stream_index]);
		}
		cudaDeviceSynchronize();
	}

	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop, NULL));

	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
	// Compute and print the performance
	const float msecPerMatrixMul = msecTotal / N_ITER;
	const double flopsPerMatrixMul = 2.0 * n * n * n*nmats;
	const double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
	int threadsSize = threads.x * threads.y;
	printf(
		"Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
					gigaFlops, msecPerMatrixMul,flopsPerMatrixMul,threadsSize);

	dumpToFile(logStream, n, nstreams, gigaFlops, msecPerMatrixMul,
				flopsPerMatrixMul, threadsSize, grid_size, block_size);
	if (CHECK)
		do_check(n, nstreams, mat_size_in_1d, uber_mat_size, uber_mat_mem_size, h_a, h_b, h_c);
	for (int i = 0; i < nstreams; i++)
		checkCudaErrors(cudaStreamDestroy(streams[i]));
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c);
	free(streams);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaDeviceReset();
	return true ? EXIT_SUCCESS : EXIT_FAILURE;
}

bool is_n_correct(int n)
{
	return n >= MATRIX_MIN_SIZE && n <= MATRIX_MAX_SIZE;
}


int get_cmd_arg(const int argc, char **argv, const char * flag, int def)
{
	if (checkCmdLineFlag(argc, const_cast<const char **>(argv), flag))
	{
		def = getCmdLineArgumentInt(argc, const_cast<const char **>(argv), flag);
	}
	return def;
}

int get_n(const int argc, char **argv, const int block_size)
{
	const int n = get_cmd_arg(argc, argv, "n", 512);
	if (!is_n_correct(n))
	{
		printf("N=%d is incorrect. n should be in the range [%d, %d].\n", n, MATRIX_MIN_SIZE, MATRIX_MAX_SIZE);
		exit(EXIT_FAILURE);
	}
	if (n % block_size != 0)
	{
		printf("n should be multiplication of %d, got %d\n", block_size, n);
		exit(EXIT_FAILURE);
	}
	return n;
}


int get_s(const int argc, char **argv, const int n_mats)
{
	const int s = get_cmd_arg(argc, argv, "s", 1);
	if (n_mats % s != 0)
	{
		printf("number of streams must be a natural divider of n of matrixes, got %d\n", s);
		exit(EXIT_FAILURE);
	}
	return s;
}

int get_block_size(const int argc, char **argv)
{
	const int block_size = get_cmd_arg(argc, argv, "b", 32);
	switch (block_size)
	{
	case 8: case 16: case 32: return block_size;
	default: {
		printf("block size must be 8/16/32, received %d\n", block_size);
		exit(EXIT_FAILURE);
	}
	}
}

int get_n_mats(const int argc, char **argv)
{
	return get_cmd_arg(argc, argv, "m", 1);
}


/**
* Program main
*/
int main(int argc, char **argv)
{
	ofstream logStream;
	logStream.open(logFilename.c_str(), ofstream::out | ofstream::app);
	if (!logStream.is_open()) {
		printf("Failed to open log file %s\n", logFilename.c_str());
		exit(EXIT_FAILURE);
	}
	// dumpToFile(logStream, "n", "Number of streams", "Performance [GFlops]", "Time [ms]", "Flops per matrix", "WorkgroupSize", "Grid size", "Block size");
	printf("[Matrix Multiply Using CUDA] - Starting...\n");
	// By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
	int devID = 0;
	checkCudaErrors(cudaGetDevice(&devID));
	cudaDeviceProp device_prop;
	checkCudaErrors(cudaGetDeviceProperties(&device_prop, devID));

	if (device_prop.computeMode == cudaComputeModeProhibited)
	{
		fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
		exit(EXIT_FAILURE);
	}

	printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, device_prop.name, device_prop.major, device_prop.minor);
	

	const int block_size = get_block_size(argc, argv);
	const int n = get_n(argc, argv, block_size);
	const int n_mats = get_n_mats(argc, argv);
	const int streams = get_s(argc, argv, n_mats);
	printf("Matrix(%d,%d) x %d // streams %d [block %d]\n", n, n, n_mats, streams, block_size);
	const int result = matrixMultiply(block_size, n, n_mats, streams, logStream);

	logStream.close();
	exit(result);
}
