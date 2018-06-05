
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


const string logFilename = "logfile.txt";

/**
* Matrix multiplication (CUDA Kernel) on the device: C = A * B
* *_off - offset in indexes
*/
template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float *C, float *A, float *B, int n, int grid_size)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;
	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int aBegin = n * BLOCK_SIZE * by;
	int aEnd = aBegin + n;
	int aStep = BLOCK_SIZE;
	int bBegin = n * BLOCK_SIZE * bx;
	int bStep = BLOCK_SIZE;
	float Csub = 0;

	for (int a = aBegin, b = bBegin; a < aEnd; a += aStep, b += bStep)
	{
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		As[ty][tx] = A[a + n * ty + tx];
		Bs[ty][tx] = B[b + n * tx + ty];

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
	int c = grid_size * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + grid_size * ty + tx] = Csub;
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

void change_lines_to_grid(float * lines, float * grid, int n, int chunk_n)
{
	const int gridSize = chunk_n * chunk_n;
	const int oneDISize = n * n;
	for (int oneDIndex = 0; oneDIndex < oneDISize; oneDIndex++)
		for(int j = 0; j < n; j++) {
			int gridIndex = oneDIndex / gridSize;
			int fieldIndexInGrid = oneDIndex % gridSize;
			int gridsInRow = n / chunk_n;
			int gridIoff = gridIndex / gridsInRow * chunk_n;
			int gridJoff = gridIndex % gridsInRow * chunk_n;
			int ii = fieldIndexInGrid / chunk_n;
			int jj = fieldIndexInGrid % chunk_n;
			grid[(gridIoff + ii) * n + gridJoff + jj] = lines[oneDIndex];
		}
}
/**
* Run a simple test of matrix multiplication using CUDA
*/
int matrixMultiply(const int block_size, const int n, const int nstreams, ofstream &logStream)
{
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

	randomInit(h_a, mat_size_in_1d);
	randomInit(h_b, mat_size_in_1d);
	transpose(h_b, n);
	// Allocate device memory
	float *d_A, *d_B, *d_C;
	
	//nstreams for async communication
	const int square_nstreams = nstreams * nstreams;
	cudaStream_t *streams = static_cast<cudaStream_t *>(malloc(square_nstreams * sizeof(cudaStream_t)));
	for (int i = 0; i < square_nstreams; i++)
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
	
	
	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
	checkCudaErrors(cudaEventCreateWithFlags(&stop, cudaEventBlockingSync));

	// Record the start event
	checkCudaErrors(cudaEventRecord(start, NULL));
	
	const int step = mat_size_in_1d / nstreams;
	const unsigned int chunk_mem_size = mat_mem_size / nstreams;
	const unsigned int result_chunk_mem_size = chunk_mem_size / nstreams;
	const int grid_size = n / block_size / nstreams;
	dim3 threads(block_size, block_size); // ALWAYS EQUAL TO BLOCK SIZE
	dim3 grid(grid_size, grid_size);
	const int result_chunk_n = n / nstreams;
	for (int iter = 0; iter < N_ITER; iter++)
	{
		DEBUG_PRINT("iteration %d\n", iter);
		
		for (int a_i = 0, a_off = 0; a_i < nstreams; a_i++, a_off += step)
		{
			for (int b_i = 0, b_off = 0; b_i < nstreams; b_i++, b_off += step)
			{
				
				float* h_a_step = h_a + a_off;
				float* d_a_step = d_A + a_off;
				const int stream_index = a_i*nstreams + b_i;
				htd_copy(chunk_mem_size, h_a_step, d_a_step, streams[stream_index]);

				float* h_b_step = h_b + b_off;
				float* d_b_step = d_B + b_off;
				htd_copy(chunk_mem_size, h_b_step, d_b_step, streams[stream_index]);

				const int result_off = a_off + b_off / nstreams;
				DEBUG_PRINT("res off: %d\n", result_off);
				float* d_c_step = d_C + result_off;
				float* h_c_step = h_c + result_off;
				switch (block_size)
				{
				case 8: matrixMulCUDA<8> << < grid, threads, 0, streams[stream_index] >> > (d_c_step, d_a_step, d_b_step, n, result_chunk_n); break;
				case 16: matrixMulCUDA<16> << < grid, threads, 0, streams[stream_index] >> > (d_c_step, d_a_step, d_b_step, n, result_chunk_n); break;
				case 32: matrixMulCUDA<32> << < grid, threads, 0, streams[stream_index] >> > (d_c_step, d_a_step, d_b_step, n, result_chunk_n); break;
				}
				dth_copy(result_chunk_mem_size, d_c_step, h_c_step, streams[stream_index]);
			}
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
	float msecPerMatrixMul = msecTotal / N_ITER;
	double flopsPerMatrixMul = 2.0 * (double)n * (double)n * (double)n;
	double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
	int threadsSize = threads.x * threads.y;
	printf(
		"Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
		gigaFlops,
		msecPerMatrixMul,
		flopsPerMatrixMul,
		threadsSize
		);

	dumpToFile(logStream,
		n,
		nstreams,
		gigaFlops, 
		msecPerMatrixMul, 
		flopsPerMatrixMul, 
		threadsSize,
		grid_size,
		block_size);
	
	printf("Checking computed result for correctness: ");
	;
	float *cres = static_cast<float*>(malloc(mat_mem_size));
	float *temp_c = static_cast<float*>(malloc(mat_mem_size));

	constantInit(cres, mat_size_in_1d, 0);
	constantInit(temp_c, mat_size_in_1d, 0);
	transpose(h_b, n);
	ikj(h_a, h_b, cres, n);
	change_lines_to_grid(h_c, temp_c, n, result_chunk_n);
	float sum_org = 0, sum_cpy = 0;
	bool is_correct = true;
	for (int i = 0; i < mat_size_in_1d; i++)
	{
		sum_org += temp_c[i];
		sum_cpy += cres[i];
		const double abs_err = fabs(temp_c[i] - cres[i]);
		const double abs_val = fabs(temp_c[i]);
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
	free(temp_c);
	free(cres);
	for (int i = 0; i < square_nstreams; i++)
		checkCudaErrors(cudaStreamDestroy(streams[i]));
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c);
	free(streams);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaDeviceReset();
	return is_correct ? EXIT_SUCCESS : EXIT_FAILURE;
}

bool is_n_correct(int n)
{
	return n >= MATRIX_MIN_SIZE && n <= MATRIX_MAX_SIZE;
}


int get_n(const int argc, char **argv, const int block_size)
{
	int n = 512;
	if (checkCmdLineFlag(argc, const_cast<const char **>(argv), "n"))
	{
		n = getCmdLineArgumentInt(argc, const_cast<const char **>(argv), "n");
	}
	if (!is_n_correct(n))
	{
		printf("N=%d is incorrect. n should be in the range [%d, %d].\n", n, MATRIX_MIN_SIZE, MATRIX_MAX_SIZE);
		exit(EXIT_FAILURE);
	}
	if (n % block_size != 0)
	{
		printf("n should be multiplication of %d", block_size);
		exit(EXIT_FAILURE);
	}
	return n;
}


bool is_s_correct(int s, int n, int block_size)
{
	const int grid_size = n / block_size;
	return s > 0 && grid_size % s == 0;
}

int get_s(const int argc, char **argv, const int block_size, int n)
{
	int streams = 1;
	if (checkCmdLineFlag(argc, const_cast<const char **>(argv), "s"))
	{
		streams = getCmdLineArgumentInt(argc, const_cast<const char **>(argv), "s");
		if (!is_s_correct(streams, n, block_size))
		{
			printf("streams count should be greater than 0 and be divider of n / block_size (%d)\n", n);
			exit(EXIT_FAILURE);
		}
	}
	return streams;
}
/**
* Program main
*/
int main(int argc, char **argv)
{
	ofstream logStream;
	logStream.open(logFilename.c_str());
	if (!logStream.is_open()) {
		printf("Failed to open log file %s\n", logFilename.c_str());
		exit(-1);
	}
	dumpToFile(logStream, "n", "Number of streams", "Performance [GFlops]", "Time [ms]", "Flops per matrix", "WorkgroupSize", "Grid size", "Block size");
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

	// width of Matrix A
	const int n = get_n(argc, argv, block_size);
	const int streams = get_s(argc, argv, block_size, n);
	printf("Matrix(%d,%d) - streams %d\n", n, n, streams);
	const int result = matrixMultiply(block_size, n, streams, logStream);

	logStream.close();
	exit(result);
}
