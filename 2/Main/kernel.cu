#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <ctime>
#include <vector>
#include <omp.h>
#include <curand_kernel.h>

using namespace std;

clock_t c_start, c_end;
int n = 1024 * 1024 * 32;

__global__ void count_pi_1(float *dev_randX, float *dev_randY, int *dev_threads_num, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	int cont = 0;
	for (int i = tid * 128; i < 128 * (tid + 1); i++) {
		if (dev_randX[i] * dev_randX[i] + dev_randY[i] * dev_randY[i] < 1.0f) {
			cont++;
		}
	}
	dev_threads_num[tid] = cont;
}

__global__ void count_pi_2(float *dev_randX, float *dev_randY, int *dev_blocks_num, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ int count_pi_2[512];
	int cont = 0;
	for (int i = tid * 128; i < 128 * (tid + 1); i++) {
		if (dev_randX[i] * dev_randX[i] + dev_randY[i] * dev_randY[i] < 1.0f) {
			cont++;
		}
	}
	count_pi_2[threadIdx.x] = cont;

	__syncthreads();

	if (threadIdx.x == 0) {
		int total = 0;
		for (int j = 0; j < 512; j++) {
			total += count_pi_2[j];
		}
		dev_blocks_num[blockIdx.x] = total;
	}
}

__global__ void count_pi_3(float *dev_randX, float *dev_randY, int *dev_blocks_num, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;
	__shared__ int count_pi_3[512];
	int cont = 0;
	for (int i = tid; i < n; i += stride) {
		if (dev_randX[i] * dev_randX[i] + dev_randY[i] * dev_randY[i] < 1.0f) {
			cont++;
		}
	}
	count_pi_3[threadIdx.x] = cont;

	__syncthreads();

	if (threadIdx.x == 0) {
		int total = 0;
		for (int j = 0; j < 512; j++) {
			total += count_pi_3[j];
		}
		dev_blocks_num[blockIdx.x] = total;
	}
}


__global__ void count_pi_4(float *dev_randX, float *dev_randY, int *dev_threads_num, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;
	int cont = 0;
	for (int i = tid; i < n; i += stride) {
		if (dev_randX[i] * dev_randX[i] + dev_randY[i] * dev_randY[i] < 1.0f) {
			cont++;
		}
	}
	dev_threads_num[tid] = cont;
}

__host__ float count_pi_cpu(vector<float> randX, vector<float> randY, int n){
	int c_count = 0;
	for (int i = 0; i < n; i++) {
		if (randX[i] * randX[i] + randY[i] * randY[i] < 1.0f) {
			c_count++;
		}
	}
	float c_num = float(c_count) * 4.0 / n;
	return c_num;
}

__host__ float count_pi_cpu_omp(vector<float> randX, vector<float> randY, int n){
	int c_count = 0;
	#pragma omp for
	for (int i = 0; i < n; i++) {
		if (randX[i] * randX[i] + randY[i] * randY[i] < 1.0f) {
			c_count++;
		}
	}
	float c_num = float(c_count) * 4.0 / n;
	return c_num;
}

__global__ void setup_kernel(curandState *state)
{
	int id = threadIdx.x + blockIdx.x * 64;
	/* Each thread gets same seed, a different
	sequence number, no offset */
	curand_init(1234, id, 0, &state[id]);
}

__global__ void generate_kernel(curandState *state, float *result)
{
	int id = threadIdx.x + blockIdx.x * 64;
	float rand;

	/* Copy state to local memory for efficiency */
	curandState localState = state[id];
	/* Generate pseudo-random float */
	rand = curand_uniform(&localState);

	/* Copy state back to global memory */
	state[id] = localState;
	/* Store results */
	result[id] = rand;
}

int main() {
	
	curandState *devStates;
	float *devRandX, *devRandY;
	float *hostRandX = new float[n];
	float *hostRandY = new float[n];
	vector<float> randX(n);
	vector<float> randY(n);
	
	cudaMalloc( &devStates, n*sizeof( curandState ) );
    cudaMalloc( &devRandX, n*sizeof( *hostRandX ) );
	cudaMalloc( &devRandY, n*sizeof( *hostRandY ) );
	
	srand((unsigned)time(NULL));
	/*
	for (int i = 0; i < n; i++) {
		randX[i] = float(rand()) / RAND_MAX;
		randY[i] = float(rand()) / RAND_MAX;
	}
	*/
	
	setup_kernel <<<64, 64>>> ( devStates );
	
	generate_kernel <<<64, 64>>> ( devStates, devRandX );
	cudaMemcpy( hostRandX, devRandX, n*sizeof(*hostRandX), cudaMemcpyDeviceToHost );
	
	generate_kernel <<<64, 64>>> ( devStates, devRandY );
	cudaMemcpy( hostRandY, devRandY, n*sizeof(*hostRandY), cudaMemcpyDeviceToHost );
	
	cudaFree(devRandX);
	cudaFree(devRandY);
    cudaFree(devStates);
	
	#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		randX[i] = hostRandX[i];
		randY[i] = hostRandY[i];
	}
	
	cout << "randX[0]= " << randX[0] << endl;
	//start cont cpu time
	c_start = clock();
	int c_count = 0;
	//CPU calculate pi
	#pragma omp for
	for (int i = 0; i < n; i++) {
		if (randX[i] * randX[i] + randY[i] * randY[i] < 1.0f) {
			c_count++;
		}
	}
	//end cont cpu time
	c_end = clock();
	float c_num = float(c_count) * 4.0 / n;
	float t_cpu = (float)(c_end - c_start) / CLOCKS_PER_SEC;
	cout << "CPU_wo_func Time" << endl;
	cout << c_num << endl;
	cout << "time= " << t_cpu * 1000 << " ms" << endl;



	//start cont cpu time
	c_start = clock();
	c_num = count_pi_cpu(randX, randY, n);
	//end cont cpu time
	c_end = clock();
	t_cpu = (float)(c_end - c_start) / CLOCKS_PER_SEC;
	cout << "CPU Time" << endl;
	cout << c_num << endl;
	cout << "time= " << t_cpu * 1000 << " ms" << endl;
	
	//start cont cpu_omp time
	double dtime = omp_get_wtime();
	c_num = count_pi_cpu_omp(randX, randY, n);
	//end cont cpu_omp time
	dtime = omp_get_wtime() - dtime;
	cout << "CPU_OMP Time" << endl;
	cout << c_num << endl;
	cout << "time= " << dtime * 1000 << " ms" << endl;


	//start cont gpu time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//send data to GPU
	size_t size = n * sizeof(float);
	float *dev_randX;
	float *dev_randY;
	cudaMalloc((void**)&dev_randX, size);
	cudaMalloc((void**)&dev_randY, size);

	cudaMemcpy(dev_randX, &randX.front(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_randY, &randY.front(), size, cudaMemcpyHostToDevice);

	int threadsPerBlock = 512;
	int block_num = n / (128 * threadsPerBlock);
	int *dev_threads_num;
	cudaMalloc((void**)&dev_threads_num, n / 128 * sizeof(int));

	count_pi_1 <<<block_num, threadsPerBlock >>> (dev_randX, dev_randY, dev_threads_num, n);

	int* threads_num = new int[n / 128];
	cudaMemcpy(threads_num, dev_threads_num, n / 128 * sizeof(int), cudaMemcpyDeviceToHost);

	int g_count = 0;
	for (int i = 0; i < n / 128; i++) {
		g_count += threads_num[i];
	};

	//end cont gpu time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float t_gpu1;
	cudaEventElapsedTime(&t_gpu1, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	float g_num = float(g_count) * 4.0 / n;
	cout << "GPU_1 Time" << endl;
	cout << g_num << endl;
	cout << "time = " << t_gpu1 << " ms" << endl;


	//start cont gpu time
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	int *dev_blocks_num;
	cudaMalloc((void**)&dev_blocks_num, 512 * sizeof(int));

	cudaMemcpy(dev_randX, &randX.front(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_randY, &randY.front(), size, cudaMemcpyHostToDevice);

	count_pi_2 << <block_num, threadsPerBlock >> > (dev_randX, dev_randY, dev_blocks_num, n);

	int *blocks_num = new int[block_num];
	cudaMemcpy(blocks_num, dev_blocks_num, block_num * sizeof(int), cudaMemcpyDeviceToHost);

	g_count = 0;
	for (int i = 0; i < block_num; i++) {
		g_count += blocks_num[i];
	};

	//end cont gpu time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float t_gpu2;
	cudaEventElapsedTime(&t_gpu2, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	g_num = float(g_count) * 4.0 / n;
	cout << "GPU_2 Time" << endl;
	cout << g_num << endl;
	cout << "time = " << t_gpu2 << " ms" << endl;


	//start cont gpu time
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaMemset(dev_blocks_num, 0, sizeof(int));

	cudaMemcpy(dev_randX, &randX.front(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_randY, &randY.front(), size, cudaMemcpyHostToDevice);

	count_pi_3 << <block_num, threadsPerBlock >> > (dev_randX, dev_randY, dev_blocks_num, n);

	blocks_num = new int[block_num];
	cudaMemcpy(blocks_num, dev_blocks_num, block_num * sizeof(int), cudaMemcpyDeviceToHost);

	g_count = 0;
	for (int i = 0; i < block_num; i++) {
		g_count += blocks_num[i];
	};

	//end cont gpu time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float t_gpu3;
	cudaEventElapsedTime(&t_gpu3, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	g_num = float(g_count) * 4.0 / n;
	cout << "GPU_3 Time" << endl;
	cout << g_num << endl;
	cout << "time = " << t_gpu3 << " ms" << endl;


	//start cont gpu time
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaMemset(dev_threads_num, 0, sizeof(int));

	cudaMemcpy(dev_randX, &randX.front(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_randY, &randY.front(), size, cudaMemcpyHostToDevice);

	count_pi_4 << <block_num, threadsPerBlock >> > (dev_randX, dev_randY, dev_threads_num, n);

	threads_num = new int[n / 128];
	cudaMemcpy(threads_num, dev_threads_num, n / 128 * sizeof(int), cudaMemcpyDeviceToHost);

	g_count = 0;
	for (int i = 0; i < n / 128; i++) {
		g_count += threads_num[i];
	};

	//end cont gpu time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float t_gpu4;
	cudaEventElapsedTime(&t_gpu4, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	g_num = float(g_count) * 4.0 / n;
	cout << "GPU_4 Time" << endl;
	cout << g_num << endl;
	cout << "time = " << t_gpu4 << " ms" << endl;

	cudaFree(dev_randX);
	cudaFree(dev_randY);
	cudaFree(dev_threads_num);
	cudaFree(dev_blocks_num);
}