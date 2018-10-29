#include <stdio.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <cooperative_groups.h>
#include <stdlib.h>
#include <time.h>
const int NUM_BLOCKS=2048*16;
const int  NUM_THREADS=512;
const int maxn = 1e7;
using namespace cooperative_groups;
const int test_size = 1e7;
enum type{global_ = 0,shared_,warp_};
size_t size = sizeof(int)*maxn;

int filter(int *dst, const int *src, int n) {
	int nres = 0;
	for (int i = 0; i < n; i++)
		if (src[i] > 0)
			dst[nres++] = src[i];
	// return the number of elements copied
	return nres;
}

__global__
void filter_global_k(int *dst, int *nres, const int *src, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n && src[i] > 0)
			dst[atomicAdd(nres, 1)] = src[i];
	
}

__global__
void filter_shared_k(int *dst, int *nres, const int* src, int n) {
	__shared__ int l_n;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

		// zero the counter
		if (threadIdx.x == 0)
			l_n = 0;
		__syncthreads();

		// get the value, evaluate the predicate, and
		// increment the counter if needed
		int d, pos;

		if (i < n) {
			d = src[i];
			if (d > 0)
				pos = atomicAdd(&l_n, 1);
		}
		__syncthreads();

		// leader increments the global counter
		if (threadIdx.x == 0)
			l_n = atomicAdd(nres, l_n);
		__syncthreads();

		// threads with true predicates write their elements
		if (i < n && d > 0) {
			pos += l_n; // increment local pos by global counter
			dst[pos] = d;
		}
		__syncthreads();
}

__device__ int atomicAggInc(int *ctr) {
	auto g = coalesced_threads();
	int warp_res;
	if (g.thread_rank() == 0) 
		warp_res = atomicAdd(ctr, g.size());
	return g.shfl(warp_res, 0) + g.thread_rank();
}

__global__ void filter_k(int *dst, const int *src, int n,int *nres) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n)
		return;
	if (src[i] > 0)
		dst[atomicAggInc(nres)] = src[i];
}


void filted_by_cpu(int *dst, const int *src) {
	clock_t begin, end;
	begin = clock();
	int num= filter(dst, src, test_size);
	end = clock();
	float time = float(end - begin);
	printf("The total time by CPU is %f\nThe number is %d\n",time,num);
}


void filted_by_global(int *d, const int *s,type t) {
	int *src, *dst;
	cudaMalloc((void**)&src, size);
	cudaMalloc((void**)&dst, size);
	int k = 0;
	int *nres;
	cudaMalloc((void**)&nres, sizeof(int));
	cudaMemcpy(nres, &k, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(src, s, size, cudaMemcpyHostToDevice);
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	switch (t)
	{
	case global_:  filter_global_k<<<NUM_BLOCKS,NUM_THREADS>>>(dst, nres, src, test_size);
		break;
	case shared_:  filter_shared_k << <NUM_BLOCKS, NUM_THREADS >> > (dst, nres, src, test_size);
		break;
	case warp_:    filter_k << <NUM_BLOCKS, NUM_THREADS >> > (dst, src, test_size,nres);
		break;
	default:
		break;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaMemcpy(&k, nres, sizeof(int), cudaMemcpyDeviceToHost);
	switch (t)
	{
	case global_:	printf("The total time by global is %f\nThe number is %d\n", time, k);
		break;
	case shared_:	printf("The total time by shared is %f\nThe number is %d\n", time, k);
		break;
	case warp_:	    printf("The total time by warp is %f\nThe number is %d\n", time, k);
		break;
	default: 	    printf("Unkown type\n"); return;
		break;
	}

	cudaFree(nres);
	cudaFree(src);
	cudaFree(dst);
}

void getfile(int *dataf,int n) {
	printf("Please input the filename:\n");
	char *filename=(char*)malloc(10*sizeof(char));
	scanf("%s", filename);
	FILE *fp = NULL;
	fp = fopen(filename, "r");
	if (fp == NULL) {
		printf("error:open file failure\n");
		getchar();
		exit(1);
	}
	int k;
	for (int i = 0; i < n; i++) {
		fscanf(fp,"%d", &k);
		dataf[i] = k;
	}
}

int main() {
	int* s = (int*)malloc(size);
	int* d = (int*)malloc(size);
	getfile(s,test_size);
	filted_by_cpu(d, s); 
	filted_by_global(d, s, global_);
	filted_by_global(d, s, shared_);
	filted_by_global(d, s, warp_);
	free(s);
	free(d);
	return 0;
}