#include <cuda.h>
#include <stdio.h>
#include "datadef.h"
#include "wfloat3.h"

__global__ void print_histories_kernel( unsigned N , unsigned* isonum, unsigned* rxn, source_point* space, float* E, unsigned* done, unsigned* yield){

	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}       //return if out of bounds
	//if (done[tid]){return;}
	
	printf("tid=%d isonum=%u rxn=%u yield=%u (x,y,z)= % 6.4E % 6.4E % 6.4E, E=% 6.4E\n",tid,isonum[tid],rxn[tid],yield[tid],space[tid].x,space[tid].y,space[tid].z,E[tid]);

}

void print_histories( unsigned NUM_THREADS, unsigned N, unsigned* isonum, unsigned* rxn, source_point* space, float* E, unsigned* done, unsigned* yield){

	unsigned blks = ( N + NUM_THREADS - 1 ) / NUM_THREADS;

	print_histories_kernel <<< blks, NUM_THREADS >>> ( N, isonum, rxn, space, E, done, yield);
	cudaThreadSynchronize();

}

