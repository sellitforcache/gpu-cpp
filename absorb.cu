#include <cuda.h>
#include <stdio.h>
#include "datadef.h"

__global__ void absorb_kernel(unsigned N, unsigned * rxn , unsigned* done){


	//PLACEHOLDER FOR FISSIONS, NEED TO READ NU TABLES LATER
	
	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}         //return if out of bounds
	if (done[tid]){return;}        // return if done, duh
	if (rxn[tid] < 102 ){return;}  //return if not some sort of absorption, ie (n,not-n)

	//printf("in abs, rxn=%u\n",rxn[tid]);

	done[tid]  = 1;

}

void absorb(unsigned blks, unsigned NUM_THREADS, unsigned N, unsigned * rxn , unsigned* done){

	absorb_kernel <<< blks, NUM_THREADS >>> (  N,  rxn , done);
	cudaThreadSynchronize();

}

