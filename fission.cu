#include <cuda.h>
#include <stdio.h>
#include "datadef.h"

__global__ void fission_kernel(unsigned N, unsigned RNUM_PER_THREAD, unsigned * rxn , unsigned * yield , float * rn_bank, unsigned* done){


	//PLACEHOLDER FOR FISSIONS, NEED TO READ NU TABLES LATER
	
	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}       //return if out of bounds
	if (done[tid]){return;}      // return if done, duh
	if (rxn[tid] != 18 ){return;}  //return if not fission

	float yield_avg = 2.53;
	unsigned this_yield = 0;

	float rn1 = rn_bank[ tid*RNUM_PER_THREAD + 10 ];

	if(2.0+rn1 < yield_avg){
		this_yield = 3;
	}
	else{
		this_yield = 2;
	}

	yield[tid] = this_yield;
	done[tid]  = 1;

}

void fission(unsigned blks, unsigned NUM_THREADS, unsigned N, unsigned RNUM_PER_THREAD, unsigned * rxn , unsigned * yield , float * rn_bank, unsigned* done){

	fission_kernel <<< blks, NUM_THREADS >>> (   N,  RNUM_PER_THREAD,  rxn , yield , rn_bank, done);
	cudaThreadSynchronize();

}

