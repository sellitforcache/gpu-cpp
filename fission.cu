#include <cuda.h>
#include <stdio.h>
#include "datadef.h"

__global__ void fission_kernel(unsigned N, unsigned RNUM_PER_THREAD, unsigned * rxn , unsigned * index, unsigned * yield , float * rn_bank, unsigned* done, float** scatterdat){


	//PLACEHOLDER FOR FISSIONS, NEED TO READ NU TABLES LATER
	
	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}       //return if out of bounds
	if (done[tid]){return;}      // return if done, duh
	if (rxn[tid] < 3 | rxn[tid] > 50 ){return;}  //return if no secondary neutron

	//printf("in fission\n");

	unsigned 	this_yield 	= 0;
//	unsigned 	inu 		= 0;
//	float 		nu 			= 0.0;
//
//	// load from arrays
//	unsigned 	this_dex 	= index[tid];
//	float * 	this_array 	= scatterdat[this_dex];
//	float 		rn1 		= rn_bank[ tid*RNUM_PER_THREAD + 11 ];
//
//	//load nu value, since e search has alrady been done!
//	memcpy(&nu, &this_array[1], sizeof(float));
//	inu = (unsigned) nu;
//
//	printf("nu=%6.4f inu=%u\n",nu,inu);
//
//	if((float)inu+rn1 < nu){
//		this_yield = inu+1;
//	}
//	else{
//		this_yield = inu;
//	}


	//sample fission spectrum and set to energy to be added to next 

	// write output and terminate history
	yield[tid] = this_yield;
	done[tid]  = 1;

}

void fission(unsigned blks, unsigned NUM_THREADS, unsigned N, unsigned RNUM_PER_THREAD, unsigned * rxn , unsigned * index, unsigned * yield , float * rn_bank, unsigned* done, float** scatterdat){

	fission_kernel <<< blks, NUM_THREADS >>> (   N,  RNUM_PER_THREAD,  rxn , index, yield , rn_bank, done, scatterdat);
	cudaThreadSynchronize();

}

