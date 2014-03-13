#include <cuda.h>
#include <stdio.h>
#include "datadef.h"
#include "LCRNG.cuh"

__global__ void fission_kernel(unsigned N, unsigned RNUM_PER_THREAD, unsigned RUN_FLAG, unsigned* active, unsigned * rxn , unsigned * index, unsigned * yield , unsigned * rn_bank, unsigned* done, float** scatterdat){

	
	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}       //return if out of bounds
	
	//remap to active
	//tid=active[tid];
	if(done[tid]){return;}

	//load rxn number, init values
	unsigned 	this_rxn 	= rxn[tid];
	unsigned 	this_yield 	= 0;
	unsigned 	inu 		= 0;
	float 		nu 			= 0.0;
	unsigned	rn 			= rn_bank[ tid ];

	//only do reactions with secondary neutrons
	if (rxn[tid] == 18 | rxn[tid] == 16 | rxn[tid] == 17 | rxn[tid] == 37 | rxn[tid] == 24 | rxn[tid] == 41){}
	else {return;} 

	//printf("in fission\n");

	if (this_rxn == 18){
		// load nu from arrays
		unsigned 	this_dex 	= index[tid];
	
		//load nu value, since e search has alrady been done!
		memcpy(&nu, &scatterdat[this_dex], sizeof(float));
		inu = (unsigned) nu;
	
		if((float)inu+get_rand(&rn) <= nu){
			this_yield = inu+1;
		}
		else{
			this_yield = inu;
		}
		//printf("nu %6.4E inu %u rn1 %6.4E yield %u\n",nu,inu,rn1,this_yield);
	}
	else if(this_rxn == 17){
		this_yield = 3;
	}
	else if(this_rxn == 16 | this_rxn==24 | rxn[tid] == 41){
		this_yield = 2;
	}

	// write output and terminate history
	yield[tid] = this_yield;
	done[tid]  = 1;    // pop will re-activate this data slot on fixed-source runs
	if (this_rxn == 18){rn_bank[tid] = rn;}  //rn was used for fission

}

void fission( cudaStream_t stream, unsigned NUM_THREADS, unsigned N, unsigned RNUM_PER_THREAD, unsigned RUN_FLAG, unsigned* active, unsigned * rxn , unsigned * index, unsigned * yield , unsigned * rn_bank, unsigned* done, float** scatterdat){

	unsigned blks = ( N + NUM_THREADS - 1 ) / NUM_THREADS;

	//fission_kernel <<< blks, NUM_THREADS >>> (   N,  RNUM_PER_THREAD, RUN_FLAG, active, rxn , index, yield , rn_bank, done, scatterdat);
	fission_kernel <<< blks, NUM_THREADS , 0 , stream >>> (   N,  RNUM_PER_THREAD, RUN_FLAG, active, rxn , index, yield , rn_bank, done, scatterdat);
	cudaThreadSynchronize();

}

