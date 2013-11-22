#include <cuda.h>
#include <stdio.h>
#include "datadef.h"

__global__ void sample_fission_spectra_kernel (unsigned N, unsigned* active, float* rn_bank , float * E){

	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}

	//remap to active
	//tid=active[tid];

	//monoenergetic for now
	E[tid]=0.001;

}

void sample_fission_spectra( unsigned NUM_THREADS, unsigned N, unsigned* active, float * rn_bank, float * E ){

	unsigned blks = ( N + NUM_THREADS - 1 ) / NUM_THREADS;

	sample_fission_spectra_kernel <<< blks, NUM_THREADS >>> (  N, active, rn_bank, E  );
	cudaThreadSynchronize();

}

