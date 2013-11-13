#include <cuda.h>
#include <stdio.h>
#include "datadef.h"

__global__ void sample_fission_spectra_kernel (unsigned N, float* rn_bank , float * E){

	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}

	//monoenergetic for now
	E[tid]=0.001;

}

void sample_fission_spectra(unsigned blks, unsigned NUM_THREADS, unsigned N, float * rn_bank, float * E ){

	sample_fission_spectra_kernel <<< blks, NUM_THREADS >>> (  N, rn_bank, E  );
	cudaThreadSynchronize();

}

