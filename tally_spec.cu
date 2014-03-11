#include <cuda.h>
#include <stdio.h>
#include "datadef.h"

__global__ void tally_spec_kernel(unsigned N, unsigned Ntally, unsigned tally_cell,  unsigned* active, source_point* space, float* E, float * tally_score, unsigned * tally_count, unsigned* done, unsigned* cellnum){

	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}
	//tid=active[tid];
	if (done[tid]){return;}
	if (cellnum[tid]!=tally_cell){return;}

	//int k;
	float 		my_E   			= E[tid];
	float 		macro_t 		= space[tid].macro_t;
	unsigned 	my_bin_index 	= 0;

	const float Emax 	= 20.00000;
	const float Emin 	=  1.0e-11;

	// determine bin number
	my_bin_index = logf(my_E/Emin)/logf(Emax/Emin)*(Ntally);

	//score the bins atomicly, could be bad if many neutrons are in a single bin since this will serialize their operations
	atomicAdd(&tally_score[my_bin_index], 1.0/macro_t);
	atomicInc(&tally_count[my_bin_index], 4294967295);

	//printf("%6.4E\n",macro_t);


}

void tally_spec(unsigned NUM_THREADS,  unsigned N, unsigned Ntally, unsigned tally_cell, unsigned* active, source_point * space, float* E, float * tally_score, unsigned * tally_count, unsigned* done, unsigned* cellnum){
	
	unsigned blks = ( N + NUM_THREADS - 1 ) / NUM_THREADS;

	tally_spec_kernel <<< blks, NUM_THREADS >>> ( N, Ntally, tally_cell, active, space, E, tally_score, tally_count, done, cellnum);
	cudaThreadSynchronize();

}

