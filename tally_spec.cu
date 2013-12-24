#include <cuda.h>
#include <stdio.h>
#include "datadef.h"

__global__ void tally_spec_kernel(unsigned N, unsigned Ntally, unsigned* active, source_point* space, float* E, float * tally_score, unsigned * tally_count, unsigned* done, unsigned* mask){

	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}
	//tid=active[tid];
	if (done[tid]){return;}
	//if (!mask[tid]){return;}

	int k;
	float 		my_E   			= E[tid];
	float 		macro_t 		= space[tid].macro_t;
	unsigned 	my_bin_index 	= 0;

	const float Emax 	= 20.00000;
	const float Emin 	=  1.0e-11;
	float log_spacing 	= (log10f(Emax)-(-11.0))/(Ntally-2+1);//(log10f(Emax)-log10f(Emin))/(Ntally-2+1);
	float multiplier  	= powf(10,log_spacing);
	float this_bin,next_bin;

	// determine bin number
	my_bin_index = logf(my_E/Emin)/logf(multiplier);
//	this_bin=Emin;
//	for(k=0;k<Ntally;k++){
//		next_bin=multiplier*this_bin;
//		if(my_E>this_bin & my_E<=next_bin){
//			my_bin_index=k;
//			break;
//		}
//		this_bin=next_bin;
//	}

	//printf("macro_t=%6.4E my_bin_index=%u: score there = %10.8E, count there = %u \n",macro_t,my_bin_index,tally_score[my_bin_index],tally_count[my_bin_index]);

	//score the bins atomicly, could be bad if many neutrons are in a single bin since this will serialize their operations
	atomicAdd(&tally_score[my_bin_index], 1.0/macro_t);
	atomicInc(&tally_count[my_bin_index], 4294967295);

	//printf("%6.4E\n",macro_t);


}

void tally_spec(unsigned NUM_THREADS,  unsigned N, unsigned Ntally, unsigned* active, source_point * space, float* E, float * tally_score, unsigned * tally_count, unsigned* done, unsigned* mask){
	
	unsigned blks = ( N + NUM_THREADS - 1 ) / NUM_THREADS;

	tally_spec_kernel <<< blks, NUM_THREADS >>> ( N, Ntally, active, space, E, tally_score, tally_count, done, mask);
	cudaThreadSynchronize();

}

