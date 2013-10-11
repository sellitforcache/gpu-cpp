#include <cuda.h>
#include <stdio.h>
#include "datadef.h"

__global__ void tally_spec_kernel(unsigned N, unsigned Ntally, source_point* space, float* E, float * tally_score, unsigned * tally_count, unsigned* done){

	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}
	if (done[tid]){return;}

	int k;
	float 		my_E   			= E[tid];
	float 		macro_t 		= space[tid].macro_t;
	unsigned 	my_bin_index 	= 0;

	const float Emax 	= 20.00000;
	const float Emin 	=  1.0e-11;
	float log_spacing 	= (log10f(Emax)-(-11.0))/(Ntally-2+1);//(log10f(Emax)-log10f(Emin))/(Ntally-2+1);
	float multiplier  	= powf(10,log_spacing);
	float this_bin,next_bin;

	//printf("%u %10.8E %12.10E %10.8E \n",Ntally,log_spacing,multiplier,my_E);

	// determine bin number
	if     ( my_E <= Emin ){ my_bin_index = 0;        }
	else if( my_E >  Emax ){ my_bin_index = Ntally-1; }
	else{
		this_bin=Emin;
		for(k=0;k<Ntally;k++){
			next_bin=multiplier*this_bin;
			if(my_E>this_bin & my_E<=next_bin){
				my_bin_index=k;
				break;
			}
			this_bin=next_bin;
		}
	}

	//printf("my_bin_index=%u: score there = %10.8E, count there = %u \n",my_bin_index,tally_score[my_bin_index],tally_count[my_bin_index]);

	//score the bins atomically, could be bad if many neutrons are in a single bin since this will serialize their operations
	atomicAdd(&tally_score[my_bin_index], macro_t);
	atomicInc(&tally_count[my_bin_index], 4294967295);


}

void tally_spec(unsigned blks, unsigned NUM_THREADS,  unsigned N, unsigned Ntally, source_point * space, float* E, float * tally_score, unsigned * tally_count, unsigned* done){

	tally_spec_kernel <<< blks, NUM_THREADS >>> ( N, Ntally, space, E, tally_score, tally_count, done);
	cudaThreadSynchronize();

}

