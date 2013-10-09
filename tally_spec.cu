#include <cuda.h>
#include <stdio.h>
#include "datadef.h"

__global__ void tally_spec_kernel(unsigned N, unsigned Ntally, source_point* space, float* E, float * tally_score, unsigned * tally_count){

	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}

	int k;
	float 		my_E   			= E[tid];
	float 		samp_dist		= space[tid].samp_dist;
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
		for(k=1;k<Ntally-1;k++){
			next_bin=multiplier*this_bin;
			if(my_E>this_bin & my_E<=next_bin){
				my_bin_index=k;
				break;
			}
			this_bin=next_bin;
		}
	}

	//printf("my_bin_index=%u\n",my_bin_index);

	//score the bins atomically, could be bad if many neutrons are in a single bin since this will serialize their operations
	atomicAdd(&tally_score[my_bin_index], 1.0/samp_dist);
	atomicInc(&tally_count[my_bin_index], 4294967295);


}

void tally_spec(unsigned blks, unsigned NUM_THREADS,  unsigned N, unsigned Ntally, source_point * space, float* E, float * tally_score, unsigned * tally_count){

	printf("Ntally = %u\n",Ntally);
	tally_spec_kernel <<< blks, NUM_THREADS >>> ( N, Ntally, space, E, tally_score, tally_count);
	cudaThreadSynchronize();

}

