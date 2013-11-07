#include <cuda.h>
#include <stdio.h>
#include "datadef.h"

__global__ void pop_secondaries_kernel(unsigned N, unsigned RNUM_PER_THREAD, unsigned* completed, unsigned* scanned, unsigned* yield, unsigned* done, unsigned* index, source_point* space, float* E , float* rn_bank, float**  energydata){

	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}
	if (!yield[tid]){return;}

	unsigned 	position = scanned[tid];
	unsigned 	this_yield = yield[tid];
	unsigned 	dex = index[tid];
	unsigned 	data_dex, k, offset, vlen, law;
	float * 	this_array = energydata[dex];
	float 		sampled_E, phi, mu, rn1, rn2;
	float 		cdf0, cdf1, e0, e1;
	const float  pi           =   3.14159265359 ;
	offset = 2;


	// adjust index for nonzero
	if(position){position=position-1;}	

	// set data
	for(k=0 ; k < this_yield ; k++ ){
		//sample energy dist
		rn1 = rn_bank[ tid*RNUM_PER_THREAD + 11 + k];
		memcpy(&vlen, 		&this_array[0], sizeof(float));
		memcpy(&law, 		&this_array[1], sizeof(float));
		for ( k=0 ; k<vlen-1 ; k++ ){
			if( rn1 >= this_array[ (offset+vlen) +k] & rn1 < this_array[ (offset+vlen) +k+1] ){
				cdf0 = this_array[ (offset+vlen) +k  ];
				cdf1 = this_array[ (offset+vlen) +k+1];
				e0  = this_array[ (offset)      +k  ];
				e1  = this_array[ (offset)      +k+1];
				sampled_E   = (e1-e0)/(cdf1-cdf0)*(rn1-cdf0)+e0; 
				break;
			}
		}
		printf("%6.4E\n",sampled_E);

		//sample isotropic directions
		rn1 = rn_bank[ tid*RNUM_PER_THREAD + 12 + k];
		rn2 = rn_bank[ tid*RNUM_PER_THREAD + 13 + k];
		mu  = 2.0*rn1-1.0; 
		phi = 2.0*pi*rn2;

		// set data
		E    [k+position] 		= sampled_E;
		space[k+position].x 	= space[tid].x;
		space[k+position].y 	= space[tid].y;
		space[k+position].z 	= space[tid].z;
		space[k+position].xhat 	= sqrtf(1.0-(mu*mu))*cosf(phi);
		space[k+position].yhat 	= sqrtf(1.0-(mu*mu))*sinf(phi); 
		space[k+position].zhat 	= mu;
		done [k+position] 		= 0;

	}


}

void pop_secondaries(unsigned blks, unsigned NUM_THREADS,  unsigned N, unsigned RNUM_PER_THREAD, unsigned* d_completed, unsigned* d_scanned, unsigned* d_yield, unsigned* d_done, unsigned* d_index, source_point* d_space, float* d_E , float* d_rn_bank, float ** energydata){

	pop_secondaries_kernel <<< blks, NUM_THREADS >>> ( N, RNUM_PER_THREAD, d_completed, d_scanned, d_yield, d_done, d_index, d_space, d_E , d_rn_bank, energydata);
	cudaThreadSynchronize();

}

