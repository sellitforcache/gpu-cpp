#include <cuda.h>
#include <stdio.h>
#include "datadef.h"

__global__ void sample_isotropic_directions_kernel(unsigned N , unsigned RNUM_PER_THREAD, unsigned* active, source_point * positions_ptr , float * rn_bank ){

	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid>=N){return;}

	//tid=active[tid];

	const float rn1   =  rn_bank[ (tid * RNUM_PER_THREAD) + 4];
	const float rn2   =  rn_bank[ (tid * RNUM_PER_THREAD) + 5];
	const float pi    =   3.14159265359 ;
	const float mu    = ( rn1 ) * 2.0 - 1.0;
	const float theta = ( rn2 ) * 2.0 * pi ;

	positions_ptr[tid].samp_dist =	500;    
	positions_ptr[tid].xhat      =	sqrtf(1-mu*mu) * cosf( theta );
	positions_ptr[tid].yhat      =	sqrtf(1-mu*mu) * sinf( theta );
	positions_ptr[tid].zhat      =	      mu;
  
}

void sample_isotropic_directions( unsigned NUM_THREADS, unsigned N, unsigned RNUM_PER_THREAD, unsigned* d_active, source_point * d_space , float * d_rn_bank){

	unsigned blks = ( N + NUM_THREADS - 1 ) / NUM_THREADS;

	sample_isotropic_directions_kernel <<<  blks, NUM_THREADS >>> ( N , RNUM_PER_THREAD, d_active, d_space , d_rn_bank );
	cudaThreadSynchronize();

}