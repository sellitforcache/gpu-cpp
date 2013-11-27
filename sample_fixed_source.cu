#include <cuda.h>
#include <stdio.h>
#include "datadef.h"

__global__ void sample_fixed_source_kernel(unsigned N, unsigned RNUM_PER_THREAD, unsigned* active, float* rn_bank , float * E, source_point* space){

	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}

	//remap to active
	//tid=active[tid];

	// load in
	const float rn1   =  rn_bank[ (tid * RNUM_PER_THREAD) + 4];
	const float rn2   =  rn_bank[ (tid * RNUM_PER_THREAD) + 5];
	const float pi    =   3.14159265359 ;
	const float mu    = ( rn1 ) * 2.0 - 1.0;
	const float theta = ( rn2 ) * 2.0 * pi ;

	//monoenergetic for now
	E[tid]=1.0e-6;

	//point source for now
	space[tid].x = 0.0;
	space[tid].y = 0.0;
	space[tid].z = 0.0;

	//set isotropic for now
	space[tid].xhat = sqrtf(1-mu*mu) * cosf( theta );
	space[tid].yhat = sqrtf(1-mu*mu) * sinf( theta );
    space[tid].zhat = mu;


}

void sample_fixed_source( unsigned NUM_THREADS, unsigned N, unsigned RNUM_PER_THREAD, unsigned* active, float * rn_bank, float * E, source_point* space){

	unsigned blks = ( N + NUM_THREADS - 1 ) / NUM_THREADS;

	sample_fixed_source_kernel <<< blks, NUM_THREADS >>> (  N, RNUM_PER_THREAD, active, rn_bank, E , space );
	cudaThreadSynchronize();

}

