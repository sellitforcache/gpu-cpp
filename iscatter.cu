#include <cuda.h>
#include <stdio.h>
#include "datadef.h"

__global__ void iscatter_kernel(unsigned N, unsigned RNUM_PER_THREAD, unsigned* isonum, unsigned * index, float * rn_bank, float * E, source_point * space, unsigned * rxn, float * awr_list, float * Q, unsigned * done){


	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}       //return if out of bounds
	if (done[tid]){return;}
	if (rxn[tid] < 50 | rxn[tid] > 91 ){return;}  //return if not inelastic scatter

	//constants
	const float  pi           =   3.14159265359 ;
	const float  m_n          =   1.00866491600 ; // u
	const float  temp         =   0.025e-6;    // MeV
	//const float  E_cutoff     =   1e-11;
	//const float  E_max        =   20.0; //MeV
	// load history data
	unsigned 	this_tope 	= isonum[tid];
	float 		this_E 		= E[tid];
	float 		this_Q 		= Q[tid];
	float 		m_tope		= awr_list[this_tope];
	float 		rn1 		= rn_bank[ tid*RNUM_PER_THREAD + 3];
	float 		rn2 		= rn_bank[ tid*RNUM_PER_THREAD + 4];
	float 		rn3 		= rn_bank[ tid*RNUM_PER_THREAD + 5];
	float 		rn4 		= rn_bank[ tid*RNUM_PER_THREAD + 6];
	float 		rn5 		= rn_bank[ tid*RNUM_PER_THREAD + 7];

	// internal kernel variables

	// istropic section
	float mu  = 2.0*rn1-1.0;
	float phi = 2*pi*rn2;
	float E_target     =   temp * ( -log10f(rn3) - log10f(rn4)*cosf(pi/2*rn5)*cosf(pi/2*rn5) );
	float alpha = (m_tope-m_n)/(m_tope+m_n);
		  alpha = alpha * alpha;
	float xhat_new = sqrtf(1.0-mu*mu)*cosf(phi);
	float yhat_new = sqrtf(1.0-mu*mu)*sinf(phi);
	float zhat_new = mu;
	float E_new = this_E  * ( (1+alpha) + (1-alpha)*mu ) / 2.0 + this_Q ;
	//printf("Q=%6.3E\n",this_Q);

	// approximate thermal
	if (E_new < 1.5*E_target){E_new=E_target;}

	// write output
	E[tid] = E_new;
	space[tid].xhat = xhat_new;
	space[tid].yhat = yhat_new;
	space[tid].zhat = zhat_new;

}

void iscatter(unsigned blks, unsigned NUM_THREADS, unsigned N, unsigned RNUM_PER_THREAD, unsigned* isonum, unsigned * index, float * rn_bank, float * E, source_point * space ,unsigned * rxn, float* awr_list, float * Q, unsigned* done){

	iscatter_kernel <<< blks, NUM_THREADS >>> (  N, RNUM_PER_THREAD, isonum, index, rn_bank, E, space, rxn, awr_list, Q, done);
	cudaThreadSynchronize();

}

