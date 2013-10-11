#include <cuda.h>
#include <stdio.h>
#include "datadef.h"
#include "wfloat3.h"

__global__ void iscatter_kernel(unsigned N, unsigned RNUM_PER_THREAD, unsigned* isonum, unsigned * index, float * rn_bank, float * E, source_point * space, unsigned * rxn, float * awr_list, float * Q, unsigned * done){


	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}       //return if out of bounds
	if (done[tid]){return;}
	if (rxn[tid] < 50 | rxn[tid] > 91 ){return;}  //return if not inelastic scatter

	//constants
	const float  pi           =   3.14159265359 ;
	const float  m_n          =   1.00866491600 ; // u
	const float  temp         =   0.025865214e-6;    // MeV
	const float  E_cutoff     =   1e-11;
	const float  E_max        =   20.0; //MeV
	// load history data
	unsigned 	this_tope 	= isonum[tid];
	float 		this_E 		= E[tid];
	float 		this_Q 		= Q[tid];
	wfloat3 	hats_old(space[tid].xhat,space[tid].yhat,space[tid].zhat);
	float 		this_awr	= awr_list[this_tope];
	float 		rn1 		= rn_bank[ tid*RNUM_PER_THREAD + 3];
	float 		rn2 		= rn_bank[ tid*RNUM_PER_THREAD + 4];
	float 		rn3 		= rn_bank[ tid*RNUM_PER_THREAD + 5];
	float 		rn4 		= rn_bank[ tid*RNUM_PER_THREAD + 6];
	float 		rn5 		= rn_bank[ tid*RNUM_PER_THREAD + 7];
	float 		rn6 		= rn_bank[ tid*RNUM_PER_THREAD + 8];
	float 		rn7 		= rn_bank[ tid*RNUM_PER_THREAD + 9];

	// internal kernel variables
	float  		E_target     		=   temp * ( -logf(rn1) - logf(rn2)*cosf(pi/2*rn3)*cosf(pi/2*rn3) );
	float 		speed_target     	=   sqrtf(2.0*E_target/(this_awr*m_n));
	float  		speed_n          	=   sqrtf(2.0*this_E/m_n);
	float 		E_new				=   0.0;
	float 		alpha 				= 	this_awr/(1.0+this_awr);
	wfloat3 	v_n_cm,v_t_cm,v_n_lf,v_t_lf,v_cm;
    //float 		v_rel,E_rel;

	// make target isotropic 
	float mu  = 2.0* rn4 - 1.0;
	float phi = 2.0* pi * rn5;
	wfloat3 hats_target(sqrtf(1.0-mu*mu)*cosf(phi),sqrtf(1.0-mu*mu)*sinf(phi), mu);

	// make scatter isotropic - change to tables later
	mu  = 2.0*rn6-1.0;
	phi = 2*pi*rn7;
	wfloat3 hats_new(sqrtf(1.0-mu*mu)*cosf(phi),sqrtf(1.0-mu*mu)*sinf(phi), mu);

	// make speed vectors
	v_n_lf = hats_old * speed_n;
	v_t_lf = hats_target * speed_target;

	// calculate  v_cm
	v_cm = v_n_lf/(1.0+this_awr) + v_t_lf*alpha ;

	//transform into CM frame
	v_n_cm = v_n_lf - v_cm;
	v_t_cm = v_t_lf - v_cm;

	// calculate final rotated neutron velocity
	v_n_cm = hats_new * sqrtf( v_n_cm.dot(v_n_cm) + (2.0* this_Q * alpha/(m_n)) );

	// transform back to LF
	v_n_lf = v_n_cm + v_cm;
	hats_new = v_n_lf / v_n_lf.norm2();

	// calculate new energy
	E_new = 0.5 * m_n * v_n_lf.dot(v_n_lf);

	// check cutoff
	if (E_new < E_cutoff){
		E_new = E_cutoff;
		//printf("enforcing E_min in iscatter");
	}
	if (E_new > E_max){
		E_new = 0.9*E_max;
		//printf("enforcing E_max in iscatter");
	}

	//printf("speed target = %6.4E, speed=%6.4E, Eold,Enew = %10.8E %10.8E\n",speed_target, speed_n,this_E,E_new);
	// write results
	E[tid]          = E_new;
	space[tid].xhat = hats_new.x;
	space[tid].yhat = hats_new.y;
	space[tid].zhat = hats_new.z;

}

void iscatter(unsigned blks, unsigned NUM_THREADS, unsigned N, unsigned RNUM_PER_THREAD, unsigned* isonum, unsigned * index, float * rn_bank, float * E, source_point * space ,unsigned * rxn, float* awr_list, float * Q, unsigned* done){

	iscatter_kernel <<< blks, NUM_THREADS >>> (  N, RNUM_PER_THREAD, isonum, index, rn_bank, E, space, rxn, awr_list, Q, done);
	cudaThreadSynchronize();

}

