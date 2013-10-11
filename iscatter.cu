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
	const float  temp         =   0.026e-6;    // MeV
	const float  E_cutoff     =   1e-11;
	const float  E_max        =   20.0; //MeV
	// load history data
	unsigned 	this_tope 	= isonum[tid];
	float 		this_E 		= E[tid];
	float 		this_Q 		= Q[tid];
	wfloat3 	hats_old(space[tid].xhat,space[tid].yhat,space[tid].zhat);
	float 		m_tope		= awr_list[this_tope];
	float 		rn1 		= rn_bank[ tid*RNUM_PER_THREAD + 3];
	float 		rn2 		= rn_bank[ tid*RNUM_PER_THREAD + 4];
	float 		rn3 		= rn_bank[ tid*RNUM_PER_THREAD + 5];
	float 		rn4 		= rn_bank[ tid*RNUM_PER_THREAD + 6];
	float 		rn5 		= rn_bank[ tid*RNUM_PER_THREAD + 7];
	float 		rn6 		= rn_bank[ tid*RNUM_PER_THREAD + 8];
	float 		rn7 		= rn_bank[ tid*RNUM_PER_THREAD + 9];

	// internal kernel variables
	float  		E_target     		=   temp * ( -logf(rn1) - logf(rn2)*cosf(pi/2*rn3)*cosf(pi/2*rn3) );
	float 		speed_target     	=   sqrtf(2.0*E_target/m_tope);
	float  		speed_n          	=   sqrtf(2.0*this_E/m_n);
	float 		E_new				=   0.0;
	wfloat3 	v_target,v_n,v_cm,v_n_cm,v_target_cm,n_hat,v_n_cm_r,v_n_lf,v_diff;
    float 		v_rel,E_rel;

	// make target isotropic 
	float mu  = 2.0*rn4-1.0;
	float phi = 2*pi*rn5;
	wfloat3 target_hats(sqrtf(1.0-mu*mu)*cosf(phi),sqrtf(1.0-mu*mu)*sinf(phi), mu);

	// make scatter isotropic - change to tables later
	mu  = 2.0*rn6-1.0;
	phi = 2*pi*rn7;

	//set quantities
	v_n      = hats_old * speed_n;
	v_target = target_hats * speed_target;

	//find v_cm
	v_cm = ( v_n*m_n + v_target*m_tope) * 1.0/(m_n+m_tope);

	//transform velocities to CM
	v_n_cm      = v_cm - v_n;
	v_target_cm = v_cm - v_target;

	//calculate relative quantities
	v_diff = v_n_cm - v_target_cm ;
	v_rel  = v_diff.norm2();
	E_rel  = 0.5*m_n*v_rel*v_rel;

	// subtract out Q value from neutron velocity
	this_Q=0;
	float A = m_tope;
	float E_rel_new = (A/(A+1)) * (A/(A+1)) * ( E_rel - ((A+1)/A)*this_Q );
	float v_rel_new = sqrtf(2.0 * E_rel_new / m_n);
	v_n_cm = v_n_cm*v_rel_new/v_n_cm.norm2();

	// find rotation axis by doing v_n cross v_t and normalizing
	wfloat3 n_hat_cp = v_n_cm.cross(v_target_cm);
	n_hat = n_hat_cp*1.0/n_hat_cp.norm2();

	// do polar rotation in CM acording to rodrigues' rotation formula
	v_n_cm_r.rodrigues_rotation( n_hat , acos(mu) );

	// make azimuth isotropic, use original vector as rotation vector 
	// needed to rotate it out in the previous step for this step to make a difference
	// can use n_hat variables since they are not used anymore
	n_hat = v_n_cm*1.0/v_n_cm.norm2();
	v_n_cm_r.rodrigues_rotation( n_hat , phi );

	// transform back to LF
	v_n_lf = v_cm - v_n_cm_r;

	// convert to angle and energy
	//E_new    = f3_norm2(v_n_lf);
	float v_norm = v_n_lf.norm2();
	wfloat3 hats_new = v_n_lf / v_norm;
	E_new    = 0.5*m_n*v_norm*v_norm;

	// check cutoff
	if (E_new < E_cutoff){
		E_new = E_cutoff;
		//printf("enforcing E_min in escatter");
	}
	if (E_new > E_max){
		E_new = 0.9*E_max;
		//printf("enforcing E_max in escatter");
	}

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

