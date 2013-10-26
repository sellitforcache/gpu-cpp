#include <cuda.h>
#include <stdio.h>
#include "datadef.h"
#include "wfloat3.h"


__global__ void escatter_kernel(unsigned N, unsigned RNUM_PER_THREAD, unsigned* isonum, unsigned * index, float * rn_bank, float * E, source_point * space, unsigned * rxn, float * awr_list, unsigned* done, float** scatterdat){


	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}       //return if out of bounds
	if (done[tid]){return;}
	if (rxn[tid] != 2){return;}  //return if not elastic scatter

	//printf("in escatter\n");

	//constants
	const float  pi           =   3.14159265359 ;
	const float  m_n          =   1.00866491600 ; // u
	const float  temp         =   0.025865214e-6;    // MeV
	const float  E_cutoff     =   1e-11;
	const float  E_max        =   20.0; //MeV
	// load history data
	unsigned 	this_tope 	= isonum[tid];
	unsigned 	this_dex	= index[tid];
	float 		this_E 		= E[tid];
	float 		this_Q 		= 0.0;
	wfloat3 	hats_old(space[tid].xhat,space[tid].yhat,space[tid].zhat);
	float 		this_awr	= awr_list[this_tope];
	float * 	this_array 	= scatterdat[this_dex];
	float 		rn1 		= rn_bank[ tid*RNUM_PER_THREAD + 3];
	float 		rn2 		= rn_bank[ tid*RNUM_PER_THREAD + 4];
	float 		rn3 		= rn_bank[ tid*RNUM_PER_THREAD + 5];
	float 		rn4 		= rn_bank[ tid*RNUM_PER_THREAD + 6];
	float 		rn5 		= rn_bank[ tid*RNUM_PER_THREAD + 7];
	float 		rn6 		= rn_bank[ tid*RNUM_PER_THREAD + 8];
	float 		rn7 		= rn_bank[ tid*RNUM_PER_THREAD + 9];
	float 		rn8 		= rn_bank[ tid*RNUM_PER_THREAD + 10];

	// internal kernel variables
	float 		mu, phi;
    unsigned 	vlen, offset; 
	float  		E_target     		=   temp * ( -logf(rn1) - logf(rn2)*cosf(pi/2*rn3)*cosf(pi/2*rn3) );
	float 		speed_target     	=   sqrtf(2.0*E_target/(this_awr*m_n));
	float  		speed_n          	=   sqrtf(2.0*this_E/m_n);
	float 		E_new				=   0.0;
	float 		a 					= 	this_awr/(this_awr+1.0);
	wfloat3 	v_n_cm,v_t_cm,v_n_lf,v_t_lf,v_cm, hats_new, hats_target;
	float 		mu0,mu1,cdf0,cdf1;
	//float 		v_rel,E_rel;

	// make target isotropic
	mu  = (2.0*rn4) - 1.0;
	phi = 2.0*pi*rn5;
	hats_target.x = sqrtf(1.0-(mu*mu))*cosf(phi);
	hats_target.y = sqrtf(1.0-(mu*mu))*sinf(phi); 
	hats_target.z = mu;
	
	// make speed vectors
	v_n_lf = hats_old    * speed_n;
	v_t_lf = hats_target * speed_target;

	// calculate  v_cm
	v_cm = (v_n_lf + (v_t_lf*this_awr))/(1.0+this_awr);

	//transform neutron velocity into CM frame
	v_n_cm = v_n_lf - v_cm;
	v_t_cm = v_t_lf - v_cm;

	// sample new phi, mu_cm
	phi = 2.0*pi*rn7;
	offset=1;
	if(this_array == 0x0){
		mu= 2*rn6-1; //MT=91 doesn't have angular tables for whatever reason
	}
	else{
		memcpy(&vlen, &this_array[0], sizeof(float));
		for(unsigned k=0;k<vlen;k++){
			if(rn6 <= this_array[1+vlen+(k+1)] ){  //look at CDF one ahead sicne first is 0
				//in this bin, linearly interpolate 
				mu0=this_array [offset+k+1];
				mu1=this_array [offset+k];
				cdf0=this_array[offset+vlen+k+1];
				cdf1=this_array[offset+vlen+k];
				mu = (mu1-mu0)/(cdf1-cdf0)*(rn6-cdf0)+mu0;
				break;
			}
		}
	}

	// transform hats to CM, sample phi and rotate
	hats_old = v_n_cm / v_n_cm.norm2();
	
	hats_new = hats_old.rotate(phi,mu);
	//hats_new.x = sqrtf(1.0-mu*mu)*cosf(phi);
	//hats_new.y = sqrtf(1.0-mu*mu)*sinf(phi); 
	//hats_new.z = mu;

	//calculate final velocity in CM
	v_n_cm = hats_new * sqrtf( v_n_cm.dot(v_n_cm) + 2.0 * a * this_Q / m_n );

	// transform back to L
	v_n_lf = v_n_cm + v_cm;
	hats_new = v_n_lf / v_n_lf.norm2();

	// calculate energy
	E_new = 0.5 * m_n * v_n_lf.dot(v_n_lf);

	// check cutoff
	if (E_new <= E_cutoff){
		//E_new = 1.5*E_cutoff;
		done[tid]=1;
		//printf("enforcing E_min in escatter");
	}
	if (E_new > E_max){
		//E_new = 0.9*E_max;
		done[tid]=1;
		//printf("enforcing E_max in escatter");
	}

	//printf("speed target = %6.4E, speed=%6.4E, Eold,Enew = %10.8E %10.8E\n",speed_target, speed_n,this_E,E_new);
	// write results
	E[tid]          = E_new;
	space[tid].xhat = hats_new.x;
	space[tid].yhat = hats_new.y;
	space[tid].zhat = hats_new.z;


}

void escatter(unsigned blks, unsigned NUM_THREADS, unsigned N, unsigned RNUM_PER_THREAD, unsigned* isonum, unsigned * index, float * rn_bank, float * E, source_point * space ,unsigned * rxn, float* awr_list, unsigned* done, float** scatterdat){

	escatter_kernel <<< blks, NUM_THREADS >>> (  N, RNUM_PER_THREAD, isonum, index, rn_bank, E, space, rxn, awr_list, done, scatterdat);
	cudaThreadSynchronize();

}

