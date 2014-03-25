#include <cuda.h>
#include <stdio.h>
#include "datadef.h"
#include "wfloat3.h"
#include "binary_search.h"
#include "LCRNG.cuh"

inline __device__ void sample_therm(unsigned* rn, float* muout, float* vt, const float temp, const float E0, const float awr){

	// adapted from OpenMC's sample_target_velocity subroutine in src/physics.F90

	float k 	= 8.617332478e-11; //MeV/k
	float pi 	= 3.14159265359 ;
	float mu,c,beta_vn,beta_vt,beta_vt_sq,r1,r2,alpha,accept_prob;
	unsigned n;

	beta_vn = sqrtf(awr * 1.00866491600 * E0 / (temp*k) );
	alpha = 1.0/(1.0 + sqrtf(pi)*beta_vn/2.0);
	
	for(n=0;n<100;n++){
	
		r1 = get_rand(rn);
		r2 = get_rand(rn);
	
		if (get_rand(rn) < alpha) {
			beta_vt_sq = -logf(r1*r2);
		}
		else{
			c = cosf(pi/2.0 * get_rand(rn) );
			beta_vt_sq = -logf(r1) - logf(r2)*c*c;
		}
	
		beta_vt = sqrtf(beta_vt_sq);
	
		mu = 2.0*get_rand(rn) - 1.0;
	
		accept_prob = sqrtf(beta_vn*beta_vn + beta_vt_sq - 2*beta_vn*beta_vt*mu) / (beta_vn + beta_vt);
	
		if ( get_rand(rn) < accept_prob){break;}
	}

	vt[0] = sqrtf(beta_vt_sq*2.0*k*temp/(awr*1.00866491600));
	muout[0] = mu;
	//printf("%6.4E %6.4E\n",vt[0],mu);

}

__global__ void escatter_kernel(unsigned N, unsigned starting_index, unsigned* remap, unsigned* isonum, unsigned * index, unsigned * rn_bank, float * E, source_point * space, unsigned * rxn, float * awr_list, unsigned* done, float** scatterdat){


	int tid_in = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid_in >= N){return;}       //return if out of bounds
	
	//remap to active
	int tid=remap[starting_index+tid_in];
	unsigned this_rxn = rxn[starting_index+tid_in];
	//if(done[tid]){return;}
	if (this_rxn != 2){printf("escatter kernel accessing rxn!=2 (%u) @ dex %d\n",this_rxn,tid);return;}  //print and return if not elastic scatter

	//constants
	const float  pi           =   3.14159265359 ;
	const float  m_n          =   1.00866491600 ; // u
	//const float  kb			  =   8.617332478e-11; //MeV/k
	const float  temp         =   300;    // K
	const float  E_cutoff     =   1e-11;
	const float  E_max        =   20.0; //MeV
	// load history data
	unsigned 	this_tope 	= isonum[tid];
	unsigned 	this_dex	= index[tid];  //this is no longer the row, it is now the exact index of the flat cross section array, set by microscopic
	float 		this_E 		= E[tid];
	//float 		this_Q 		= 0.0;
	wfloat3 	hats_old(space[tid].xhat,space[tid].yhat,space[tid].zhat);
	float 		this_awr	= awr_list[this_tope];
	float * 	this_Sarray = scatterdat[this_dex];
	unsigned	rn 			= rn_bank[tid];

	// internal kernel variables
	float 		mu, phi, next_E, last_E;
    unsigned 	vlen, next_vlen, offset, k; 
    unsigned  	isdone = 0;
	float  		speed_n          	=   sqrtf(2.0*this_E/m_n);
	float 		E_new				=   0.0;
	wfloat3 	v_n_cm,v_t_cm,v_n_lf,v_t_lf,v_cm, hats_new, hats_target, rotation_hat;
	float 		mu0,mu1,cdf0,cdf1,speed_target,rn1;
	//float 		v_rel,E_rel;

	//make target isotropic
	phi = 2.0*pi*get_rand(&rn);
	mu  = (2.0*get_rand(&rn)) - 1.0;
	hats_target.x = sqrtf(1.0-(mu*mu))*cosf(phi);
	hats_target.y = sqrtf(1.0-(mu*mu))*sinf(phi); 
	hats_target.z = mu;

	//sample therm dist if low E
	//if(this_E <= 600*kb*temp ){
	sample_therm(&rn,&mu,&speed_target,temp,this_E,this_awr);
	hats_target = hats_old.rotate(mu, get_rand(&rn));
	
	// make speed vectors
	v_n_lf = hats_old    * speed_n;
	v_t_lf = hats_target * speed_target;

	// calculate  v_cm
	v_cm = (v_n_lf + (v_t_lf*this_awr))/(1.0+this_awr);

	//transform neutron velocity into CM frame
	v_n_cm = v_n_lf - v_cm;
	v_t_cm = v_t_lf - v_cm;	

	// sample new phi, mu_cm
	phi = 2.0*pi*get_rand(&rn);
	rn1 = get_rand(&rn);
	offset=4;
	if(this_Sarray == 0x0){
		mu= 2.0*rn1-1.0; 
		printf("null pointer in escatter!,dex %u tope %u E %6.4E\n",this_dex,this_tope,this_E);
	}
	else{  // 
		//printf("rxn=%u dex=%u %p %6.4E\n",rxn[tid],this_dex,this_array,this_E);
		memcpy(&last_E, 	&this_Sarray[0], sizeof(float));
		memcpy(&next_E, 	&this_Sarray[1], sizeof(float));
		memcpy(&vlen, 		&this_Sarray[2], sizeof(float));
		memcpy(&next_vlen, 	&this_Sarray[3], sizeof(float));
		float r = (this_E-last_E)/(next_E-last_E);
		//printf("(last,this,next) = %6.4E %6.4E %6.4E, prob=%6.4E, (this,next)_vlen= %u %u\n",last_E,this_E,next_E,(next_E-this_E)/(next_E-last_E),vlen,next_vlen);
		if(  get_rand(&rn) >= r ){   //sample last E
			k = binary_search(&this_Sarray[offset+vlen], rn1, vlen);
			cdf0 = this_Sarray[ (offset+vlen) +k  ];
			cdf1 = this_Sarray[ (offset+vlen) +k+1];
			mu0  = this_Sarray[ (offset)      +k  ];
			mu1  = this_Sarray[ (offset)      +k+1];
			mu   = (mu1-mu0)/(cdf1-cdf0)*(rn1-cdf0)+mu0; 
		}
		else{   // sample E+1
			k = binary_search(&this_Sarray[offset+2*vlen+next_vlen], rn1, next_vlen);
			cdf0 = this_Sarray[ (offset+2*vlen+next_vlen) +k  ];
			cdf1 = this_Sarray[ (offset+2*vlen+next_vlen) +k+1];
			mu0  = this_Sarray[ (offset+2*vlen)           +k  ];
			mu1  = this_Sarray[ (offset+2*vlen)           +k+1];
			mu   = (mu1-mu0)/(cdf1-cdf0)*(rn1-cdf0)+mu0; 
		}
	}

	// pre rotation directions
	hats_old = v_n_cm / v_n_cm.norm2();
	hats_old = hats_old.rotate(mu, get_rand(&rn));
	v_n_cm = hats_old * v_n_cm.norm2();

	// transform back to L
	v_n_lf = v_n_cm + v_cm;
	hats_new = v_n_lf / v_n_lf.norm2();
	hats_new = hats_new / hats_new.norm2();  // get higher precision, make SURE vector is length one
	
	// calculate energy
	E_new = 0.5 * m_n * v_n_lf.dot(v_n_lf);

	// enforce limits
	if ( E_new <= E_cutoff | E_new > E_max ){
		isdone=1;
	}

	// write results
	done[tid]       = isdone;
	E[tid]          = E_new;
	space[tid].xhat = hats_new.x;
	space[tid].yhat = hats_new.y;
	space[tid].zhat = hats_new.z;
	rn_bank[tid] 	= rn;


}

void escatter( cudaStream_t stream, unsigned NUM_THREADS, unsigned N, unsigned starting_index, unsigned* remap, unsigned* isonum, unsigned * index, unsigned * rn_bank, float * E, source_point * space ,unsigned * rxn, float* awr_list, unsigned* done, float** scatterdat){

	if(N<1){return;}
	unsigned blks = ( N + NUM_THREADS - 1 ) / NUM_THREADS;

	//escatter_kernel <<< blks, NUM_THREADS >>> (  N, RNUM_PER_THREAD, active, isonum, index, rn_bank, E, space, rxn, awr_list, done, scatterdat);
	escatter_kernel <<< blks, NUM_THREADS , 0 , stream >>> (  N, starting_index, remap, isonum, index, rn_bank, E, space, rxn, awr_list, done, scatterdat);
	cudaThreadSynchronize();

}

