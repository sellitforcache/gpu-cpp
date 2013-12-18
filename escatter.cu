#include <cuda.h>
#include <stdio.h>
#include "datadef.h"
#include "wfloat3.h"
#include "binary_search.h"

__global__ void escatter_kernel(unsigned N, unsigned RNUM_PER_THREAD, unsigned* active, unsigned* isonum, unsigned * index, float * rn_bank, float * E, source_point * space, unsigned * rxn, float * awr_list, unsigned* done, float** scatterdat){


	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}       //return if out of bounds
	
	//remap to active
	//tid=active[tid];
	if(done[tid]){return;}

	if (rxn[tid] != 2){return;}  //return if not elastic scatter

	//printf("in escatter - tid %u Nrun %u \n",tid,N);

	//constants
	const float  pi           =   3.14159265359 ;
	const float  m_n          =   1.00866491600 ; // u
	const float  temp         =   0.02585202857e-6;    // MeV
	const float  E_cutoff     =   1e-11;
	const float  E_max        =   20.0; //MeV
	// load history data
	unsigned 	this_tope 	= isonum[tid];
	unsigned 	this_dex	= index[tid];  //this is no longer the row, it is now the exact index of the array, set by microscopic
	float 		this_E 		= E[tid];
	//float 		this_Q 		= 0.0;
	wfloat3 	hats_old(space[tid].xhat,space[tid].yhat,space[tid].zhat);
	float 		this_awr	= awr_list[this_tope];
	float * 	this_Sarray = scatterdat[this_dex];
	float 		rn1 		= rn_bank[ tid*RNUM_PER_THREAD + 3];
	float 		rn2 		= rn_bank[ tid*RNUM_PER_THREAD + 4];
	float 		rn3 		= rn_bank[ tid*RNUM_PER_THREAD + 5];
	float 		rn4 		= rn_bank[ tid*RNUM_PER_THREAD + 6];
	float 		rn5 		= rn_bank[ tid*RNUM_PER_THREAD + 7];
	float 		rn6 		= rn_bank[ tid*RNUM_PER_THREAD + 8];
	float 		rn7 		= rn_bank[ tid*RNUM_PER_THREAD + 9];
	float 		rn8 		= rn_bank[ tid*RNUM_PER_THREAD + 10];
	//float 		rn9 		= rn_bank[ tid*RNUM_PER_THREAD + 11];

	// internal kernel variables
	float 		mu, phi, next_E, last_E;
    unsigned 	vlen, next_vlen, offset, k; 
    unsigned  	isdone = 0;
	float  		E_target     		=   1.2 * temp * ( -logf(rn1) - logf(rn2)*cosf(pi/2*rn3)*cosf(pi/2*rn3) );
	float 		speed_target     	=   sqrtf(2.0*E_target/(this_awr*m_n));
	float  		speed_n          	=   sqrtf(2.0*this_E/m_n);
	float 		E_new				=   0.0;
	//float 		a 					= 	this_awr/(this_awr+1.0);
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
	offset=4;
	if(this_Sarray == 0x0){
		mu= 2*rn6-1; 
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
		if(  rn8 >= r ){   //sample last E
			k = binary_search(&this_Sarray[offset+vlen], rn6, vlen);
			cdf0 = this_Sarray[ (offset+vlen) +k  ];
			cdf1 = this_Sarray[ (offset+vlen) +k+1];
			mu0  = this_Sarray[ (offset)      +k  ];
			mu1  = this_Sarray[ (offset)      +k+1];
			mu   = (mu1-mu0)/(cdf1-cdf0)*(rn6-cdf0)+mu0; 
		}
		else{   // sample E+1
			k = binary_search(&this_Sarray[offset+2*vlen+next_vlen], rn6, next_vlen);
			cdf0 = this_Sarray[ (offset+2*vlen+next_vlen) +k  ];
			cdf1 = this_Sarray[ (offset+2*vlen+next_vlen) +k+1];
			mu0  = this_Sarray[ (offset+2*vlen)           +k  ];
			mu1  = this_Sarray[ (offset+2*vlen)           +k+1];
			mu   = (mu1-mu0)/(cdf1-cdf0)*(rn6-cdf0)+mu0; 
		}
	}


	// pre rotation directions
	hats_old = v_n_cm / v_n_cm.norm2();
	//  create a perpendicular roation vector 
	//wfloat3 rotation_hat( 0.0, 0.0, 1.0 );
	wfloat3 rotation_hat = hats_target.cross( v_n_cm );
	rotation_hat = rotation_hat / rotation_hat.norm2();
	//  do rotations, polar first, then azimuthal
	v_n_cm.rodrigues_rotation( rotation_hat, acosf(mu) );
	v_n_cm.rodrigues_rotation( hats_old,     phi       );
	// transform back to L
	v_n_lf = v_n_cm + v_cm;
	hats_new = v_n_lf / v_n_lf.norm2();
	// calculate energy
	E_new = 0.5 * m_n * v_n_lf.dot(v_n_lf);

	// enforce limits
	if ( E_new <= E_cutoff | E_new > E_max ){
		isdone=1;
	}

	//printf("speed target = %6.4E, speed=%6.4E, Eold,Enew = %10.8E %10.8E\n",speed_target, speed_n,this_E,E_new);
	// write results
	done[tid]       = isdone;
	E[tid]          = E_new;
	space[tid].xhat = hats_new.x;
	space[tid].yhat = hats_new.y;
	space[tid].zhat = hats_new.z;


}

void escatter( cudaStream_t stream, unsigned NUM_THREADS, unsigned N, unsigned RNUM_PER_THREAD, unsigned* active, unsigned* isonum, unsigned * index, float * rn_bank, float * E, source_point * space ,unsigned * rxn, float* awr_list, unsigned* done, float** scatterdat){

	unsigned blks = ( N + NUM_THREADS - 1 ) / NUM_THREADS;

	escatter_kernel <<< blks, NUM_THREADS , 0 , stream >>> (  N, RNUM_PER_THREAD, active, isonum, index, rn_bank, E, space, rxn, awr_list, done, scatterdat);
	cudaThreadSynchronize();

}

