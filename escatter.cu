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

//inline __device__ wfloat3 rotate_angle(unsigned* rn, wfloat3 uvw0, float mu){
//
//    wfloat3 uvw;  //rotated directional cosine
//
//    float phi    ;// azimuthal angle
//    float sinphi ;// sine of azimuthal angle
//    float cosphi ;// cosine of azimuthal angle
//    float a      ;// sqrt(1 - mu^2)
//    float b      ;// sqrt(1 - w^2)
//    float u0     ;// original cosine in x direction
//    float v0     ;// original cosine in y direction
//    float w0     ;// original cosine in z direction
//    float pi 	= 3.14159;
//
//    // Copy original directional cosines
//    u0 = uvw0.x;
//    v0 = uvw0.y;
//    w0 = uvw0.z;
//
//    // Sample azimuthal angle in [0,2pi)
//    phi = 2.0 * pi * get_rand(rn);
//
//    // Precompute factors to save flops
//    sinphi = sinf(phi);
//    cosphi = cosf(phi);
//    a = sqrtf(max(0.0, 1.0 - mu*mu));
//    b = sqrtf(max(0.0, 1.0 - w0*w0));
//
//    // Need to treat special case where sqrt(1 - w**2) is close to zero by
//    // expanding about the v component rather than the w component
//    if (b > 1e-10) {
//      uvw.x = mu*u0 + a*(u0*w0*cosphi - v0*sinphi)/b;
//      uvw.y = mu*v0 + a*(v0*w0*cosphi + u0*sinphi)/b;
//      uvw.z = mu*w0 - a*b*cosphi;
//  	}
//    else{
//      b = sqrtf(1.0 - v0*v0);
//      uvw.x = mu*u0 + a*(u0*v0*cosphi + w0*sinphi)/b;
//      uvw.y = mu*v0 - a*b*cosphi;
//      uvw.z = mu*w0 + a*(v0*w0*cosphi - u0*sinphi)/b;
//    }
//
//    return uvw;
//
// }

__global__ void escatter_kernel(unsigned N, unsigned RNUM_PER_THREAD, unsigned* active, unsigned* isonum, unsigned * index, unsigned * rn_bank, float * E, source_point * space, unsigned * rxn, float * awr_list, unsigned* done, float** scatterdat){


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
	//const float  kb			  =   8.617332478e-11; //MeV/k
	const float  temp         =   300;    // K
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
		//hats_target = rotate_angle(&rn,hats_old,mu);
		rotation_hat = hats_old.cross( hats_target );
		rotation_hat = rotation_hat / rotation_hat.norm2();
		hats_target = hats_old;
		hats_target.rodrigues_rotation( rotation_hat, acosf(mu) );
		hats_target.rodrigues_rotation( hats_old,     phi       );
	//}
	//else{
	//	speed_target = 0.0;
	//}
	//__syncthreads();
	
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
	//printf("k %u vlen %u\n",k,vlen);


	// pre rotation directions
	hats_old = v_n_cm / v_n_cm.norm2();
	//  create a perpendicular roation vector 
	rotation_hat = hats_target.cross( v_n_cm );
	rotation_hat = rotation_hat / rotation_hat.norm2();
	//  do rotations, polar first, then azimuthal
	v_n_cm.rodrigues_rotation( rotation_hat, acosf(mu) );
	v_n_cm.rodrigues_rotation( hats_old,     phi       );
	//v_n_cm = rotate_angle(&rn,hats_old,mu) * v_n_cm.norm2();

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

	//printf("%u esatter hat length % 10.8E\n",tid,sqrtf(hats_new.x*hats_new.x+hats_new.y*hats_new.y+hats_new.z*hats_new.z));


	//printf("speed target = %6.4E, speed=%6.4E, Eold,Enew = %10.8E %10.8E\n",speed_target, speed_n,this_E,E_new);
	// write results
	done[tid]       = isdone;
	E[tid]          = E_new;
	space[tid].xhat = hats_new.x;
	space[tid].yhat = hats_new.y;
	space[tid].zhat = hats_new.z;
	rn_bank[tid] 	= rn;

	//printf("% 6.4E % 6.4E % 6.4E\n",hats_new.x,hats_new.y,hats_new.z);


}

void escatter( cudaStream_t stream, unsigned NUM_THREADS, unsigned N, unsigned RNUM_PER_THREAD, unsigned* active, unsigned* isonum, unsigned * index, unsigned * rn_bank, float * E, source_point * space ,unsigned * rxn, float* awr_list, unsigned* done, float** scatterdat){

	unsigned blks = ( N + NUM_THREADS - 1 ) / NUM_THREADS;

	//escatter_kernel <<< blks, NUM_THREADS >>> (  N, RNUM_PER_THREAD, active, isonum, index, rn_bank, E, space, rxn, awr_list, done, scatterdat);
	escatter_kernel <<< blks, NUM_THREADS , 0 , stream >>> (  N, RNUM_PER_THREAD, active, isonum, index, rn_bank, E, space, rxn, awr_list, done, scatterdat);
	cudaThreadSynchronize();

}

