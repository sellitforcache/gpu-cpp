#include <cuda.h>
#include <stdio.h>
#include "datadef.h"
#include "wfloat3.h"

__global__ void cscatter_kernel(unsigned N, unsigned RNUM_PER_THREAD, unsigned* active, unsigned* isonum, unsigned * index, float * rn_bank, float * E, source_point * space, unsigned * rxn, float * awr_list, float * Q, unsigned * done, float** scatterdat, float** energydat){


	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}       //return if out of bounds
	
	//remap to active
	//tid=active[tid];
	if(done[tid]){return;}

	// return if not inelastic
	unsigned 	this_rxn 	= rxn[tid];
	if (this_rxn == 91 | this_rxn == 16 | this_rxn == 17 | this_rxn == 37 | this_rxn == 24 | this_rxn == 22 | this_rxn == 28 | this_rxn == 24 | this_rxn == 32 | this_rxn == 33 | this_rxn == 41 ){}
	else {return;}  //return if not continuum inelastic scatter or n,Xn

	//printf("in cscatter, tid %u rxn %u\n",tid,rxn[tid]);

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
	wfloat3 	hats_old(space[tid].xhat,space[tid].yhat,space[tid].zhat);
	float 		this_awr	= awr_list[this_tope];
	float * 	this_Sarray = scatterdat[this_dex];
	float * 	this_Earray =  energydat[this_dex];
	//float 		rn1 		= rn_bank[ tid*RNUM_PER_THREAD + 3];
	//float 		rn2 		= rn_bank[ tid*RNUM_PER_THREAD + 4];
	//float 		rn3 		= rn_bank[ tid*RNUM_PER_THREAD + 5];
	float 		rn4 		= rn_bank[ tid*RNUM_PER_THREAD + 6];
	float 		rn5 		= rn_bank[ tid*RNUM_PER_THREAD + 7];
	float 		rn6 		= rn_bank[ tid*RNUM_PER_THREAD + 8];
	float 		rn7 		= rn_bank[ tid*RNUM_PER_THREAD + 9];
	float 		rn8 		= rn_bank[ tid*RNUM_PER_THREAD + 10];
	float 		rn9 		= rn_bank[ tid*RNUM_PER_THREAD + 11];
	float 		rn10 		= rn_bank[ tid*RNUM_PER_THREAD + 12];

	// internal kernel variables
	float 		mu, phi, next_E, last_E, sampled_E, e_start, E0, E1, Ek, next_e_start, next_e_end, last_e_start, last_e_end, diff;
    unsigned 	vlen, next_vlen, offset, n, law; 
    unsigned  	isdone = 0;
	//float  		E_target     		=   temp * ( -logf(rn1) - logf(rn2)*cosf(pi/2*rn3)*cosf(pi/2*rn3) );
	float 		speed_target     	=   0;//sqrtf(2.0*E_target/(this_awr*m_n));
	float  		speed_n          	=   sqrtf(2.0*this_E/m_n);
	float 		E_new				=   0.0;
	//float 		a 					= 	this_awr/(this_awr+1.0);
	wfloat3 	v_n_cm,v_t_cm,v_n_lf,v_t_lf,v_cm, hats_new, hats_target;
	float 		cdf0,cdf1,e0,e1,A,R,pdf0,pdf1;
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

	//
	//sample energy
	//
	//read in values
	offset = 5;
	memcpy(&last_E,   	&this_Earray[0], sizeof(float));
	memcpy(&next_E,   	&this_Earray[1], sizeof(float));
	memcpy(&vlen,   	&this_Earray[2], sizeof(float));
	memcpy(&next_vlen,	&this_Earray[3], sizeof(float));
	memcpy(&law, 		&this_Earray[4], sizeof(float));
	float r = (this_E-last_E)/(next_E-last_E);
	last_e_start = this_Earray[ offset ];
	last_e_end   = this_Earray[ offset + vlen - 1 ];
	next_e_start = this_Earray[ offset + 3*vlen ];
	next_e_end   = this_Earray[ offset + 3*vlen + next_vlen - 1];
	//unsigned svlen,next_svlen,len;
	//printf("law=%u vlen/next= %u %u, E-last/this/next= %6.4E %6.4E %6.4E\n",law,vlen,next_vlen,last_E,this_E,next_E);
	sampled_E = 0.0;
	if(  rn6 >= r ){   //sample last E
		diff = last_e_end - last_e_start;
		e_start = last_e_start;
		for ( n=0 ; n<vlen-1 ; n++ ){
			cdf0 		= this_Earray[ (offset +   vlen ) + n+0];
			cdf1 		= this_Earray[ (offset +   vlen ) + n+1];
			pdf0		= this_Earray[ (offset + 2*vlen ) + n+0];
			pdf1		= this_Earray[ (offset + 2*vlen ) + n+1];
			e0  		= this_Earray[ (offset          ) + n+0];
			e1  		= this_Earray[ (offset          ) + n+1]; 
			//printf("cdf0=%6.4E\n",cdf0);
			if( rn7 >= cdf0 & rn7 < cdf1 ){
				//printf("found emission energy\n");
				//memcpy(&svlen, 		&this_Sarray[2], sizeof(float));
				//if(svlen!=vlen){printf("svlen length does not match!\n");}
				//len=vlen;
				offset = 4;
				A = this_Sarray[ (offset)      + n ];
				R = this_Sarray[ (offset+vlen) + n ];
				break;
			}
		}
	}
	else{
		diff = next_e_end - next_e_start;
		e_start = next_e_start;
		for ( n=0 ; n<next_vlen-1 ; n++ ){
			cdf0 		= this_Earray[ (offset + 3*vlen +   next_vlen ) + n+0];
			cdf1  		= this_Earray[ (offset + 3*vlen +   next_vlen ) + n+1];
			pdf0		= this_Earray[ (offset + 3*vlen + 2*next_vlen ) + n+0];
			pdf1		= this_Earray[ (offset + 3*vlen + 2*next_vlen ) + n+1];
			e0   		= this_Earray[ (offset + 3*vlen               ) + n+0];
			e1   		= this_Earray[ (offset + 3*vlen               ) + n+1];
			if( rn7 >= cdf0 & rn7 < cdf1 ){
				//printf("found emission energy\n");
				//memcpy(&next_svlen, 		&this_Sarray[3], sizeof(float));
				//if(next_svlen!=next_vlen){printf("next_svlen length does not match!\n");}
				//len=next_vlen;
				offset = 4;
				A = this_Sarray[ (offset+2*vlen)           +n  ] ;
				R = this_Sarray[ (offset+2*vlen+next_vlen) +n  ];
				break;
			}
		}
	}

	// histogram interpolation
	E0 = e0 + (e1-e0)/(cdf1-cdf0)*(rn7-cdf0);
	E1 =  last_e_start + r*( next_e_start - last_e_start );
	Ek =  next_e_end   + r*( next_e_end   - last_e_end   );
	sampled_E = E1 +(E0-e_start)*(Ek-E1)/diff;
	//lin-lin interpolation
	//float m   = (pdf1 - pdf0)/(e1-e0);
	//float arg = pdf0*pdf0 + 2.0 * m * (rn7-cdf0);
	//if(arg<0){arg=0.0;}
	//sampled_E 	= e0 + (  sqrtf( arg ) - pdf0) / m ;


	// find mu
	if(rn8>R){
		float T = (2.0*rn9-1.0)*sinhf(A);
		mu = logf(T+sqrtf(T*T+1.0))/A;
	}
	else{
		mu = logf(rn9*expf(A)+(1.0-rn9)*expf(-A))/A;
	}

	//mu = 2.0*rn9 - 1.0;

	// sample new phi
	phi = 2.0*pi*rn10;

	// pre rotation directions
	hats_old = v_n_cm / v_n_cm.norm2();
	//  create a perpendicular roation vector 
	//wfloat3 rotation_hat( 0.0, 0.0, 1.0 );
	wfloat3 rotation_hat = hats_target.cross( v_n_cm );
	rotation_hat = rotation_hat / rotation_hat.norm2();
	//  do rotations, polar first, then azimuthal
	v_n_cm.rodrigues_rotation( rotation_hat, acosf(mu) );
	v_n_cm.rodrigues_rotation( hats_old,     phi       );
	//  scale to sampled energy
	v_n_cm = v_n_cm/v_n_cm.norm2() * sqrtf(2.0*sampled_E/m_n);
	// transform back to L
	v_n_lf = v_n_cm + v_cm;
	hats_new = v_n_lf / v_n_lf.norm2();
	// calculate energy in lab frame
	E_new = 0.5 * m_n * v_n_lf.dot(v_n_lf);

	// enforce limits
	if ( E_new <= E_cutoff | E_new > E_max ){
		isdone=1;
	}

	if(this_rxn==91){printf("%u % 6.4E %6.4E %6.4E %6.4E %6.4E %6.4E %6.4E %6.4E\n",this_rxn,mu,sampled_E,this_E,E_new, E0, E1, Ek, e_start);}
	//if(this_rxn==91){printf("%6.4E %6.4E %6.4E\n",E_new,this_E,E_new/this_E);}
	//printf("n,vlen %u %u S,Eptrs %p %p Enew,samp %6.4E %6.4E A,R %6.4E %6.4E\n",n,vlen,this_Sarray,this_Earray,E_new,sampled_E,A,R);

	// write results
	done[tid]       = isdone;
	E[tid]          = E_new;
	space[tid].xhat = hats_new.x;
	space[tid].yhat = hats_new.y;
	space[tid].zhat = hats_new.z;

}

void cscatter( unsigned NUM_THREADS, unsigned N, unsigned RNUM_PER_THREAD, unsigned* active, unsigned* isonum, unsigned * index, float * rn_bank, float * E, source_point * space ,unsigned * rxn, float* awr_list, float * Q, unsigned* done, float** scatterdat, float** energydat){

	unsigned blks = ( N + NUM_THREADS - 1 ) / NUM_THREADS;

	cscatter_kernel <<< blks, NUM_THREADS >>> (  N, RNUM_PER_THREAD, active, isonum, index, rn_bank, E, space, rxn, awr_list, Q, done, scatterdat, energydat);
	cudaThreadSynchronize();

}