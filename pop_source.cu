#include <cuda.h>
#include <stdio.h>
#include "datadef.h"
#include "wfloat3.h"

__global__ void pop_source_kernel(unsigned N, unsigned RNUM_PER_THREAD, unsigned* isonum, unsigned* completed, unsigned* scanned, unsigned* yield, unsigned* done, unsigned* index, unsigned* rxn, source_point* space, float* E , float* rn_bank, float**  energydata, source_point* space_out, float* E_out, float * awr_list){

	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}
	if (yield[tid]==0){return;}
	//if(done[tid]){return;}

	// external data
	unsigned 		position 	= scanned[tid];
	unsigned 		this_tope 	= isonum[tid];
	unsigned 		this_yield 	= yield[tid];
	unsigned 		dex  		= index[tid];
	float 			this_E 		= E[tid];
	unsigned 		this_rxn 	= rxn[tid];
	float * 		this_array 	= energydata[dex];
	unsigned 		data_dex 	= 0;
	source_point 	this_space 	= space[tid];
	wfloat3 		hats_old(this_space.xhat,this_space.yhat,this_space.zhat);
	float 			this_awr	= awr_list[this_tope];
	//printf("tid %u rxn %u space % 6.4E % 6.4E % 6.4E\n",tid,this_rxn,this_space.x,this_space.y,this_space.z);

	// internal data
	float 		Emin=1e-11;
	float 		Emax=20.0;
	unsigned 	k, n, offset, vlen, next_vlen, law, fork;
	float 		sampled_E, phi, mu, rn1, rn2, last_E, next_E;
	float 		cdf0, cdf1, e0, e1, m, pdf0, pdf1, arg,x,y,z;
	const float  pi           =   3.14159265359 ;
	const float  m_n          =   1.00866491600 ; // u
	float 		speed_n 	=   sqrtf(2.0*this_E/m_n);

	//read in values
	offset = 5;
	//printf("rxn %u eptr %p\n",this_rxn,this_array);
	memcpy(&last_E,   	&this_array[0], sizeof(float));
	memcpy(&next_E,   	&this_array[1], sizeof(float));
	memcpy(&vlen,   	&this_array[2], sizeof(float));
	memcpy(&next_vlen,	&this_array[3], sizeof(float));
	memcpy(&law, 		&this_array[4], sizeof(float)); 

	for(k=0 ; k < this_yield ; k++ ){
		//get proper data index
		data_dex=completed[position+k];
		//printf("tid %u position %u k %u data_dex %u done %u (xyz) % 6.4E % 6.4E % 6.4E\n",tid,position,k,data_dex,done[data_dex],this_space.x,this_space.y,this_space.z);
		//make sure data is done
		if(!done[data_dex]){printf("overwriting into active data!\n");}
		//copy in values
		rn1 = rn_bank[ tid*RNUM_PER_THREAD + 11 + k*3];
		rn2 = rn_bank[ tid*RNUM_PER_THREAD + 12 + k*3];
		//sample energy dist
		sampled_E = 0.0;
		if(  rn2 <= (next_E-this_E)/(next_E-last_E) ){   //sample last E
			fork=0;
			for ( n=0 ; n<vlen-1 ; n++ ){
				cdf0 		= this_array[ (offset +   vlen ) + n+0];
				cdf1 		= this_array[ (offset +   vlen ) + n+1];
				pdf0		= this_array[ (offset + 2*vlen ) + n+0];
				pdf1		= this_array[ (offset + 2*vlen ) + n+1];
				e0  		= this_array[ (offset          ) + n+0];
				e1  		= this_array[ (offset          ) + n+1]; 
				if( rn1 >= cdf0 & rn1 < cdf1 ){
					break;
				}
			}
		}
		else{
			fork=1;
			for ( n=0 ; n<next_vlen-1 ; n++ ){
				cdf0 		= this_array[ (offset + 3*vlen +   next_vlen ) + n+0];
				cdf1  		= this_array[ (offset + 3*vlen +   next_vlen ) + n+1];
				pdf0		= this_array[ (offset + 3*vlen + 2*next_vlen ) + n+0];
				pdf1		= this_array[ (offset + 3*vlen + 2*next_vlen ) + n+1];
				e0   		= this_array[ (offset + 3*vlen               ) + n+0];
				e1   		= this_array[ (offset + 3*vlen               ) + n+1];
				if( rn1 >= cdf0 & rn1 < cdf1 ){
					break;
				}
			}
		}
	
		// interpolate the values
		m 	= (pdf1 - pdf0)/(e1-e0);
		arg = pdf0*pdf0 + 2.0 * m * (rn1-cdf0);
		if(arg<0){arg=0.0;}
		sampled_E 	= e0 + (  sqrtf( arg ) - pdf0) / m ;
		//sampled_E = e0 + (rn1-cdf0)/pdf0;
		//printf("%u %u %u %u %u %p %6.4E %u %u %6.4E %6.4E %6.4E %6.4E %6.4E %6.4E %6.4E %6.4E %6.4E\n",tid,tid*RNUM_PER_THREAD + 11 + (k+1)*3,fork,n,dex,this_array,rn1,next_vlen,vlen,this_E,e0,e1,cdf0,cdf1,pdf0,pdf1,m,sampled_E);

		//if(this_rxn>=16 & this_rxn<=17){ //convert to CM
		//	wfloat3 hats_target;
		//	mu  = (2.0*   rn_bank[ tid*RNUM_PER_THREAD + 9  + k*3]) - 1.0;
		//	phi = 2.0*pi* rn_bank[ tid*RNUM_PER_THREAD + 10 + k*3];
		//	hats_target.x = sqrtf(1.0-(mu*mu))*cosf(phi);
		//	hats_target.y = sqrtf(1.0-(mu*mu))*sinf(phi); 
		//	hats_target.z = mu;
		//	// calc v_lf
		//	wfloat3 v_n_lf = hats_old    * speed_n;
		//	// calculate  v_cm
		//	wfloat3 v_cm = (v_n_lf)/(1.0+this_awr);
		//	//transform neutron velocity into CM frame
		//	wfloat3 v_n_cm = v_n_lf - v_cm;
		//	// pre rotation directions
		//	wfloat3 hats_old = v_n_cm / v_n_cm.norm2();
		//	//  create a perpendicular roation vector 
		//	//wfloat3 rotation_hat( 0.0, 0.0, 1.0 );
		//	rn1 = rn_bank[ tid*RNUM_PER_THREAD + 13 + k*3];
		//	rn2 = rn_bank[ tid*RNUM_PER_THREAD + 14 + k*3];
		//	mu  = 2.0*rn1-1.0; 
		//	phi = 2.0*pi*rn2;
		//	wfloat3 rotation_hat = hats_target.cross( v_n_cm );
		//	rotation_hat = rotation_hat / rotation_hat.norm2();
		//	//  do rotations, polar first, then azimuthal
		//	v_n_cm.rodrigues_rotation( rotation_hat, acosf(mu) );
		//	v_n_cm.rodrigues_rotation( hats_old,     phi       );
		//	//  scale to sampled energy
		//	v_n_cm = v_n_cm/v_n_cm.norm2() * sqrtf(2.0*sampled_E/m_n);
		//	// transform back to L
		//	v_n_lf = v_n_cm + v_cm;
		//	wfloat3 hats_new = v_n_lf / v_n_lf.norm2();
		//	// calculate energy in lab frame
		//	sampled_E = 0.5 * m_n * v_n_lf.dot(v_n_lf);
		//	x = hats_new.x;
		//	y = hats_new.y;
		//	z = hats_new.z;
		//}
		//else{
			//sample isotropic directions
			rn1 = rn_bank[ tid*RNUM_PER_THREAD + 13 + k*3];
			rn2 = rn_bank[ tid*RNUM_PER_THREAD + 14 + k*3];
			mu  = 2.0*rn1-1.0; 
			phi = 2.0*pi*rn2;
			x = sqrtf(1.0-(mu*mu))*cosf(phi);
			y = sqrtf(1.0-(mu*mu))*sinf(phi);
			z = mu;
		//}
	
		//printf("tid %u k %u mu % 6.4E phi % 6.4E rn1 % 6.4E rn2 % 6.4E compactpos %u realpos %u\n",tid,k,mu,phi,rn1,rn2, position, completed[k+position]);
	
		//check data
		//printf("done? %u\n",done[ data_dex ]);
	
		//check limits
		if (sampled_E >= Emax){sampled_E = Emax * 0.9;}//printf("enforcing limits in pop data_dex=%u, sampled_E = %6.4E\n",data_dex,sampled_E);}
		if (sampled_E <= Emin){sampled_E = Emin * 1.1;}//printf("enforcing limits in pop data_dex=%u, sampled_E = %6.4E\n",data_dex,sampled_E);}


		// sync before writes
		__syncthreads();

		// set data
		if(this_rxn==18){
			E_out 	 [ data_dex ] 		= sampled_E;
			space_out[ data_dex ].x 	= this_space.x;
			space_out[ data_dex ].y 	= this_space.y;
			space_out[ data_dex ].z 	= this_space.z;
			space_out[ data_dex ].xhat 	= x;
			space_out[ data_dex ].yhat 	= y;
			space_out[ data_dex ].zhat 	= z;
			rxn  	 [ data_dex ]		= this_rxn;
		}
		else{ // pass to cscatter
			E_out 	 [ data_dex ] 		= this_E;
			space_out[ data_dex ].x 	= this_space.x;
			space_out[ data_dex ].y 	= this_space.y;
			space_out[ data_dex ].z 	= this_space.z;
			space_out[ data_dex ].xhat 	= this_space.xhat;
			space_out[ data_dex ].yhat 	= this_space.yhat;
			space_out[ data_dex ].zhat 	= this_space.zhat;
			rxn  	 [ data_dex ]		= this_rxn;
			index 	 [ data_dex ] 		= dex;
			isonum   [ data_dex ]  		= this_tope;
		}
		//done [ data_dex ] 		= 0;
		//yield[ data_dex ] 		= 0;
		//printf("popped - dex %u space % 6.4E % 6.4E % 6.4E\n",data_dex,sampled_E,this_space.x,this_space.y,this_space.z); 

	}


}

void pop_source( unsigned NUM_THREADS,  unsigned N, unsigned RNUM_PER_THREAD, unsigned* isonum, unsigned* d_completed, unsigned* d_scanned, unsigned* d_yield, unsigned* d_done, unsigned* d_index, unsigned* d_rxn, source_point* d_space, float* d_E , float* d_rn_bank, float ** energydata, source_point* space_out, float* E_out, float * awr_list){

	unsigned blks = ( N + NUM_THREADS - 1 ) / NUM_THREADS;

	pop_source_kernel <<< blks, NUM_THREADS >>> ( N, RNUM_PER_THREAD, isonum, d_completed, d_scanned, d_yield, d_done, d_index, d_rxn, d_space, d_E , d_rn_bank, energydata, space_out, E_out, awr_list);
	cudaThreadSynchronize();

}

