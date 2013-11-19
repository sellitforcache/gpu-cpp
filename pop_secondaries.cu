#include <cuda.h>
#include <stdio.h>
#include "datadef.h"

__global__ void pop_secondaries_kernel(unsigned N, unsigned RNUM_PER_THREAD, unsigned* completed, unsigned* scanned, unsigned* yield, unsigned* done, unsigned* index, source_point* space, float* E , float* rn_bank, float**  energydata){

	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}
	if (yield[tid]==0){return;}

	// external data
	unsigned 		position 	= scanned[tid];
	unsigned 		this_yield 	= yield[tid];
	unsigned 		dex  		= index[tid];
	float 			this_E 		= E[tid];
	float * 		this_array 	= energydata[dex];
	unsigned 		data_dex 	= 0;
	source_point 	this_space 	= space[tid];

	// internal data
	float 		Emin=1e-11;
	float 		Emax=20.0;
	unsigned 	k, n, offset, vlen, next_vlen, law, fork;
	float 		sampled_E, phi, mu, rn1, rn2, last_E, next_E;
	float 		cdf0, cdf1, e0, e1, m, pdf0, pdf1;
	const float  pi           =   3.14159265359 ;
	offset = 5;

	// sample spectrum, set data.  
	// reset self then write elsewhere

	//read in values
	rn1 = rn_bank[ tid*RNUM_PER_THREAD + 11 ];
	rn2 = rn_bank[ tid*RNUM_PER_THREAD + 12 ];
	memcpy(&last_E,   	&this_array[0], sizeof(float));
	memcpy(&next_E,   	&this_array[1], sizeof(float));
	memcpy(&vlen,   	&this_array[2], sizeof(float));
	memcpy(&next_vlen,	&this_array[3], sizeof(float));
	memcpy(&law, 		&this_array[4], sizeof(float));
	//printf("law=%u vlen/next= %u %u, E-last/this/next= %6.4E %6.4E %6.4E\n",law,vlen,next_vlen,last_E,this_E,next_E);
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
	m 			= (pdf1 - pdf0)/(e1-e0);
	sampled_E 	= e0 + (  sqrtf(pdf0*pdf0 + 2.0 * m * (rn1-cdf1)) - pdf0) / m ;
	//printf("%u %u %u %p %6.4E %u %u %6.4E %6.4E %6.4E %6.4E %6.4E %6.4E %6.4E %6.4E\n",fork,n,dex,this_array,rn1,next_vlen,vlen,this_E,e0,e1,cdf0,cdf1,pdf0,pdf1,sampled_E);

	//sample isotropic directions
	rn1 = rn_bank[ tid*RNUM_PER_THREAD + 13 ];
	rn2 = rn_bank[ tid*RNUM_PER_THREAD + 14 ];
	mu  = 2.0*rn1-1.0; 
	phi = 2.0*pi*rn2;

	// set self data
	E    [ tid ] 		= sampled_E;
	space[ tid ].xhat 	= sqrtf(1.0-(mu*mu))*cosf(phi);
	space[ tid ].yhat 	= sqrtf(1.0-(mu*mu))*sinf(phi); 
	space[ tid ].zhat 	= mu;
	done [ tid ] 		= 0;
	yield[ tid ] 		= 0; 

	for(k=0 ; k < this_yield-1 ; k++ ){
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
		m 			= (pdf1 - pdf0)/(e1-e0);
		sampled_E 	= e0 + (  sqrtf(pdf0*pdf0 + 2.0 * m * (rn1-cdf1)) - pdf0) / m;
		//printf("%u %u %u %p %6.4E %u %u %6.4E %6.4E %6.4E %6.4E %6.4E %6.4E %6.4E %6.4E\n",fork,n,dex,this_array,rn1,next_vlen,vlen,this_E,e0,e1,cdf0,cdf1,pdf0,pdf1,sampled_E);

		//sample isotropic directions
		rn1 = rn_bank[ tid*RNUM_PER_THREAD + 13 + k*3];
		rn2 = rn_bank[ tid*RNUM_PER_THREAD + 14 + k*3];
		mu  = 2.0*rn1-1.0; 
		phi = 2.0*pi*rn2;

		//printf("tid %u k %u mu % 6.4E phi % 6.4E rn1 % 6.4E rn2 % 6.4E compactpos %u realpos %u\n",tid,k,mu,phi,rn1,rn2, position, completed[k+position]);

		//check data
		//printf("done? %u\n",done[ data_dex ]);

		//check limits
		if (sampled_E >= Emax){sampled_E = Emax * 0.9;}//printf("enforcing limits in pop data_dex=%u, sampled_E = %6.4E\n",data_dex,sampled_E);}
		if (sampled_E <= Emin){sampled_E = Emin * 1.1;}//printf("enforcing limits in pop data_dex=%u, sampled_E = %6.4E\n",data_dex,sampled_E);}

		// sync before writes
		__syncthreads();

		// set data
		E    [ data_dex ] 		= sampled_E;
		space[ data_dex ].x 	= this_space.x;
		space[ data_dex ].y 	= this_space.y;
		space[ data_dex ].z 	= this_space.z;
		space[ data_dex ].xhat 	= sqrtf(1.0-(mu*mu))*cosf(phi);
		space[ data_dex ].yhat 	= sqrtf(1.0-(mu*mu))*sinf(phi); 
		space[ data_dex ].zhat 	= mu;
		done [ data_dex ] 		= 0;
		yield[ data_dex ] 		= 0; 

	}


}

void pop_secondaries( unsigned NUM_THREADS,  unsigned N, unsigned RNUM_PER_THREAD, unsigned* d_completed, unsigned* d_scanned, unsigned* d_yield, unsigned* d_done, unsigned* d_index, source_point* d_space, float* d_E , float* d_rn_bank, float ** energydata){

	unsigned blks = ( N + NUM_THREADS - 1 ) / NUM_THREADS;

	pop_secondaries_kernel <<< blks, NUM_THREADS >>> ( N, RNUM_PER_THREAD, d_completed, d_scanned, d_yield, d_done, d_index, d_space, d_E , d_rn_bank, energydata);
	cudaThreadSynchronize();

}

