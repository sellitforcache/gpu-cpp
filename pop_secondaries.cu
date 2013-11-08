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
	float * 		this_array 	= energydata[dex];
	unsigned 		data_dex 	= completed[position];
	source_point 	this_space 	= space[tid];

	// internal data
	unsigned 	k, n, offset, vlen, law;
	float 		sampled_E, phi, mu, rn1, rn2;
	float 		cdf0, cdf1, e0, e1;
	const float  pi           =   3.14159265359 ;
	offset = 2;

	// sample spectrum, set data.  
	// reset self then write elsewhere

	//read in values
	rn1 = rn_bank[ tid*RNUM_PER_THREAD + 11 ];
	memcpy(&vlen, 		&this_array[0], sizeof(float));
	memcpy(&law, 		&this_array[1], sizeof(float));
	//sample energy dist
	for ( n=0 ; n<vlen-1 ; n++ ){
		if( rn1 >= this_array[ (offset+vlen) +n] & rn1 < this_array[ (offset+vlen) +n+1] ){
			cdf0 		= this_array[ (offset+vlen) +n  ];
			cdf1 		= this_array[ (offset+vlen) +n+1];
			e0   		= this_array[ (offset)      +n  ];
			e1   		= this_array[ (offset)      +n+1];
			sampled_E 	= (e1-e0)/(cdf1-cdf0)*(rn1-cdf0)+e0; 
			break;
		}
	};

	//sample isotropic directions
	rn1 = rn_bank[ tid*RNUM_PER_THREAD + 12 ];
	rn2 = rn_bank[ tid*RNUM_PER_THREAD + 13 ];
	mu  = 2.0*rn1-1.0; 
	phi = 2.0*pi*rn2;

	// set self data
	E    [ tid ] 		= sampled_E;
	space[ tid ].xhat 	= sqrtf(1.0-(mu*mu))*cosf(phi);
	space[ tid ].yhat 	= sqrtf(1.0-(mu*mu))*sinf(phi); 
	space[ tid ].zhat 	= mu;
	done [ tid ] 		= 0;
	yield[ tid ] 		= 0; 

	for(k=1 ; k < this_yield ; k++ ){
		//copy in values
		rn1 = rn_bank[ tid*RNUM_PER_THREAD + 11 + k*3];
		memcpy(&vlen, 		&this_array[0], sizeof(float));
		memcpy(&law, 		&this_array[1], sizeof(float));
		//sample energy dist
		for ( n=0 ; n<vlen-1 ; n++ ){
			if( rn1 >= this_array[ (offset+vlen) +n] & rn1 < this_array[ (offset+vlen) +n+1] ){
				cdf0 		= this_array[ (offset+vlen) +n  ];
				cdf1 		= this_array[ (offset+vlen) +n+1];
				e0   		= this_array[ (offset)      +n  ];
				e1   		= this_array[ (offset)      +n+1];
				sampled_E 	= (e1-e0)/(cdf1-cdf0)*(rn1-cdf0)+e0; 
				break;
			}
		}
		//printf("%6.4E\n",sampled_E);

		//sample isotropic directions
		rn1 = rn_bank[ tid*RNUM_PER_THREAD + 12 + k*3];
		rn2 = rn_bank[ tid*RNUM_PER_THREAD + 13 + k*3];
		mu  = 2.0*rn1-1.0; 
		phi = 2.0*pi*rn2;

		//printf("tid %u k %u mu % 6.4E phi % 6.4E rn1 % 6.4E rn2 % 6.4E compactpos %u realpos %u\n",tid,k,mu,phi,rn1,rn2, position, completed[k+position]);

		//check data
		//printf("done? %u\n",done[ data_dex ]);

		__syncthreads();

		// set data
		E    [ data_dex +k ] 		= sampled_E;
		space[ data_dex +k ].x 		= this_space.x;
		space[ data_dex +k ].y 		= this_space.y;
		space[ data_dex +k ].z 		= this_space.z;
		space[ data_dex +k ].xhat 	= sqrtf(1.0-(mu*mu))*cosf(phi);
		space[ data_dex +k ].yhat 	= sqrtf(1.0-(mu*mu))*sinf(phi); 
		space[ data_dex +k ].zhat 	= mu;
		done [ data_dex +k ] 		= 0;
		yield[ data_dex +k ] 		= 0; 

	}


}

void pop_secondaries(unsigned blks, unsigned NUM_THREADS,  unsigned N, unsigned RNUM_PER_THREAD, unsigned* d_completed, unsigned* d_scanned, unsigned* d_yield, unsigned* d_done, unsigned* d_index, source_point* d_space, float* d_E , float* d_rn_bank, float ** energydata){

	pop_secondaries_kernel <<< blks, NUM_THREADS >>> ( N, RNUM_PER_THREAD, d_completed, d_scanned, d_yield, d_done, d_index, d_space, d_E , d_rn_bank, energydata);
	cudaThreadSynchronize();

}

