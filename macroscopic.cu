#include <cuda.h>
#include <stdio.h>
#include "datadef.h"
#include "LCRNG.cuh"

__global__ void macroscopic_kernel(unsigned N, unsigned n_isotopes, unsigned n_columns, unsigned* active, source_point * space, unsigned* isonum, unsigned * index, unsigned * matnum, float * main_E_grid, unsigned * rn_bank, float * E, float * xs_data_MT , float* material_matrix, unsigned* done){


	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}
	
	//remap to active
	//tid=active[tid];
	if(done[tid]){return;}

	// load from arrays
	unsigned  	RNUM_PER_THREAD = 15;
	unsigned 	this_mat 		= matnum[tid];
	unsigned 	dex 			= index[tid];   
	float 		this_E  		= E[tid];
	unsigned 	rn 				= rn_bank[tid];
	float 		samp_dist 		= 0.0;
	float 		cum_prob 		= 0.0;
	unsigned 	tope 			= 999999999;

	float macro_t_total = 0.0;
	float e0 = main_E_grid[dex];
	float e1 = main_E_grid[dex+1];
	float t0,t1;

	// compute the total macroscopic cross section for this material
	for(int k=0; k<n_isotopes; k++){
		//lienarly interpolate
		t0 = xs_data_MT[n_columns* dex    + k];     //dex is the row number
		t1 = xs_data_MT[n_columns*(dex+1) + k];
		macro_t_total += ( (t1-t0)/(e1-e0)*(this_E-e0) + t0 ) * material_matrix[n_isotopes*this_mat+k];    //interpolated micro times number density
		//printf("mat %u - density of tope %u = %6.3E\n",this_mat,k,material_matrix[n_isotopes*this_mat+k]);
	}

	// compute the interaction length
	samp_dist = -logf(get_rand(&rn))/macro_t_total;
	float rn1 = get_rand(&rn);

	// determine the isotope which the reaction occurs
	for(int k=0; k<n_isotopes; k++){
		//lienarly interpolate
		t0 = xs_data_MT[n_columns* dex    + k];     
		t1 = xs_data_MT[n_columns*(dex+1) + k];
		cum_prob += ( ( (t1-t0)/(e1-e0)*(this_E-e0) + t0 ) * material_matrix[n_isotopes*this_mat+k] ) / macro_t_total;
		if( k==n_isotopes-1 & cum_prob<1.0){cum_prob=1.0;}  //sometimes roundoff makes this a problem
		if( rn1 <= cum_prob){
			// reactions happen in isotope k
			tope = k;
			break;
		}
	}
	if(tope == 999999999){ 
		printf("macro - ISOTOPE NOT SAMPLED CORRECTLY! tope=%u E=%10.8E dex=%u mat=%u rn=%u cum_prob=%12.10E\n",tope, this_E, dex, this_mat, rn, cum_prob);
	}

	// write results out
	space[tid].samp_dist 	= samp_dist;
	space[tid].macro_t 		= macro_t_total;
	isonum[tid] 			= tope;
	rn_bank[tid] 			= rn;


}

void macroscopic( unsigned NUM_THREADS,  unsigned N, unsigned Ntopes, unsigned n_col , unsigned* active, source_point * space, unsigned* isonum, unsigned * index, unsigned * matnum, float * main_E_grid, unsigned * rn_bank, float * E, float * xs_data_MT , float* material_matrix, unsigned* done){

	unsigned blks = ( N + NUM_THREADS - 1 ) / NUM_THREADS;

	macroscopic_kernel <<< blks, NUM_THREADS >>> ( N, Ntopes, n_col, active, space, isonum, index, matnum, main_E_grid, rn_bank, E, xs_data_MT , material_matrix, done);
	cudaThreadSynchronize();

}

