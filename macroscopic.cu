#include <cuda.h>
#include <stdio.h>
#include "datadef.h"

__global__ void macroscopic_kernel(unsigned N, unsigned * index, unsigned * matnum, float * main_E_grid, float * rn_bank, float * E, float * xs_data_MT ){

/*
	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}

	// load from arrays
	unsigned 	this_mat 		= matnum[tid];
	unsigned 	dex 			= index[tid];
	float 		this_E  		= E[tid];

	unsigned n_istopes = material_isotope_matrix[this_mat];
	unsigned * material_isotope_matrix
	float macro_t_total = 0.0;
	float e0 = main_E_grid[dex];
	float e1 = main_E_grid[dex+1];
	float t0,t1;

	// compute the total macroscopic cross section for this material
	for(int k=0; k<n_isotopes; k++){

		//lienarly interpolate


	}

	// compute the interaction length

	// determine the isotope which the reaction occurs
*/

}

void macroscopic(unsigned blks, unsigned NUM_THREADS,  unsigned N, unsigned * index, unsigned * matnum, float * main_E_grid, float * rn_bank, float * E, float * xs_data_MT ){

	macroscopic_kernel <<< blks, NUM_THREADS >>> (   N, index,  matnum,  main_E_grid,  rn_bank, E, xs_data_MT );
	cudaThreadSynchronize();

}

