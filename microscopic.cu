#include <cuda.h>
#include <stdio.h>
#include "datadef.h"

__global__ void microscopic_kernel(unsigned N, unsigned n_isotopes, unsigned n_columns, unsigned* isonum, unsigned * index, float * main_E_grid, float * rn_bank, float * E, float * xs_data_MT , unsigned * xs_MT_numbers_total, unsigned * xs_MT_numbers, unsigned * rxn){


	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}

	// load from array
	unsigned  	RNUM_PER_THREAD = 15;
	unsigned 	this_tope 		= isonum[tid];
	unsigned 	dex 			= index[tid];
	unsigned 	tope_beginning;
	unsigned 	tope_ending;
	float 		this_E  		= E[tid];
	float 		rn1 			= rn_bank[tid*RNUM_PER_THREAD + 0];
	float 		cum_prob 		= 0.0;
	unsigned 	k 				= 0;
	unsigned 	this_rxn 		= 999999999;

	if (this_tope == 0){  //first isotope
		tope_beginning = n_isotopes + 0;
		tope_ending    = n_isotopes + xs_MT_numbers_total[0]-1;
	}
	else{  //interior space
		tope_beginning = n_isotopes + xs_MT_numbers_total[this_tope-1];
		tope_ending    = n_isotopes + xs_MT_numbers_total[this_tope]-1;
	}

	//printf("tope,begin,end = %u %u %u\n",this_tope,tope_beginning,tope_ending);

	float xs_total = 0.0;
	float e0 = main_E_grid[dex];
	float e1 = main_E_grid[dex+1];
	float t0,t1;

	// compute the total microscopic cross section for this material
	// linearly interpolate, dex is the row number
	t0 			= xs_data_MT[n_columns* dex    + this_tope];     
	t1 			= xs_data_MT[n_columns*(dex+1) + this_tope];
	xs_total 	= (t1-t0)/(e1-e0)*this_E + t0 ;    

	// determine the reaction for this isotope
	for(k=tope_beginning; k<tope_ending; k++){
		//lienarly interpolate
		t0 = xs_data_MT[n_columns* dex    + k];     
		t1 = xs_data_MT[n_columns*(dex+1) + k];
		cum_prob += ( (t1-t0)/(e1-e0)*this_E + t0 ) / xs_total;
		if(rn1 <= cum_prob){
			// reactions happen in isotope k
			this_rxn = xs_MT_numbers[k];
			break;
		}
	}

	if(this_rxn == 999999999){printf("REACTION NOT SAMPLED CORRECTLY! this_rxn=%u\n",this_rxn);}

	// write results out
	//printf("this_rxn(%d,(1:3))=[%u,%u,%u];\n",tid+1,this_rxn,this_tope,k);
	rxn[tid] = this_rxn;


}

void microscopic(unsigned blks, unsigned NUM_THREADS,  unsigned N, unsigned n_isotopes, unsigned n_columns, unsigned* isonum, unsigned * index, float * main_E_grid, float * rn_bank, float * E, float * xs_data_MT , unsigned * xs_MT_numbers_total, unsigned * xs_MT_numbers, unsigned * rxn){

	microscopic_kernel <<< blks, NUM_THREADS >>> ( N, n_isotopes, n_columns, isonum, index, main_E_grid, rn_bank, E, xs_data_MT , xs_MT_numbers_total, xs_MT_numbers, rxn);
	cudaThreadSynchronize();

}

