#include <cuda.h>
#include <stdio.h>
#include "datadef.h"

__global__ void find_E_grid_index_kernel(unsigned N, unsigned N_energies, float * main_E_grid, float* E , unsigned * index){

	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}

	float value = E[tid];
	unsigned done = 0;
	unsigned cnt  = 1;
	unsigned powtwo = 2;
	int dex  = (N_energies-1) / 2;  //N_energiesgth starts at 1, duh

	//printf("%p %d %10.4E\n",main_E_grid,dex,value);
	//int k;

	while(!done){
		powtwo = powtwo * 2;
		if      ( 	main_E_grid[dex]   <= value && 
					main_E_grid[dex+1] >  value ) { done = 1; }
		else if ( 	main_E_grid[dex]   >  value ) { dex  = dex - ((N_energies / powtwo) + 1) ; cnt++; }  // +1's are to do a ceiling instead of a floor on integer division
		else if ( 	main_E_grid[dex]   <  value ) { dex  = dex + ((N_energies / powtwo) + 1) ; cnt++; }

		if(cnt>30){
			done=1;
			printf("binary search iteration overflow! %p %d % 10.8f\n",main_E_grid,N_energies,value);
			dex=0;
		}

		// edge checks... fix later???
		if(dex<0){
			//printf("binary search error! dex=%d, (ptr,N_energies,value) %p %d % 10.8f\n",dex,main_E_grid,N_energies,value);
			//for(k=0;k<N_energies;k++){printf("%10.8E\n",main_E_grid[k]);}
			dex=0;
			//done=1;
		}
		if(dex>=N_energies){
			//printf("binary search error! dex=%d, (ptr,N_energies,value) %p %d % 10.8f\n",dex,main_E_grid,N_energies,value);
			//for(k=0;k<N_energies;k++){printf("%10.8E\n",main_E_grid[k]);}
			dex=N_energies-1;
			//done=1;
		}
	}


	//write output index
	index[tid]=dex;

}

void find_E_grid_index(unsigned blks, unsigned NUM_THREADS, unsigned N, unsigned N_energies, float * main_E_grid, float* E , unsigned * index ){

	find_E_grid_index_kernel <<< blks, NUM_THREADS >>> ( N, N_energies, main_E_grid,  E , index );
	cudaThreadSynchronize();

}

