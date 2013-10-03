#include <cuda.h>
#include <stdio.h>
#include "datadef.h"

__global__ void copy_points_kernel( unsigned Nfrom, unsigned Nout , unsigned current_index , unsigned * is_fissle , source_point * positions_in , source_point * positions_out ){

	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= Nfrom){return;}



}

void copy_points(unsigned blks, unsigned NUM_THREADS, unsigned Nfrom, unsigned Nout , unsigned current_index , unsigned * is_fissle , source_point * positions_in , source_point * positions_out ){

	copy_points_kernel <<< blks, NUM_THREADS >>> (  Nfrom,  Nout ,  current_index , is_fissle , positions_in , positions_out );
	cudaThreadSynchronize();

}

