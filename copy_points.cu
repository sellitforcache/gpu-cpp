#include <cuda.h>
#include <stdio.h>
#include "datadef.h"

__global__ void copy_points_kernel( unsigned Nout, unsigned * Nvalid , unsigned current_index , unsigned * to_valid, source_point * positions_out , source_point * positions_in ){

	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= Nvalid[0]){return;}

	unsigned index_in  = to_valid[tid];
	unsigned index_out = current_index + tid;
	if (index_out>=Nout){index_out=index_out-Nout;} //wrap to start

	// copy points
	positions_out[index_out].x = positions_in[index_in].x; 
	positions_out[index_out].y = positions_in[index_in].y; 
	positions_out[index_out].z = positions_in[index_in].z;  

}

void copy_points(unsigned blks, unsigned NUM_THREADS,  unsigned Nout , unsigned * Nvalid,  unsigned current_index , unsigned * to_valid , source_point * positions_out , source_point * positions_in){

	copy_points_kernel <<< blks, NUM_THREADS >>> (  Nout , Nvalid,  current_index , to_valid , positions_out , positions_in );
	cudaThreadSynchronize();

}

