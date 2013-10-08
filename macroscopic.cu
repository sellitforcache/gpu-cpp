#include <cuda.h>
#include <stdio.h>
#include "datadef.h"

__global__ void macroscopic_kernel( ){

	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= Nvalid[0]){return;}


}

void macroscopic(unsigned blks, unsigned NUM_THREADS,  ){

	macroscopic_kernel <<< blks, NUM_THREADS >>> (   );
	cudaThreadSynchronize();

}

