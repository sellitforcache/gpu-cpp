#include <cuda.h>
#include <stdio.h>
#include "datadef.h"

__global__ void find_E_grid_index_quad_kernel(unsigned N, unsigned depth, unsigned width, unsigned* active, qnode* tree, float* E, unsigned * index, unsigned* done){

	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}

	//remap to active
	tid=active[tid];

	//load data
	float this_E = E[tid];
	unsigned dex=0;

	//find initial node bin
	unsigned 	bin  = tid*width/N;
	qnode 		node = tree[bin];
	qnode*		next_ptr;

	//printf("tid=%d bin=%u values=(%6.4E %6.4E %6.4E %6.4E %6.4E)\n",tid,bin,tree[bin].values[0],tree[bin].values[1],tree[bin].values[2],tree[bin].values[3],tree[bin].values[4]);
	//return;
	// traverse tree
	for(unsigned it=0; it<depth ;it++){
		for(unsigned k=0;k<4;k++){
			if(this_E >= node.values[k] & this_E < node.values[k+1]){
				next_ptr = node.leaves[k];
				break;
			}
		}
		memcpy(&node,next_ptr,sizeof(qnode));
	}
	// find final bin, copy index
	for(unsigned k=0;k<4;k++){
		if(this_E >= node.values[k] & this_E < node.values[k+1]){
			dex=(unsigned long)node.leaves[k];
			break;
		}
	}
	//printf("dex=%u\n",dex);

	//write output index
	index[tid]=dex;

}

void find_E_grid_index_quad( unsigned NUM_THREADS, unsigned N, unsigned depth, unsigned width, unsigned* active, qnode* tree, float* E, unsigned * index, unsigned* done){

	unsigned blks = ( N + NUM_THREADS - 1 ) / NUM_THREADS;

	find_E_grid_index_quad_kernel <<< blks, NUM_THREADS >>> (  N,  depth,  width, active, tree, E, index, done);
	cudaThreadSynchronize();

}

