#include <cuda.h>

__global__ void reaction_edges_kernel( unsigned N, unsigned* edges, unsigned* rxn){

	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}

	// the reaction vector has been sorted by this time.

	if(rxn[tid]<2 & rxn[tid+1]==2){
		edges[0]=tid+1;
	}
	else if(rxn[tid]==2 & rxn[tid+1]>2){
		edges[1]=tid;
	}
	else if(rxn[tid]<11 & rxn[tid+1]>=11){
		edges[2]=tid+1;
	}
	else if(rxn[tid]<45 & rxn[tid+1]>=45){
		edges[3]=tid+1;
	}
	else if(rxn[tid]<51 & rxn[tid+1]>=51){
		edges[4]=tid+1;
	}
	else if(rxn[tid]<90 & rxn[tid+1]>=90){
		edges[5]=tid+1;
	}
	else if(rxn[tid]<91 & rxn[tid+1]=91){
		edges[6]=tid;
	}
	else if(rxn[tid]<=91 & rxn[tid+1]>91){
		edges[7]=tid;
	}


}

void reaction_edges( unsigned NUM_THREADS,  unsigned N, unsigned* d_edges, unsigned* d_rxn){

	unsigned blks = ( N + NUM_THREADS - 1 ) / NUM_THREADS;

	reaction_edges_kernel <<< blks, NUM_THREADS >>> ( N, d_edges, d_rxn);
	cudaThreadSynchronize();

}

