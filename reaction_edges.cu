#include <cuda.h>

__global__ void reaction_edges_kernel( unsigned N, unsigned* edges, unsigned* rxn){

	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N-1){return;}

	// the reaction vector has been sorted by this time.

	unsigned rxn1 = rxn[tid];
	unsigned rxn2 = rxn[tid+1];

	if(rxn1<2 & rxn2>=2){  //upper bound for <2
		edges[0]=tid;
	}
	if(rxn1<=2 & rxn2>2){  //upper bound for 2
		edges[1]=tid;
	}
	if(rxn1<=90 & rxn2>90){  // upper bound for 90
		edges[2]=tid;
	}
	if(rxn1<=91 & rxn2>91){  // upper bound for 91
		edges[3]=tid;
	}
	if(rxn1<=800 & rxn2>800){  //upper bound for 800
		edges[4]=tid;
	}
	if(rxn1<=845 & rxn2>845){  //upper bound for 845
		edges[5]=tid;
	}


}

void reaction_edges( unsigned NUM_THREADS,  unsigned N, unsigned* d_edges, unsigned* d_rxn){

	unsigned blks = ( N + NUM_THREADS - 1 ) / NUM_THREADS;

	reaction_edges_kernel <<< blks, NUM_THREADS >>> ( N, d_edges, d_rxn);
	cudaThreadSynchronize();

}

