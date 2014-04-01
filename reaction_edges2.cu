#include <cuda.h>
#include <stdio.h>

__global__ void reaction_edges_kernel( unsigned N, unsigned* edges, unsigned* rxn){

	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	if (tid >= N){return;}

	// the reaction vector has been sorted by this time.  data loads are expensive, computation is free, do as much as you can with 2 loads
	// need to find the lower AND upper bounds of the blocks.
	// array structure is:
	// 0  = done flag
	// 1  = lower bound for 2 block 
	// 2  = upper bound for 2 block
	// 3  = lower bound for 51/90 block 
	// 4  = upper bound for 51/90 block
	// 5  = lower bound for 91 block 
	// 6  = upper bound for 91 block
	// 7  = lower bound for 800 block 
	// 8  = upper bound for 800 block
	// 9  = lower bound for 811/845 block 
	// 10 = upper bound for 811/845 block

	unsigned rxn1, rxn2;
	int diff = 0; 

	// load data
	rxn1 = rxn[tid];
	if(tid < N-1){  //both elements if not last
		rxn2 = rxn[tid+1];
		diff = rxn2-rxn1;   //diff should be >0 since the list is sorted
		if(diff<0){printf("non-ascending value found in reaction list at index = %u (%u -> %u)\n!",tid,rxn1,rxn2);}
	}
	else{  //check last or only element, do not return

		if     (rxn1==2) 				{edges[2]  = tid+1;}
		else if(rxn1>=50 & rxn1<=90)	{edges[4]  = tid+1;}
		else if(rxn1==91)				{edges[6]  = tid+1;}
		else if(rxn1==800)				{edges[8]  = tid+1;}
		else if(rxn1>=811 & rxn1<=845)	{edges[10] = tid+1;}

	}

	// first (or only) element doesn't have a preceeding, write it in as the start of something, do not return
	if(tid==0){

		if     (rxn1==2) 				{edges[1] = 0;}
		else if(rxn1>=50 & rxn1<=90)	{edges[3] = 0;}
		else if(rxn1==91)				{edges[5] = 0;}
		else if(rxn1==800)				{edges[7] = 0;}
		else if(rxn1>=811 & rxn1<=845)	{edges[9] = 0;  edges[0]=1;}
		else if(rxn1>845)				{edges[0] = 1;}
		
	}

	// return if the same element, or if last/only element (diff will not be set and remain at 0)
	if(diff<0){return;}   

	// check edge
	if(rxn1<2 		& rxn2>=2)		{edges[1]  = tid+1;} //printf("setting starting edge of 2\n");}
	if(rxn1<=2 		& rxn2>2)		{edges[2]  = tid+1;} //printf("setting ending edge of 2\n");}
	if(rxn1<50 		& rxn2>=50)     {edges[3]  = tid+1;} //printf("setting starting edge of 50\n");}
	if(rxn1<=90 	& rxn2>90)		{edges[4]  = tid+1;} //printf("setting ending edge of 50\n");}
	if(rxn1<91 		& rxn2>=91)		{edges[5]  = tid+1;} //printf("setting starting edge of 91\n");}
	if(rxn1<=91 	& rxn2>91)		{edges[6]  = tid+1;} //printf("setting ending edge of 91\n");}
	if(rxn1<800 	& rxn2>=800)	{edges[7]  = tid+1;} //printf("setting starting edge of 800\n");}
	if(rxn1<=800 	& rxn2>800)		{edges[8]  = tid+1;} //printf("setting ending edge of 800\n");}
	if(rxn1<811 	& rxn2>=811)	{edges[9]  = tid+1;} //printf("setting starting edge of 811\n");}
	if(rxn1<=845  	& rxn2>845)		{edges[10] = tid+1;} //printf("setting ending edge of 811\n");}


}

void reaction_edges( unsigned NUM_THREADS,  unsigned N, unsigned* d_edges, unsigned* d_rxn){

	if(N<1){return;}
	unsigned blks = ( N + NUM_THREADS - 1 ) / NUM_THREADS;

	reaction_edges_kernel <<< blks, NUM_THREADS >>> ( N, d_edges, d_rxn);
	cudaThreadSynchronize();

}

