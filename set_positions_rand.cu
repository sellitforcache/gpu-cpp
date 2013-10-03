#include <cuda.h>
#include <stdio.h>
#include "datadef.h"

__global__ void set_positions_rand_kernel(unsigned N , unsigned RNUM_PER_THREAD, source_point * positions_ptr , float * rn_bank , float x_min , float y_min , float z_min , float x_max , float y_max , float z_max ){

    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid>=N){return;}

    const float rn1   =  rn_bank[ (tid * RNUM_PER_THREAD) + 0];
    const float rn2   =  rn_bank[ (tid * RNUM_PER_THREAD) + 1];
    const float rn3   =  rn_bank[ (tid * RNUM_PER_THREAD) + 2];
    const float rn4   =  rn_bank[ (tid * RNUM_PER_THREAD) + 3];
    const float rn5   =  rn_bank[ (tid * RNUM_PER_THREAD) + 4];
    const float pi    =   3.14159265359 ;
    const float mu    = ( rn1 ) * 2.0 - 1.0;
    const float theta = ( rn2 ) * 2.0 * pi ;

    positions_ptr[tid].samp_dist =     500;   
    positions_ptr[tid].x         =     0.9 * ( ( x_max - x_min ) * rn3 + x_min );  
    positions_ptr[tid].y         =     0.9 * ( ( y_max - y_min ) * rn4 + y_min );  
    positions_ptr[tid].z         =     0.9 * ( ( z_max - z_min ) * rn5 + z_min ); 
    positions_ptr[tid].xhat      =  sqrtf(1-mu*mu) * cosf( theta );
    positions_ptr[tid].yhat      =  sqrtf(1-mu*mu) * sinf( theta );
    positions_ptr[tid].zhat      =        mu;

    printf("tid=%d, rn1=%10.8E, rn2=%10.8E, x=%6.3f, y=%6.3f, z=%6.3f, xhat=%6.3f, yhat=%6.3f, zhat=%6.3f \n",tid,rn1,rn2,positions_ptr[tid].x,positions_ptr[tid].y,positions_ptr[tid].z,positions_ptr[tid].xhat,positions_ptr[tid].yhat,positions_ptr[tid].zhat);

}

void set_positions_rand(unsigned blks, unsigned NUM_THREADS, unsigned N, unsigned RNUM_PER_THREAD, source_point * d_space , float * d_rn_bank, float * outer_cell_dims){

    printf("%u %u %u %u %p %p\n",blks,NUM_THREADS,N,RNUM_PER_THREAD,d_space,d_rn_bank);
    printf("% 10.8f % 10.8f % 10.8f % 10.8f % 10.8f % 10.8f \n",outer_cell_dims[0], outer_cell_dims[1], outer_cell_dims[2], outer_cell_dims[3], outer_cell_dims[4], outer_cell_dims[5]);
    set_positions_rand_kernel <<<  blks, NUM_THREADS >>> ( N , RNUM_PER_THREAD, d_space , d_rn_bank, outer_cell_dims[0], outer_cell_dims[1], outer_cell_dims[2], outer_cell_dims[3], outer_cell_dims[4], outer_cell_dims[5]);
    cudaThreadSynchronize();

}