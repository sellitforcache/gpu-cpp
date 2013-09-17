//library and SDK includes
//#include <optix_world.h>
//#include <optix_cuda_interop.h>
//#include <cuda.h>
//#include <curand.h>
//#include <stdlib.h>
//#include <stdio.h>
//#include <math.h>
//#include <time.h>
//#include <sys/time.h>
//#include <regex.h>
//#include <assert.h>
//#include <string.h>
//#include <sys/errno.h>
//#include <unistd.h>
//#include <cudpp_hash.h>
//#include "mt19937ar.h"

// all mt classes
#include "warp.h"

// preprocessor constants and flags
#define NUM_THREADS 256
#define RNUM_PER_THREAD 8//12
#define ISOTOPES_MAX 5
#define print_errors 0

int main(){

	using namespace std;

	///////////////////
	// BEGIN PROGRAM //
	///////////////////

	//print banner
	print_banner();

	// set up geometry
	wgeometry geom;

	geom.add_primitive();
	geom.primitives[0].type=1;
	geom.primitives[0].material=2;
	geom.primitives[0].min[0]=-20.0;
	geom.primitives[0].max[0]=20.0;
	geom.primitives[0].max[1]=1.0;
	geom.primitives[0].add_transform(2,30.0,30.0,30.0,0.0,0.0);
	geom.primitives[0].make_hex_array(7,0.0,0.0,200);
	geom.add_primitive();
	geom.primitives[1].type=2;
	geom.primitives[1].material=1;
	geom.primitives[1].min[0]=-21;
	geom.primitives[1].max[0]=21;
	geom.primitives[1].max[1]=31.0;
	geom.add_primitive();
	geom.primitives[2].type=0;
	geom.primitives[2].material=1;
	geom.primitives[2].min[0]=-42;
	geom.primitives[2].min[1]=-42;
	geom.primitives[2].min[2]=-42;
	geom.primitives[2].max[0]=42;
	geom.primitives[2].max[1]=42;
	geom.primitives[2].max[2]=42;
	geom.update();
	geom.print();


	// set flags, anything needed before cuda context creation
/*

	//////////////////////
	// INIT OPTIX STUFF //
	//////////////////////
	
	//init optix - set programs and variables



	/////////////////////
	// INIT CUDA STUFF //    !! MUST BE DONE AFTER OPTIX !!
	/////////////////////

	//determine cuda parameters
	printf("\e[1;32m%-6s\e[m \n","Determining parameters...");
	int blks=(N+NUM_THREADS-1)/NUM_THREADS;
	// set printf max size
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, (size_t) 10*1048576 );

	///////////////////////////
	// INIT PARTICLE DATASET //
	///////////////////////////

	printf("\e[1;32m%-6s\e[m \n","Initializing particle dataset...");
	//set optix pointer as device pointer
	d_hist.space   = (source_point *) positions_ptr;
	d_hist.rxn     = (unsigned *)           rxn_ptr;
	d_hist.done    = (unsigned *)          done_ptr;
	d_hist.cellnum = (unsigned *)       cellnum_ptr;


	/////////////////////////
	// LOAD CROSS SECTIONS //
	/////////////////////////
	
	// read in cross sections
	printf("\e[1;32m%-6s\e[m \n","Reading cross sections...");
	
	// copy cross sections over
	printf("\e[1;32m%-6s\e[m \n","Copying cross sections to GPU... which?");

	/////////////////////////////////
	// SET MATERIAL CONCENTRATIONS //
	/////////////////////////////////

	set_concentrations(&all_geom,xsdat);

	/////////////////////////////////////////////////
	// MAKE MATERIAL HASHES and COPY MATERIAL INFO //
	/////////////////////////////////////////////////
	
	// make material data arrays, copy to device for hash table creation, copy material table over
	printf("\e[1;32m%-6s\e[m \n","Creating material key/value pairs...");
	init_mat(&all_geom,xsdat,&d_materials_ptr,&d_hash_val_mate,&d_hash_val_fiss ,&d_hash_key);
	
	//////////////////////
	// INIT CUDPP STUFF //
	//////////////////////
	
	printf("\e[1;32m%-6s\e[m \n","Initializing CUDPP...");
	// global objects
	res = cudppCreate(&theCudpp);
	if (res != CUDPP_SUCCESS){fprintf(stderr, "Error initializing CUDPP Library.\n");}
	
	printf("\e[0;32m%-6s\e[m \n","  Configuring sort...");
	// sort stuff
	compact_config.op = CUDPP_ADD;
	compact_config.datatype = CUDPP_INT;
	compact_config.algorithm = CUDPP_COMPACT;
	compact_config.options = CUDPP_OPTION_FORWARD;
	res = cudppPlan(theCudpp, &compactplan, compact_config, N, 1, 0);
	if (CUDPP_SUCCESS != res){printf("Error creating CUDPPPlan for compact\n");exit(-1);}
	cudaThreadSynchronize();

	printf("\e[0;32m%-6s\e[m \n","  Configuring reduction...");
	// int reduction stuff
	redu_int_config.op = CUDPP_ADD;
	redu_int_config.datatype = CUDPP_INT;
	redu_int_config.algorithm = CUDPP_REDUCE;
	redu_int_config.options = 0;
	res = cudppPlan(theCudpp, &reduplan_int, redu_int_config, N, 1, 0);
	if (CUDPP_SUCCESS != res){printf("Error creating CUDPPPlan for reduction\n");exit(-1);}
	cudaThreadSynchronize();
	// float reduction stuff
	redu_float_config.op = CUDPP_ADD;
	redu_float_config.datatype = CUDPP_FLOAT;
	redu_float_config.algorithm = CUDPP_REDUCE;
	redu_float_config.options = 0;
	res = cudppPlan(theCudpp, &reduplan_float, redu_float_config, N, 1, 0);
	if (CUDPP_SUCCESS != res){printf("Error creating CUDPPPlan for reduction\n");exit(-1);}
	cudaThreadSynchronize();
	
	printf("\e[0;32m%-6s\e[m \n","  Configuring hashes...");
	// hash config stuff
	hash_config.type = CUDPP_BASIC_HASH_TABLE;
	hash_config.kInputSize = all_geom.all_total;
	hash_config.space_usage = 1.2f;

	//material hash stuff
	res = cudppHashTable(theCudpp, &mate_hash_table_handle, &hash_config);
	if (res != CUDPP_SUCCESS){fprintf(stderr, "Error in cudppHashTable call (make sure your device is at least compute version 2.0\n");exit(-1);}
	printf("\e[0;32m%-6s\e[m \n","  Inserting values into cellnum/matnum hash table...");
	res = cudppHashInsert(mate_hash_table_handle, d_hash_key, d_hash_val_mate, hash_config.kInputSize);
	if (res != CUDPP_SUCCESS){fprintf(stderr, "Error in inserting values into hash table\n");exit(-1);}
	cudaThreadSynchronize();

	// fissile hash stuff
	res = cudppHashTable(theCudpp, &fiss_hash_table_handle, &hash_config);
	if (res != CUDPP_SUCCESS){fprintf(stderr, "Error in cudppHashTable call (make sure your device is at least compute version 2.0\n");exit(-1);}
	printf("\e[0;32m%-6s\e[m \n","  Inserting values into cellnum/fissile hash table...");
	res = cudppHashInsert(fiss_hash_table_handle, d_hash_key, d_hash_val_fiss, hash_config.kInputSize);
	if (res != CUDPP_SUCCESS){fprintf(stderr, "Error in inserting values into hash table\n");exit(-1);}
	cudaThreadSynchronize();

	/////////////////////
	// INIT CURAND RNG //
	/////////////////////

	// THIS MUST COME AFTER THE PARTICLE DATASET IS ALLOCATED
	printf("\e[0;32m%-6s\e[m \n","  Generating random number bank on device using MTGP32...");
	curandGenerator_t rand_gen ;
	curandCreateGenerator( &rand_gen , CURAND_RNG_PSEUDO_MTGP32 );  //mersenne twister type
	curandSetPseudoRandomGeneratorSeed( rand_gen , 1234ULL );
	curandGenerateUniform( rand_gen , d_hist.rn_bank , N*RNUM_PER_THREAD );

	//////////////////////////////
	// COPY HISTORIES TO DEVICE //
	//////////////////////////////

	copy_hist(N,&hist,&d_hist,d_hist_ptr);
*/

}

