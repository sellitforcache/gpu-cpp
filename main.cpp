#include "warp.h"

int main(){

	using namespace std;

	///////////////////
	// BEGIN PROGRAM //
	///////////////////

	//get inputs?
	int N=1e5;

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

	//////////////////////
	// INIT OptiX STUFF //
	//////////////////////
	
	optix_stuff this_optix ( N , 4 );
	this_optix.init(geom);
	this_optix.print();

	/////////////////////////////////////////////////////////////////
	// INIT CUDA and HISTORY STUFF and LOAD/UNIONIZE CROS SECTIONS //    !! MUST BE DONE AFTER OPTIX !!
	/////////////////////////////////////////////////////////////////

	whistory hist ( N , this_optix );
	hist.init_host();
	//hist.init_RNG();
	//hist.init_CUDPP();
	hist.load_cross_sections("92235");
	hist.print_xs_data();
	hist.copy_to_device();
	hist.print_pointers();
	hist.write_xs_data("xsdata");

	// trace geom
	this_optix.check_errors();

	try {
		this_optix.trace_geometry(1024,1024);	
	} 
	catch( optix::Exception &e ){
		std::cout << e.getErrorString().c_str();
		exit(1);
	}


}

