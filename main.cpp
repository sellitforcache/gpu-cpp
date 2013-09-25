#include "warp.h"

int main(){

	using namespace std;

	///////////////////
	// BEGIN PROGRAM //
	///////////////////

	//get inputs?
	int N=1e4;

	//print banner
	print_banner();

	// set up geometry
	wgeometry geom;

	geom.add_primitive();
	geom.primitives[0].type=2;
	geom.primitives[0].material=1;
	geom.primitives[0].min[0]=-20.0;
	geom.primitives[0].min[1]=-20.0;
	geom.primitives[0].min[2]=-20.0;
	geom.primitives[0].max[0]= 20.0;
	geom.primitives[0].max[1]= 1.0;
	geom.primitives[0].max[2]= 10.0;
	geom.primitives[0].make_hex_array(15,0.0,0.0,3.14159/2,0);
	geom.add_primitive();
	geom.primitives[1].type=2;
	geom.primitives[1].material=2;
	geom.primitives[1].min[0]=-21;
	geom.primitives[1].max[0]=21;
	geom.primitives[1].max[1]=31.0;
	geom.primitives[1].add_transform();
	geom.primitives[1].transforms[0].cellnum = 800;
	geom.primitives[1].transforms[0].dx      = 0;
	geom.primitives[1].transforms[0].dy      = 0;
	geom.primitives[1].transforms[0].dz      = 0;
	geom.primitives[1].transforms[0].theta   = 0;
	geom.primitives[1].transforms[0].phi     = 0;
	geom.add_primitive();
	geom.primitives[2].type=0;
	geom.primitives[2].material=3;
	geom.primitives[2].min[0]=-42;
	geom.primitives[2].min[1]=-42;
	geom.primitives[2].min[2]=-42;
	geom.primitives[2].max[0]=42;
	geom.primitives[2].max[1]=42;
	geom.primitives[2].max[2]=42;
	geom.primitives[2].add_transform();
	geom.primitives[2].transforms[0].cellnum = 999;
	geom.primitives[2].transforms[0].dx      = 0;
	geom.primitives[2].transforms[0].dy      = 0;
	geom.primitives[2].transforms[0].dz      = 0;
	geom.primitives[2].transforms[0].theta   = 0;
	geom.primitives[2].transforms[0].phi     = 0;
	geom.set_outer_cell(999);
	geom.update();
	//geom.print_all();
	geom.print_summary();

	//////////////////////
	// INIT OptiX STUFF //
	//////////////////////
	
	// tracer for histories
	optix_stuff this_optix ( N , 4 );
	this_optix.init(geom);
	this_optix.print();

	// trace geom if requested
	// make new context that fits the reqested image size, trace, then destroy to free resources
	unsigned geom_width  = 1024; 
	unsigned geom_height = 1024;
	unsigned N_geom = geom_width*geom_height;
	optix_stuff geom_optix ( N_geom , 4 );
	geom_optix.init(geom);
	geom_optix.trace_geometry(geom_width,geom_height,"geom.png");
	geom_optix.~optix_stuff();


	/////////////////////////////////////////////////////////////////
	// INIT CUDA and HISTORY STUFF and LOAD/UNIONIZE CROS SECTIONS //    !! MUST BE DONE AFTER OPTIX !!
	/////////////////////////////////////////////////////////////////

	whistory hist ( N , this_optix );
	hist.init_host();
	hist.init_RNG();
	hist.init_CUDPP();
	hist.load_cross_sections("92235");
	hist.print_xs_data();
	hist.copy_to_device();
	hist.print_pointers();
	//hist.write_xs_data("xsdata");


}

