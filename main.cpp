#include "warp.h"

int main(){

	using namespace std;

	///////////////////
	// BEGIN PROGRAM //
	///////////////////

	std::string tallyname = "tally";

	//print banner
	print_banner();

	// set up geometry
	wgeometry geom;

	// UO2 w boron materials
	//unsigned topes[5]={92235,92238,8016,1001,5010};
	//float    fracs_fuel[5]={0.01,0.99,2,0,0};
	//float    fracs_water[5]={0,0,1,2,0.2};
	//float    dens_fuel = 10.424;
	//float 	 dens_water = 1.0;
	//geom.add_material(1,1,5,dens_fuel,topes,fracs_fuel);
	//geom.add_material(2,0,5,dens_water,topes,fracs_water);
	//geom.add_material(3,0,5,dens_water,topes,fracs_water);

	// homogenized UO2 w boron materials
	//unsigned topes[4]={92235,92238,8016,1001};
	//float    fracs[4]={0.01,  0.99,   3,   2};
	//float 	 dens = 3.0;
	//geom.add_material(1,1,4,dens,topes,fracs);
	//geom.add_material(2,0,4,dens,topes,fracs);
	//geom.add_material(3,0,4,dens,topes,fracs);
	//tallyname = "homfuel.tally";

	// water
	//unsigned topes[2]={8016,1001};
	//float    fracs[2]={1,2};
	//float    dens = 3.00;
	//geom.add_material(1,0,2,dens,topes,fracs);
	//geom.add_material(2,0,2,dens,topes,fracs);
	//geom.add_material(3,0,2,dens,topes,fracs);
	//tallyname = "water.tally";

	// aluminum
	//unsigned topes[1]={13027};
	//float    fracs[1]={1};
	//float    dens = 1.00;
	//geom.add_material(1,1,1,dens,topes,fracs);
	//geom.add_material(2,0,1,dens,topes,fracs);
	//geom.add_material(3,0,1,dens,topes,fracs);
	//tallyname = "aluminum.tally";

	// o16
	//unsigned topes[1]={8016};
	//float    fracs[1]={1};
	//float    dens = 1.00;
	//geom.add_material(1,1,1,dens,topes,fracs);
	//geom.add_material(2,0,1,dens,topes,fracs);
	//geom.add_material(3,0,1,dens,topes,fracs);
	//tallyname = "o16.tally";

	// lead
	//unsigned topes[1]={82208};
	//float    fracs[1]={1};
	//float    dens = 1.00;//11.34; fuck you density
	//geom.add_material(1,1,1,dens,topes,fracs);
	//geom.add_material(2,0,1,dens,topes,fracs);
	//geom.add_material(3,0,1,dens,topes,fracs);
	//tallyname = "lead.tally";

	// carbon
	//unsigned topes[1]={6000};
	//float    fracs[1]={1};
	//float    dens = 1.00;
	//geom.add_material(1,1,1,dens,topes,fracs);
	//geom.add_material(2,0,1,dens,topes,fracs);
	//geom.add_material(3,0,1,dens,topes,fracs);
	//tallyname = "carbon.tally";

	// lithium
	//unsigned topes[1]={3006};
	//float    fracs[1]={1};
	//float    dens = 1.00;
	//geom.add_material(1,1,1,dens,topes,fracs);
	//geom.add_material(2,0,1,dens,topes,fracs);
	//geom.add_material(3,0,1,dens,topes,fracs);
	//tallyname = "lithium.tally";

	// u235
	//unsigned topes[1]={92235};
	//float    fracs[1]={1};
	//float    dens = 3.0;
	//geom.add_material(1,1,1,dens,topes,fracs);
	//geom.add_material(2,0,1,dens,topes,fracs);
	//geom.add_material(3,0,1,dens,topes,fracs);
	//tallyname = "u235_crit.tally";

	// pu239
	//unsigned topes[1]={94239};
	//float    fracs[1]={1};
	//float    dens = 1;
	//geom.add_material(1,1,1,dens,topes,fracs);
	//geom.add_material(2,0,1,dens,topes,fracs);
	//geom.add_material(3,0,1,dens,topes,fracs);
	//tallyname = "1kpu239.tally";

	// u238
	//unsigned topes[1]={92238};
	//float    fracs[1]={1};
	//float    dens = 3;
	//geom.add_material(1,1,1,dens,topes,fracs);
	//geom.add_material(2,0,1,dens,topes,fracs);
	//geom.add_material(3,0,1,dens,topes,fracs);
	//tallyname = "u238_crit.tally";

	// hydrogen2
	//unsigned topes[1]={1002};
	//float    fracs[1]={1};
	//float    dens = 1.0;
	//geom.add_material(1,1,1,dens,topes,fracs);
	//geom.add_material(2,0,1,dens,topes,fracs);
	//geom.add_material(3,0,1,dens,topes,fracs);
	//tallyname = "hydrogen2.tally";

	// hydrogen1
	//unsigned topes[1]={1001};
	//float    fracs[1]={1};
	//float    dens = 1.0;
	//geom.add_material(1,1,1,dens,topes,fracs);
	//geom.add_material(2,0,1,dens,topes,fracs);
	//geom.add_material(3,0,1,dens,topes,fracs);
	//tallyname = "hydrogen1.tally";

	// assembly mats
	unsigned topes      [2] = {92235,92238};//,8016,1001};
	float    fracs_fuel [2] = {1,  0};//,   0,   0};
	float 	 fracs_water[2] = {   0,     1};//,   1,   2};
	float    dens_fuel = 5;
	float 	 dens_water = 3;
	geom.add_material(1,1,2,dens_fuel, topes,fracs_fuel);
	geom.add_material(2,0,2,dens_water,topes,fracs_water);
	geom.add_material(3,0,2,dens_water,topes,fracs_water);
	tallyname = "assembly.tally";

	// assembly geom
	geom.add_primitive();
	geom.primitives[0].type=1;
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

	//simple geom
	//geom.add_primitive();
	//geom.primitives[0].type=0;
	//geom.primitives[0].material=1;
	//geom.primitives[0].min[0]=-500;
	//geom.primitives[0].min[1]=-500;
	//geom.primitives[0].min[2]=-500;
	//geom.primitives[0].max[0]= 500;
	//geom.primitives[0].max[1]= 500;
	//geom.primitives[0].max[2]= 500;
	//geom.primitives[0].add_transform();
	//geom.primitives[0].transforms[0].cellnum = 999;
	//geom.primitives[0].transforms[0].dx      = 0;
	//geom.primitives[0].transforms[0].dy      = 0;
	//geom.primitives[0].transforms[0].dz      = 0;
	//geom.primitives[0].transforms[0].theta   = 0;
	//geom.primitives[0].transforms[0].phi     = 0;

	// finalize geom
	geom.set_outer_cell(999);
	geom.update();
	if(geom.check()){std::cout << "geometry failed check!\n"; return 1;}
	//geom.print_all();
	geom.print_summary();

	///////////////////////////////////
	// INIT OptiX STUFF for plotting //
	///////////////////////////////////

	// trace geom if requested
	// make new context that fits the reqested image size, trace, then destroy to free resources
	//unsigned geom_width  = 1024; 
	//unsigned geom_height = 1024;
	//unsigned N_geom = geom_width*geom_height;
	//optix_stuff geom_optix ( N_geom , 4 );
	//geom_optix.init(geom);
	//geom_optix.trace_geometry(geom_width,geom_height,"geom.png");
	//geom_optix.~optix_stuff();


	/////////////////////////////////////////////////////////////////
	// INIT CUDA and HISTORY STUFF and LOAD/UNIONIZE CROS SECTIONS //
	/////////////////////////////////////////////////////////////////

	int N = 1e5;
	whistory hist ( N , geom );
	hist.set_device(0);
	hist.init();
	hist.print_xs_data();
	hist.print_materials_table();

	/////////////////////////////////////
	// converge fission source and run //
	/////////////////////////////////////

	hist.set_run_type("criticality");
	//hist.converge(10);
	hist.set_tally_cell(999);
	hist.set_run_param(1,1);  //run, skip
	hist.run();
	hist.write_tally(0,tallyname);
	hist.write_xs_data("xsdata");


	return 0;

}

