#include "warp.h"
#include "optix_stuff.h"

int main(int argc, char** argv){

	////////////////////////
	// OPTIX TEST PROGRAM //
	////////////////////////

	std::string tallyname = "tally";

	// set up geometry
	wgeometry geom;

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


	//simple geom
	geom.add_primitive();
	geom.primitives[0].type=0;
	geom.primitives[0].material=1;
	geom.primitives[0].min[0]=-42.0;
	geom.primitives[0].min[1]=-42.0;
	geom.primitives[0].min[2]=-42.0;
	geom.primitives[0].max[0]= 42.0;
	geom.primitives[0].max[1]= 42.0;
	geom.primitives[0].max[2]= 42.0;
	geom.primitives[0].add_transform();
	geom.primitives[0].transforms[0].cellnum = 999;
	geom.primitives[0].transforms[0].dx      = 0;
	geom.primitives[0].transforms[0].dy      = 0;
	geom.primitives[0].transforms[0].dz      = 0;
	geom.primitives[0].transforms[0].theta   = 0;
	geom.primitives[0].transforms[0].phi     = 0;
	geom.set_outer_cell(999);
	tallyname = "simple";

	// assembly geom
	//geom.add_primitive();
	//geom.primitives[0].type=1;
	//geom.primitives[0].material=1;
	//geom.primitives[0].min[0]=-20.0;
	//geom.primitives[0].min[1]=-9990.0;
	//geom.primitives[0].min[2]=-9990.0;
	//geom.primitives[0].max[0]= 20.0;
	//geom.primitives[0].max[1]= 1.0;
	//geom.primitives[0].max[2]= -10.0;
	//geom.primitives[0].make_hex_array(15,0.0,0.0,3.14159/2,0);  
	//geom.add_primitive();
	//geom.primitives[1].type=2;
	//geom.primitives[1].material=2;
	//geom.primitives[1].min[0]=-21;
	//geom.primitives[1].max[0]=21;
	//geom.primitives[1].max[1]=30.0;
	//geom.primitives[1].add_transform();
	//geom.primitives[1].transforms[0].cellnum = 800;
	//geom.primitives[1].transforms[0].dx      = 0;
	//geom.primitives[1].transforms[0].dy      = 0;
	//geom.primitives[1].transforms[0].dz      = 0;
	//geom.primitives[1].transforms[0].theta   = 0;
	//geom.primitives[1].transforms[0].phi     = 0;
	//geom.add_primitive();
	//geom.primitives[2].type=0;
	//geom.primitives[2].material=2;
	//geom.primitives[2].min[0]=-42;
	//geom.primitives[2].min[1]=-42;
	//geom.primitives[2].min[2]=-42;
	//geom.primitives[2].max[0]=42;
	//geom.primitives[2].max[1]=42;
	//geom.primitives[2].max[2]=42;
	//geom.primitives[2].add_transform();
	//geom.primitives[2].transforms[0].cellnum = 999;
	//geom.primitives[2].transforms[0].dx      = 0;
	//geom.primitives[2].transforms[0].dy      = 0;
	//geom.primitives[2].transforms[0].dz      = 0;
	//geom.primitives[2].transforms[0].theta   = 0;
	//geom.primitives[2].transforms[0].phi     = 0;
	//geom.set_outer_cell(999);
	//tallyname = "assembly";


	// big assembly geom
	//geom.add_primitive();
	//geom.primitives[0].type=1;
	//geom.primitives[0].material=1;
	//geom.primitives[0].min[0]=-20.0;
	//geom.primitives[0].min[1]=-9990.0;
	//geom.primitives[0].min[2]=-9990.0;
	//geom.primitives[0].max[0]= 20.0;
	//geom.primitives[0].max[1]= 1.0;
	//geom.primitives[0].max[2]= -10.0;
	//geom.primitives[0].make_hex_array(25,0.0,0.0,3.14159/2,0);  
	//geom.add_primitive();
	//geom.primitives[1].type=2;
	//geom.primitives[1].material=2;
	//geom.primitives[1].min[0]=-21;
	//geom.primitives[1].max[0]=21;
	//geom.primitives[1].max[1]=50.0;
	//geom.primitives[1].add_transform();
	//geom.primitives[1].transforms[0].cellnum = 1900;
	//geom.primitives[1].transforms[0].dx      = 0;
	//geom.primitives[1].transforms[0].dy      = 0;
	//geom.primitives[1].transforms[0].dz      = 0;
	//geom.primitives[1].transforms[0].theta   = 0;
	//geom.primitives[1].transforms[0].phi     = 0;
	//geom.add_primitive();
	//geom.primitives[2].type=0;
	//geom.primitives[2].material=2;
	//geom.primitives[2].min[0]=-65;
	//geom.primitives[2].min[1]=-65;
	//geom.primitives[2].min[2]=-65;
	//geom.primitives[2].max[0]=65;
	//geom.primitives[2].max[1]=65;
	//geom.primitives[2].max[2]=65;
	//geom.primitives[2].add_transform();
	//geom.primitives[2].transforms[0].cellnum = 2001;
	//geom.primitives[2].transforms[0].dx      = 0;
	//geom.primitives[2].transforms[0].dy      = 0;
	//geom.primitives[2].transforms[0].dz      = 0;
	//geom.primitives[2].transforms[0].theta   = 0;
	//geom.primitives[2].transforms[0].phi     = 0;
	//geom.set_outer_cell(2001);
	//tallyname = "big_assembly";

	// interleaved geom
	//geom.add_primitive();
	//geom.primitives[0].type=1;
	//geom.primitives[0].material=1;
	//geom.primitives[0].min[0]=-20.0;
	//geom.primitives[0].min[1]=-9990.0;
	//geom.primitives[0].min[2]=-9990.0;
	//geom.primitives[0].max[0]= 20.0;
	//geom.primitives[0].max[1]= 1.0;
	//geom.primitives[0].max[2]= -10.0;
	//geom.primitives[0].make_hex_array(7,0.0,0.0,2.328,0.0,0);  
	//geom.add_primitive();
	//geom.primitives[1].type=2;
	//geom.primitives[1].material=1;
	//geom.primitives[1].min[0]=-20.0;
	//geom.primitives[1].min[1]=-9990.0;
	//geom.primitives[1].min[2]=-9990.0;
	//geom.primitives[1].max[0]= 20.0;
	//geom.primitives[1].max[1]= 1.0;
	//geom.primitives[1].max[2]= -10.0;
	//geom.primitives[1].make_hex_array(7,2.328,1.35,2.328,0.0,3*7*(7-1)+2);
	//geom.add_primitive();
	//geom.primitives[2].type=1;
	//geom.primitives[2].material=1;
	//geom.primitives[2].min[0]=-20.0;
	//geom.primitives[2].min[1]=-9990.0;
	//geom.primitives[2].min[2]=-9990.0;
	//geom.primitives[2].max[0]= 20.0;
	//geom.primitives[2].max[1]= 1.0;
	//geom.primitives[2].max[2]= -10.0;
	//geom.primitives[2].make_hex_array(7,2.328,-1.35,2.328,0.0,2*(3*7*(7-1)+1)+1);  
	//geom.add_primitive();
	//geom.primitives[3].type=2;
	//geom.primitives[3].material=2;
	//geom.primitives[3].min[0]=-21;
	//geom.primitives[3].max[0]=21;
	//geom.primitives[3].max[1]=30.0;
	//geom.primitives[3].add_transform();
	//geom.primitives[3].transforms[0].cellnum = 500;
	//geom.primitives[3].transforms[0].dx      = 1.65;
	//geom.primitives[3].transforms[0].dy      = 0;
	//geom.primitives[3].transforms[0].dz      = 0;
	//geom.primitives[3].transforms[0].theta   = 0;
	//geom.primitives[3].transforms[0].phi     = 0;
	//geom.add_primitive();
	//geom.primitives[4].type=0;
	//geom.primitives[4].material=2;
	//geom.primitives[4].min[0]=-42;
	//geom.primitives[4].min[1]=-42;
	//geom.primitives[4].min[2]=-42;
	//geom.primitives[4].max[0]=42;
	//geom.primitives[4].max[1]=42;
	//geom.primitives[4].max[2]=42;
	//geom.primitives[4].add_transform();
	//geom.primitives[4].transforms[0].cellnum = 505;
	//geom.primitives[4].transforms[0].dx      = 0;
	//geom.primitives[4].transforms[0].dy      = 0;
	//geom.primitives[4].transforms[0].dz      = 0;
	//geom.primitives[4].transforms[0].theta   = 0;
	//geom.primitives[4].transforms[0].phi     = 0;
	//geom.set_outer_cell(505);
	//tallyname = "interleaved";


	// finalize geom
	geom.update();
	if(geom.check()){std::cout << "geometry failed check!\n"; return 1;}
	geom.print_summary();

	///////////////////////////////////
	// INIT OptiX STUFF for plotting //
	///////////////////////////////////

	// trace geom, return time
	unsigned N;
	std::string accel_type_in;
	if (argc==2){ 
		N = atoi(argv[1]);
		accel_type_in = "Sbvh";
	}
	else if(argc==3){
		N = atoi(argv[1]);
		accel_type_in.insert(0,argv[2]);
	}
	else{
		printf("please specify a number of rays\n"); 
		return 1;
	}
	optix_stuff optix_test ( N , 4 );
	optix_test.init(geom,0,accel_type_in);
	optix_test.print();
	double tracetime = optix_test.trace_test();
	optix_test.~optix_stuff();

	printf("trace time = %10.8E seconds\n",tracetime);
	std::cout.flush();
	fflush(stdout);

	FILE* resultsfile = fopen("optix_log","a");
	fprintf(resultsfile,"%s %s %u %10.8E\n",tallyname.c_str(),accel_type_in.c_str(),N,tracetime);
	fclose(resultsfile);

	return 0;

}

