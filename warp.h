#include <vector> 
#include <iostream>
#include <cmath>
#include "datadef.h"

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//							Prototype declaration stuff
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

void print_banner();


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//							Geometry stuff
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


class primative
{	
	public:
	 primative();
	 primative(int,int,int,float,float,float,float,float,float,float,float,float);
	~primative();
	void add_transform();
	void add_transform(int,float,float,float,float,float);
	void print_transform();
	void print_transform(int);
	void make_hex_array(int,float,float,int);
	float       min[3];
	float       max[3];
	float       location[3];
	static int  num_primatives;
	int			type;      // 0=box, 1=cyl, 2=hex
	int 		primative_id;
	int         n_transforms;
	int         material;
	std::vector<transform>   transforms;
};

int primative::num_primatives=0;

primative::primative(){
	//box default constructor
	min[0]=0;min[1]=0;min[2]=0;
	max[0]=0;max[1]=0;max[2]=0;
	location[0]=0;location[1]=0;location[2]=0;
	type=0;
	material=0;
	transform this_transform;
	this_transform.cellnum = 0;
	this_transform.cellmat = 0;
	this_transform.dx      = 0;
	this_transform.dy      = 0;
	this_transform.dz      = 0;
	this_transform.theta   = 0;
	this_transform.phi     = 0;
	transforms.push_back(this_transform);
	n_transforms=1;
	primative_id=num_primatives;
	num_primatives++;
}
primative::primative(int ptype, int cellnum ,int cellmat ,float xmin,float ymin,float zmin,float xmax,float ymax,float zmax,float x,float y,float z){
	//box valued constructor
	min[0]=xmin;min[1]=ymin;min[2]=zmin;
	max[0]=xmax;max[1]=ymax;max[2]=zmax;
	location[0]=x;location[1]=y;location[2]=z;
	type=ptype;
	material=cellmat;
	transform this_transform;
	this_transform.cellnum = cellnum;
	this_transform.cellmat = cellmat;
	this_transform.dx      = 0;
	this_transform.dy      = 0;
	this_transform.dz      = 0;
	this_transform.theta   = 0;
	this_transform.phi     = 0;
	transforms.push_back(this_transform);
	n_transforms=1;
	primative_id=num_primatives;
	num_primatives++;
}
primative::~primative(){
	//box destructor
	//delete min;
	//delete max;
	//delete location;
	//delete &box_id;
	//delete &n_transforms;
	//delete   &transforms;
}
void primative::add_transform(){
	transform this_transform;
	this_transform.cellnum = 0;
	this_transform.cellmat = material;
	this_transform.dx      = 0;
	this_transform.dy      = 0;
	this_transform.dz      = 0;
	this_transform.theta   = 0;
	this_transform.phi     = 0;
	transforms.push_back(this_transform);
	n_transforms++;
}
void primative::add_transform(int cellnum , float dx , float dy , float dz , float theta , float phi ){
	transform this_transform;
	this_transform.cellnum = cellnum;
	this_transform.cellmat = material;
	this_transform.dx      = dx;
	this_transform.dy      = dy;
	this_transform.dz      = dz;
	this_transform.theta   = theta;
	this_transform.phi     = phi;
	transforms.push_back(this_transform);
	n_transforms++;
}
void primative::print_transform(){
	std::cout << "--- primative id  = " << primative_id << " ---" << "\n";
	std::cout << "   min,max  = (" << min[0] << min[1] << min[2] << "),(" << max[0] << max[1] << max[2] << ")" << "\n";
	std::cout << "   location = (" << location[0] << location[1] << location[2] << ")" << "\n";
	std::cout << "   type =  " << type << "\n";
	std::cout << "   material = " << material << "\n";
	for (int tnum=0;tnum<n_transforms;tnum++){
		std::cout << "   ************ " << "\n";
		std::cout << "   transform= " << tnum << "\n";
		std::cout << "   cellnum  = " << transforms[tnum].cellnum << "\n";
		std::cout << "   cellmat  = " << transforms[tnum].cellmat << "\n";
		std::cout << "   dx       = " << transforms[tnum].dx << "\n";
		std::cout << "   dy       = " << transforms[tnum].dy << "\n";
		std::cout << "   dz       = " << transforms[tnum].dz << "\n";
		std::cout << "   theta    = " << transforms[tnum].theta << "\n";
		std::cout << "   phi      = " << transforms[tnum].phi << "\n";
	}
}
void primative::print_transform(int tnum){
	std::cout << "--- primative id  = " << primative_id << " ---" << "\n";
	std::cout << "   min,max  = (" << min[0] << min[1] << min[2] << "),(" << max[0] << max[1] << max[2] << ")" << "\n";
	std::cout << "   location = (" << location[0] << location[1] << location[2] << ")" << "\n";
	std::cout << "   type =  " << type << "\n";
	std::cout << "   material = " << material << "\n";
	std::cout << "   ************ " << "\n";
	std::cout << "   transform= " << tnum << "\n";
	std::cout << "   cellnum  = " << transforms[tnum].cellnum << "\n";
	std::cout << "   cellmat  = " << transforms[tnum].cellmat << "\n";
	std::cout << "   dx       = " << transforms[tnum].dx << "\n";
	std::cout << "   dy       = " << transforms[tnum].dy << "\n";
	std::cout << "   dz       = " << transforms[tnum].dz << "\n";
	std::cout << "   theta    = " << transforms[tnum].theta << "\n";
	std::cout << "   phi      = " << transforms[tnum].phi << "\n";
}

void primative::make_hex_array(int n, float x, float y, int starting_index){

	int k, j, num, cnt;
	float offsetx, offsety, fnum, lattr, PD_ratio;

	// add num of transforms to whatever is there
    n_transforms += 3*n*(n-1)+1;

    transform this_transform;
	num=n;
	cnt=0;
	PD_ratio=1.164;
	lattr = PD_ratio * (2.0 * max[1]) / sqrt(3.0);
	offsety=(n-1)*lattr*1.5;

	//row
	for (k=0;k<(2*n-1);k++){
		fnum=num-1;
		offsetx=-(sqrt(3.0)*lattr)*(fnum/2.0);
		//column
		for(j=0;j<num;j++){

    		this_transform.cellnum=starting_index+cnt;
    		this_transform.cellmat=material;
    		this_transform.dx=offsetx;
    		this_transform.dy=offsety;
    		this_transform.dz=0;
    		this_transform.theta=0;
    		this_transform.phi=0;
    		transforms.push_back(this_transform);
    		cnt++;
    		offsetx+=sqrt(3.0)*lattr;
		}

		if ( k < n-1 ){
			num++;   }
		else{
		 	num--;   }

		offsety-=lattr*1.5;

	}

}


class geometry {
	int n_box;
	int n_cyl;
	int n_hex;
	int n_primatives;
	int n_transforms;
public:
	 geometry();
	~geometry();
	void add_primative();
	void update();
	std::vector<primative>   primatives;
};

geometry::geometry(){
	n_box        = 0;
	n_cyl        = 0;
	n_hex        = 0;
	n_primatives = 0;
	n_transforms = 0;
}
geometry::~geometry(){
	
}
void geometry::add_primative(){
	primative this_primative;
	primatives.push_back(this_primative);
	n_primatives++;
}
void geometry::update(){
	n_box        = 0;
	n_cyl        = 0;
	n_hex        = 0;
	n_transforms = 0;
	for(int k=0;k<n_primatives;k++){
		switch (primatives[k].type){
			case 0:
				n_box+=primatives[k].n_transforms;
			case 1:
				n_cyl+=primatives[k].n_transforms;
			case 2:
				n_hex+=primatives[k].n_transforms;
		}
		n_transforms+=primatives[k].n_transforms;
	}
}


/*
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//							OptiX stuff
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


class optix_stuff{
	void * positions_ptr; 
	void * rxn_ptr; 
	void * done_ptr;
	void * cellnum_ptr;
public:
	int        stack_size_multiplier;
	RTcontext  context;
	RTvariable trace_type_var;
	void init();
	void read_geometry();
};

void optix_stuff::init(int N){

	// local variables
	char                path_to_ptx[512];
	RTprogram           ray_gen_program;
	RTprogram           exception_program;  
	RTprogram           miss_program;
	RTgroup             top_level_group;
	RTvariable          top_object;
	RTacceleration      top_level_acceleration;
	RTbuffer            particles_obj;
	RTbuffer 			rxn_obj;
	RTbuffer 			done_obj;
	RTbuffer 			cellnum_obj;
	RTvariable          particles;
	RTvariable 			rxn_buffer;
	RTvariable 			done_buffer;
	RTvariable 			cellnum_buffer;
	RTvariable          outer_cell_var;
	RTvariable 			boundary_condition_var;
	RTsize              stack_size;
	
	// create OptiX context
	rtContextCreate( context );
	rtContextSetPrintEnabled(*context, 1);
	rtContextSetExceptionEnabled(*context, RT_EXCEPTION_ALL, 1);
	rtContextSetRayTypeCount( *context, 1 ); 
	rtContextSetEntryPointCount( *context, 1 );
	rtContextGetStackSize(*context, &stack_size);
	stack_size = stack_size_multiplier*stack_size;
	rtContextSetStackSize(*context, stack_size);
	printf("OptiX stack size is %d bytes\n",(unsigned) stack_size);
	
	// Render particle buffer and attach to variable, get pointer for CUDA
	rtContextDeclareVariable( *context, "particles", &particles );
	rtBufferCreate( *context, RT_BUFFER_INPUT_OUTPUT, &particles_obj );
	rtBufferSetFormat( particles_obj, RT_FORMAT_USER );
	rtBufferSetElementSize( particles_obj, sizeof(source_point) );
	rtBufferSetSize1D( particles_obj, N );
	rtVariableSetObject( particles, particles_obj );
	rtBufferGetDevicePointer(particles_obj,0,&positions_ptr);

	// Render reaction buffer and attach to variable, get pointer for CUDA
	rtContextDeclareVariable( *context, "rxn_buffer", &rxn_buffer );
	rtBufferCreate( *context, RT_BUFFER_INPUT_OUTPUT, &rxn_obj );
	rtBufferSetFormat( rxn_obj, RT_FORMAT_USER );
	rtBufferSetElementSize( rxn_obj, sizeof(unsigned) );
	rtBufferSetSize1D( rxn_obj, N );
	rtVariableSetObject( rxn_buffer, rxn_obj );
	rtBufferGetDevicePointer(rxn_obj,0,&rxn_ptr);

	// Render done buffer and attach to variable, get pointer for CUDA
	rtContextDeclareVariable( *context, "done_buffer", &done_buffer );
	rtBufferCreate( *context, RT_BUFFER_INPUT_OUTPUT, &done_obj );
	rtBufferSetFormat( done_obj, RT_FORMAT_USER );
	rtBufferSetElementSize( done_obj, sizeof(unsigned) );
	rtBufferSetSize1D( done_obj, N );
	rtVariableSetObject( done_buffer, done_obj );
	rtBufferGetDevicePointer(done_obj,0,&done_ptr);

	// Render done buffer and attach to variable, get pointer for CUDA
	rtContextDeclareVariable( *context, "cellnum_buffer", &cellnum_buffer );
	rtBufferCreate( *context, RT_BUFFER_INPUT_OUTPUT, &cellnum_obj );
	rtBufferSetFormat( cellnum_obj, RT_FORMAT_USER );
	rtBufferSetElementSize( cellnum_obj, sizeof(unsigned) );
	rtBufferSetSize1D( cellnum_obj, N );
	rtVariableSetObject( cellnum_buffer, cellnum_obj );
	rtBufferGetDevicePointer(cellnum_obj,0,&cellnum_ptr);
	
	// Ray generation program 
	sprintf( path_to_ptx, "%s", "camera.ptx" );
	rtProgramCreateFromPTXFile( *context, path_to_ptx, "camera", &ray_gen_program );
	rtContextSetRayGenerationProgram( *context, 0, ray_gen_program  );
	
	// Exception program
	rtProgramCreateFromPTXFile( *context, path_to_ptx, "exception", &exception_program ) ;
	rtContextSetExceptionProgram( *context, 0, exception_program  );    
	
	// Miss program 
	sprintf( path_to_ptx, "%s", "miss.ptx" );
	rtProgramCreateFromPTXFile( *context, path_to_ptx, "miss", &miss_program  );
	rtContextSetMissProgram( *context, 0, miss_program  );
	
	//set outer cell number as (basically) global
	rtContextDeclareVariable(*context,"outer_cell",&outer_cell_var);
	rtVariableSet1i(outer_cell_var,all_geom[0].outer_cell);

	//set boundary condition for outer cell
	rtContextDeclareVariable(*context,"boundary_condition",&boundary_condition_var);
	rtVariableSet1i(boundary_condition_var,all_geom[0].boundary_condition);

	//set trace type, 1=transport (writes intersection point and next cell), 2=fission (writes origin and current cell)
	rtContextDeclareVariable(*context,"trace_type",trace_type_var);
	rtVariableSet1i(*trace_type_var,1);
	
	// Make top level group/accel as children of the top level object 
	rtGroupCreate( *context, &top_level_group );
	rtGroupSetChildCount( top_level_group, all_geom[0].all_total );
	rtContextDeclareVariable( *context, "top_object", &top_object );
	rtVariableSetObject( top_object, top_level_group );
	rtAccelerationCreate( *context, &top_level_acceleration );
	rtAccelerationSetBuilder(top_level_acceleration,"Sbvh");
	rtAccelerationSetTraverser(top_level_acceleration,"Bvh");
	rtGroupSetAcceleration( top_level_group, top_level_acceleration);

}

void optix_stuff::read_geometry(){

	// make all geometry instances
	make_geom(all_geom, context, &top_level_group); 
	
	// mark acceleration as dirty
	RTresult result = rtAccelerationMarkDirty( top_level_acceleration );
	
	//printf("CUDA ERROR, %s\n",cudaGetErrorString(cudaPeekAtLastError()));
	//const char * oerr;
	//rtContextGetErrorString(*context,result,&oerr);
	//printf("OPTIX ERROR, %s\n",oerr);

	//validate and compile
	result = rtContextValidate( *context );
	result = rtContextCompile( *context );
}



/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//							History stuff
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


//history struct
typedef history { 
    source_point *  space;
    float *         E;
    float *         Q;
    float *         rn_bank;
    unsigned *      cellnum;
    unsigned *      matnum;
    unsigned *      isonum;
    unsigned *      rxn;
    unsigned *      done;
    unsigned *      yield;
};

*/

