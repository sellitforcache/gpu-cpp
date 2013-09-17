#include <vector> 
#include <iostream>
#include <cmath>
#include <optixu/optixpp_namespace.h>
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


class primitive
{	
	public:
	 primitive();
	 primitive(int,int,int,float,float,float,float,float,float,float,float,float);
	~primitive();
	void add_transform();
	void add_transform(int,float,float,float,float,float);
	void print_transform();
	void print_transform(int);
	void make_hex_array(int,float,float,int);
	float       min[3];
	float       max[3];
	float       location[3];
	static int  num_primitives;
	int			type;      // 0=box, 1=cyl, 2=hex
	int 		primitive_id;
	int         n_transforms;
	int         material;
	int         is_outer_boundary;
	std::vector<wtransform>   transforms;
};

int primitive::num_primitives=0;

primitive::primitive(){
	//box default constructor
	min[0]=0;min[1]=0;min[2]=0;
	max[0]=0;max[1]=0;max[2]=0;
	location[0]=0;location[1]=0;location[2]=0;
	type=0;
	material=0;
	wtransform this_transform;
	this_transform.cellnum = 0;
	this_transform.cellmat = 0;
	this_transform.dx      = 0;
	this_transform.dy      = 0;
	this_transform.dz      = 0;
	this_transform.theta   = 0;
	this_transform.phi     = 0;
	transforms.push_back(this_transform);
	is_outer_boundary=0;
	n_transforms=1;
	primitive_id=num_primitives;
	num_primitives++;
}
primitive::primitive(int ptype, int cellnum ,int cellmat ,float xmin,float ymin,float zmin,float xmax,float ymax,float zmax,float x,float y,float z){
	//box valued constructor
	min[0]=xmin;min[1]=ymin;min[2]=zmin;
	max[0]=xmax;max[1]=ymax;max[2]=zmax;
	location[0]=x;location[1]=y;location[2]=z;
	type=ptype;
	material=cellmat;
	wtransform this_transform;
	this_transform.cellnum = cellnum;
	this_transform.cellmat = cellmat;
	this_transform.dx      = 0;
	this_transform.dy      = 0;
	this_transform.dz      = 0;
	this_transform.theta   = 0;
	this_transform.phi     = 0;
	transforms.push_back(this_transform);
	is_outer_boundary=0;
	n_transforms=1;
	primitive_id=num_primitives;
	num_primitives++;
}
primitive::~primitive(){
	//box destructor
	//delete min;
	//delete max;
	//delete location;
	//delete &box_id;
	//delete &n_transforms;
	//delete   &transforms;
}
void primitive::add_transform(){
	wtransform this_transform;
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
void primitive::add_transform(int cellnum , float dx , float dy , float dz , float theta , float phi ){
	wtransform this_transform;
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
void primitive::print_transform(){
	std::cout << "--- primitive id  = " << primitive_id << " ---" << "\n";
	std::cout << "   min,max  = (" << min[0] << " , " << min[1] << " , " << min[2] << "),(" << max[0] << " , " << max[1] << " , " << max[2] << ")" << "\n";
	std::cout << "   location = (" << location[0] << " , " << location[1] << " , " << location[2] << ")" << "\n";
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
void primitive::print_transform(int tnum){
	std::cout << "--- primitive id  = " << primitive_id << " ---" << "\n";
	std::cout << "   min,max  = (" << min[0] << " , " << min[1] << " , " << min[2] << "),(" << max[0] << " , " << max[1] << " , " << max[2] << ")" << "\n";
	std::cout << "   location = (" << location[0] << " , " << location[1] << " , " << location[2] << ")" << "\n";
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

void primitive::make_hex_array(int n, float x, float y, int starting_index){

	int k, j, num, cnt;
	float offsetx, offsety, fnum, lattr, PD_ratio;

	// add num of transforms to whatever is there
    n_transforms += 3*n*(n-1)+1;

    wtransform this_transform;
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


class wgeometry {
	int n_box;
	int n_cyl;
	int n_hex;
	int n_primitives;
	int n_transforms;
public:
	 wgeometry();
	~wgeometry();
	int get_primitive_count();
	void add_primitive();
	void update();
	void print();
	std::vector<primitive>   primitives;
};

wgeometry::wgeometry(){
	n_box        = 0;
	n_cyl        = 0;
	n_hex        = 0;
	n_primitives = 0;
	n_transforms = 0;
}
wgeometry::~wgeometry(){
	
}
void wgeometry::add_primitive(){
	primitive this_primitive;
	primitives.push_back(this_primitive);
	n_primitives++;
}
void wgeometry::update(){
	n_box        = 0;
	n_cyl        = 0;
	n_hex        = 0;
	n_transforms = 0;
	for(int k=0;k<n_primitives;k++){
		if(primitives[k].type==0){
				n_box+=primitives[k].n_transforms;
		}
		else if(primitives[k].type==1){
				n_cyl+=primitives[k].n_transforms;
		}
		else if(primitives[k].type==2){
				n_hex+=primitives[k].n_transforms;
		}
		n_transforms+=primitives[k].n_transforms;
	}
}
void wgeometry::print(){
	std::cout << "--- GEOMETRY SUMMARY ---"       << "\n";
	std::cout << "rectangular prisms = " << n_box << "\n";
	std::cout << "cylinders          = " << n_cyl << "\n";
	std::cout << "hexagons           = " << n_hex << "\n";
	std::cout << "total primitives   = " << n_primitives << "\n";
	std::cout << "total transforms   = " << n_transforms << "\n";
}
int wgeometry::get_primitive_count(){
	return(n_primitives);
}



/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//							OptiX stuff
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


class optix_stuff{
	optix::Context 	context;
	void make_geom(wgeometry);
public:
	CUdeviceptr * 			positions_ptr; 
	CUdeviceptr * 			rxn_ptr; 
	CUdeviceptr * 			done_ptr;
	CUdeviceptr * 			cellnum_ptr;
	int 			stack_size_multiplier;
	optix_stuff(int,wgeometry);
	void set_trace_type(int);
};

optix_stuff::optix_stuff(int N,wgeometry problem_geom){

	using namespace optix;

	// local variables

	char                path_to_ptx[512];
	Program           	ray_gen_program;
	Program           	exception_program;  
	Program           	miss_program;
	Buffer 				particles_buffer;
	Buffer 				      rxn_buffer;
	Buffer 				     done_buffer;
	Buffer 				  cellnum_buffer;
	Variable          	particles_obj;
	Variable 			      rxn_obj;
	Variable 			     done_obj;
	Variable 			  cellnum_obj;
	Variable          	outer_cell_var;
	Variable 			boundary_condition_var;
	Variable 			trace_type_var;
	RTsize              stack_size;
	
	// create OptiX context
	// Set up context
	//context = Context::create();
  	context->setRayTypeCount( 1 );
  	context->setEntryPointCount( 1 );
  	context["radiance_ray_type"]->setUint( 0u );
  	context["scene_epsilon"]->setFloat( 1.e-4f );
	context->setPrintEnabled( 1);
	context->setExceptionEnabled( RT_EXCEPTION_ALL, 1);

	stack_size = context->getStackSize();
	stack_size = stack_size_multiplier*stack_size;
	context->setStackSize( stack_size );
	printf("OptiX stack size is %d bytes\n",(unsigned) stack_size);
	
	// Render particle buffer and attach to variable, get pointer for CUDA
	particles_obj = context["particles"];
	particles_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT,RT_FORMAT_USER,N);
	particles_buffer->setElementSize( sizeof(source_point) );
	particles_obj->set(particles_buffer);
	particles_buffer->getDevicePointer(0,positions_ptr);

	/*

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

	*/
	
	// Ray generation program 
	sprintf( path_to_ptx, "%s", "camera.ptx" );
	ray_gen_program = context->createProgramFromPTXFile( path_to_ptx, "camera" );
	context->setRayGenerationProgram( 0, ray_gen_program );
	
	// Exception program
	exception_program = context->createProgramFromPTXFile( path_to_ptx, "exception" );
	context->setExceptionProgram( 0, exception_program ); 
	
	// Miss program 
	sprintf( path_to_ptx, "%s", "miss.ptx" );
	miss_program = context->createProgramFromPTXFile( path_to_ptx, "miss" );
	context->setMissProgram( 0, miss_program ); 
	
	//set outer cell number as (basically) global
	context["outer_cell"]->setUint(0);

	//set boundary condition for outer cell
	context["boundary_condition"]->setUint(0);

	//set trace type, 1=transport (writes intersection point and next cell), 2=fission (writes origin and current cell)
	context["trace_type"]->setUint(1);

	// make all geometry instances
	optix_stuff::make_geom(problem_geom);

	//validate and compile
	context->validate();
    context->compile();
}

void optix_stuff::set_trace_type(int trace_type){
	context["trace_type"]->setUint(trace_type);
}
void optix_stuff::make_geom(wgeometry problem_geom){

	using namespace optix;

	Group             	top_level_group;
	Variable          	top_object;
	RTacceleration      top_level_acceleration;
	Acceleration 		this_accel;
	GeometryGroup 		this_geom_group;

	// Make top level group/accel as children of the top level object 
	top_level_group = context->createGroup();
	context["top_object"]->set( top_level_group );
	top_level_group->setChildCount(problem_geom.get_primitive_count());
	this_accel = context->createAcceleration("Sbvh","Bvh");
	this_accel -> markDirty();
	top_level_group -> setAcceleration( this_accel );

	for(int j=0;j<problem_geom.get_primitive_count();j++){

		this_geom_group = context->createGeometryGroup();
		this_geom_group -> setChildCount(problem_geom.primitives[j].n_transforms);
		this_accel = context->createAcceleration("Sbvh","Bvh");
		this_accel -> markDirty();
		this_geom_group -> setAcceleration( this_accel );

		for (int k=0;k<problem_geom.primitives[j].n_transforms;k++){



		}

	}
	

	//make boxes


	//make cylinders


	//make hexes

}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//							History stuff
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


//history struct
struct history { 
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



