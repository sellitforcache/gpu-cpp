#include <vector> 
#include <iostream>
#include <sstream>
#include <cmath>
#include <cuda.h>
#include <curand.h>
#include <optixu/optixpp_namespace.h>
#include "datadef.h"
#include <cudpp_hash.h>
#include <Python.h>
#include <png++/png.hpp>
#include <cmath>


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
	 primitive(int,unsigned,unsigned,float,float,float,float,float,float,float,float,float);
	~primitive();
	void add_transform();
	void add_transform(unsigned,float,float,float,float,float);
	void print_transform();
	void print_transform(int);
	void make_hex_array(int,float,float,unsigned);
	float       min[3];
	float       max[3];
	float       location[3];
	static int  num_primitives;
	int			type;      // 0=box, 1=cyl, 2=hex
	int 		primitive_id;
	int         n_transforms;
	int         material;
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
	primitive_id=num_primitives;
	num_primitives++;
	n_transforms=0;
	//wtransform this_transform;
	//this_transform.cellnum = primitive_id;
	//this_transform.cellmat = 0;
	//this_transform.dx      = 0;
	//this_transform.dy      = 0;
	//this_transform.dz      = 0;
	//this_transform.theta   = 0;
	//this_transform.phi     = 0;
	//transforms.push_back(this_transform);

}
primitive::primitive(int ptype, unsigned cellnum ,unsigned cellmat ,float xmin,float ymin,float zmin,float xmax,float ymax,float zmax,float x,float y,float z){
	//box valued constructor
	min[0]=xmin;min[1]=ymin;min[2]=zmin;
	max[0]=xmax;max[1]=ymax;max[2]=zmax;
	location[0]=x;location[1]=y;location[2]=z;
	type=ptype;
	material=cellmat;
	primitive_id=num_primitives;
	num_primitives++;
	wtransform this_transform;
	this_transform.cellnum = cellnum;
	this_transform.cellmat = cellmat;
	this_transform.dx      = 0;
	this_transform.dy      = 0;
	this_transform.dz      = 0;
	this_transform.theta   = 0;
	this_transform.phi     = 0;
	transforms.push_back(this_transform);
	n_transforms=1;
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
	if(transforms.empty()){
		this_transform.cellnum = 0;
	}
	else{
	 	this_transform = transforms.back();
	}
	this_transform.cellmat = material;
	this_transform.dx      = 0;
	this_transform.dy      = 0;
	this_transform.dz      = 0;
	this_transform.theta   = 0;
	this_transform.phi     = 0;
	transforms.push_back(this_transform);
	n_transforms++;
}
void primitive::add_transform(unsigned cellnum , float dx , float dy , float dz , float theta , float phi ){
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

void primitive::make_hex_array(int n, float x, float y, unsigned starting_index){

	wtransform this_transform;

	int k, j, num, cnt;
	float offsetx, offsety, fnum, lattr, PD_ratio;

	// get strting cell number as the one set for the last
	//wtransform this_transform = transforms.back();
	//unsigned starting_index = this_transform.cellnum + 1;

	// add num of transforms to whatever is there
    n_transforms += 3*n*(n-1)+1;

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
	unsigned n_box;
	unsigned n_cyl;
	unsigned n_hex;
	unsigned n_primitives;
	unsigned n_transforms;
	unsigned outer_cell;
public:
	 wgeometry();
	~wgeometry();
	unsigned get_minimum_cell();
	unsigned get_maximum_cell();
	unsigned get_primitive_count();
	unsigned get_transform_count();
	void add_primitive();
	void update();
	void print_summary();
	void print_all();
	void set_outer_cell(unsigned);
	unsigned get_outer_cell();
	std::vector<primitive>   primitives;
};

wgeometry::wgeometry(){
	n_box        = 0;
	n_cyl        = 0;
	n_hex        = 0;
	n_primitives = 0;
	n_transforms = 0;
	outer_cell   = 0;
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
		if (primitives[k].n_transforms==0){
			std::cout << "No transforms for primitive id = " << primitives[k].primitive_id << ", it will not be included in problem geometry" << "\n";
		}
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
void wgeometry::print_summary(){
	std::cout << "\e[1;32m" << "--- GEOMETRY SUMMARY ---" << "\e[m \n";
	std::cout << "rectangular prisms = " << n_box << "\n";
	std::cout << "cylinders          = " << n_cyl << "\n";
	std::cout << "hexagons           = " << n_hex << "\n";
	std::cout << "total primitives   = " << n_primitives << "\n";
	std::cout << "total transforms   = " << n_transforms << "\n";
	std::cout << "outer cell         = " << outer_cell << "\n";
}
void wgeometry::print_all(){
	for(int k=0;k<n_primitives;k++){
		primitives[k].print_transform();
	}
	print_summary();
}
unsigned wgeometry::get_primitive_count(){
	return(n_primitives);
}
unsigned wgeometry::get_transform_count(){
	return(n_transforms);
}
void wgeometry::set_outer_cell(unsigned ocell){
	outer_cell = ocell;
	unsigned this_cell;
	for(int j=0;j<n_primitives;j++){
		for(int k=0;k<primitives[j].n_transforms;k++){
			this_cell = primitives[j].transforms[k].cellnum;
			if (this_cell==outer_cell){break;}
		}
	}
	if (this_cell!=outer_cell) {
		std::cout << "Cell " << ocell << " not found, outer cell not set!!!" << "\n";
	}
}
unsigned wgeometry::get_outer_cell(){
	return outer_cell;
}
unsigned wgeometry::get_minimum_cell(){
	unsigned mincell=-1;
	for(int j=0;j<n_primitives;j++){
		for(int k=0;k<primitives[j].n_transforms;k++){
			if (primitives[j].transforms[k].cellnum<mincell){mincell=primitives[j].transforms[k].cellnum;}
		}
	}
	return mincell;
}
unsigned wgeometry::get_maximum_cell(){
	unsigned maxcell=0;
	for(int j=0;j<n_primitives;j++){
		for(int k=0;k<primitives[j].n_transforms;k++){
			if (primitives[j].transforms[k].cellnum>maxcell){maxcell=primitives[j].transforms[k].cellnum;}
		}
	}
	return maxcell;
}




/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//							OptiX stuff
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


class optix_stuff{
	optix::Context 	context;
	unsigned mincell;
	unsigned maxcell;
	void make_geom(wgeometry);
	void init_internal(wgeometry);
public:
	CUdeviceptr 	positions_ptr; 
	CUdeviceptr 	      rxn_ptr; 
	CUdeviceptr 	     done_ptr;
	CUdeviceptr 	  cellnum_ptr;
	CUdeviceptr 	   matnum_ptr;
	unsigned 			stack_size_multiplier;
	unsigned 			N;
	optix_stuff(unsigned,unsigned);
	~optix_stuff();
	void init(wgeometry);
	void trace();
	void trace(unsigned);
	void set_trace_type(unsigned);
	void print();
	void trace_geometry(unsigned,unsigned,std::string);
	void make_color(float*,unsigned,unsigned,unsigned);
};
optix_stuff::optix_stuff(unsigned Nin,unsigned mult){
	//set stack size multiplier
	stack_size_multiplier = mult;
	//set main N
	N=Nin;
}
optix_stuff::~optix_stuff(){
	try {
		context->destroy();	
	} 
	catch( optix::Exception &e ){
		std::cout << e.getErrorString().c_str();
		exit(1);
	}
}
void optix_stuff::init_internal(wgeometry problem_geom){

	using namespace optix;

	// local variables
	char                path_to_ptx[512];
	Program           	ray_gen_program;
	Program           	exception_program;  
	Program           	miss_program;
	Buffer 				positions_buffer;
	Buffer 				      rxn_buffer;
	Buffer 				     done_buffer;
	Buffer 				  cellnum_buffer;
	Buffer 				   matnum_buffer;
	Variable          	positions_var;
	Variable 			      rxn_var;
	Variable 			     done_var;
	Variable 			  cellnum_var;
	Variable 			   matnum_var;
	Variable          	outer_cell_var;
	Variable 			boundary_condition_var;
	Variable 			trace_type_var;
	RTsize              stack_size;
	RTsize				printf_size;
	
	// Set up context
	context = Context::create();
  	context->setRayTypeCount( 1u );
  	context->setEntryPointCount( 1u );
  	context["radiance_ray_type"]->setUint( 0u );
  	context["scene_epsilon"]->setFloat( 1.e-4f );
	context->setPrintEnabled( 1);
	printf_size = context->getPrintBufferSize();
	context->setPrintBufferSize(printf_size*10);
	context->setExceptionEnabled( RT_EXCEPTION_ALL, 1);

	// set stack size
	stack_size = context->getStackSize();
	stack_size = stack_size_multiplier*stack_size;
	context->setStackSize( stack_size );
	
	// Render particle buffer and attach to variable, get pointer for CUDA
	positions_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT,RT_FORMAT_USER,N);
	positions_buffer -> setElementSize( sizeof(source_point) );
	positions_buffer -> getDevicePointer(0,&positions_ptr);  // 0 is optix device
	positions_var = context["positions_buffer"];
	positions_var -> set(positions_buffer);

	// Render reaction buffer and attach to variable, get pointer for CUDA
	rxn_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT,RT_FORMAT_USER,N);
	rxn_buffer -> setElementSize( sizeof(unsigned) );
	rxn_buffer -> getDevicePointer(0,&rxn_ptr);
	rxn_var = context["rxn_buffer"];
	rxn_var -> set(rxn_buffer);

	// Render done buffer and attach to variable, get pointer for CUDA
	done_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT,RT_FORMAT_USER,N);
	done_buffer -> setElementSize( sizeof(unsigned) );
	done_buffer -> getDevicePointer(0,&done_ptr);
	done_var = context["done_buffer"];
	done_var -> set(done_buffer);

	// Render cellnum buffer and attach to variable, get pointer for CUDA
	cellnum_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT,RT_FORMAT_USER,N);
	cellnum_buffer -> setElementSize( sizeof(unsigned) );
	cellnum_buffer -> getDevicePointer(0,&cellnum_ptr);
	cellnum_var = context["cellnum_buffer"];
	cellnum_var -> set(cellnum_buffer);

	// Render matnum buffer and attach to variable, get pointer for CUDA
	matnum_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT,RT_FORMAT_USER,N);
	matnum_buffer -> setElementSize( sizeof(unsigned) );
	matnum_buffer -> getDevicePointer(0,&matnum_ptr);
	matnum_var = context["matnum_buffer"];
	matnum_var -> set(matnum_buffer);

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

	//set boundary condition for outer cell
	context["boundary_condition"]->setUint(0);

	//set trace type, 1=transport (writes intersection point and next cell), 2=fission (writes origin and current cell)
	context["trace_type"]->setUint(1);

	// make all geometry instances
	make_geom(problem_geom);

	//set outer cell
	context["outer_cell"]->setUint(problem_geom.get_outer_cell());

	//validate and compile
	context->validate();
    context->compile();
}
void optix_stuff::init(wgeometry problem_geom){
	// set min and max cell numbers
	mincell = problem_geom.get_minimum_cell();
	maxcell = problem_geom.get_maximum_cell();
	// try to init optix
	try {
		init_internal(problem_geom);	
	} 
	catch( optix::Exception &e ){
		std::cout << e.getErrorString().c_str();
		exit(1);
	}
}
void optix_stuff::set_trace_type(unsigned trace_type){
	context["trace_type"]->setUint(trace_type);
}
void optix_stuff::trace(unsigned trace_type){
	context["trace_type"]->setUint(trace_type);
	context -> launch( 0 , N );
}
void optix_stuff::trace(){
	context -> launch( 0 , N );
}
void optix_stuff::make_geom(wgeometry problem_geom){

	using namespace optix;

	Group 				top_level_group;
	Variable 			top_object;
	Acceleration 		top_level_acceleration;
	Acceleration 		this_accel;

	GeometryGroup 		this_geom_group;
	Variable 			this_geom_min;
	Variable 			this_geom_max;
	Geometry 			this_geom;
	GeometryInstance 	ginst;
	Material 			material;
	Program  			intersection_program;
	Program  			bounding_box_program;
	Program  			closest_hit_program;
	Transform 			this_transform;
	Acceleration  		acceleration;
	Variable  			cellnum_var;
	Variable  			cellmat_var;

	char 				path_to_ptx[512];
	unsigned 			cellnum,cellmat;
	float 				dx,dy,dz,theta,phi;
	float 				m[16];
	unsigned 			uniqueindex = 0;

	// Make top level group/accel as children of the top level object 
	this_accel 	= context -> createAcceleration("Sbvh","Bvh");
	this_accel 	-> markDirty();
	top_level_group = context->createGroup();
	top_level_group ->setChildCount(problem_geom.get_transform_count());   // every primitive has at least 1 transform, so the total number of transforms is the number of instances
	top_level_group -> setAcceleration( this_accel );
	context["top_object"] -> set( top_level_group );

	for(int j=0;j<problem_geom.get_primitive_count();j++){

		//create this geometry type
		this_geom = context->createGeometry();
		this_geom -> setPrimitiveCount(1u);

		//set intersection and BB programs
		if      (problem_geom.primitives[j].type == 0)	{sprintf( path_to_ptx, "%s", "box.ptx" );}
		else if (problem_geom.primitives[j].type == 1)	{sprintf( path_to_ptx, "%s", "cylinder.ptx" );}
		else if (problem_geom.primitives[j].type == 2)	{sprintf( path_to_ptx, "%s", "hex.ptx" );}
		bounding_box_program = context->createProgramFromPTXFile( path_to_ptx, "bounds" );
		intersection_program = context->createProgramFromPTXFile( path_to_ptx, "intersect" );
		this_geom -> setBoundingBoxProgram ( bounding_box_program );
		this_geom -> setIntersectionProgram( intersection_program );

		//set hit programs to material
		sprintf( path_to_ptx, "%s", "hits.ptx" );
		closest_hit_program = context->createProgramFromPTXFile( path_to_ptx, "closest_hit" );
		material = context -> createMaterial();
		material -> setClosestHitProgram( 0, closest_hit_program );

		//set program variables for this instance
    	this_geom_min = this_geom["mins"];
    	this_geom_max = this_geom["maxs"];
    	this_geom_min -> set3fv( problem_geom.primitives[j].min );
    	this_geom_max -> set3fv( problem_geom.primitives[j].max );

		for (int k=0;k<problem_geom.primitives[j].n_transforms;k++){

			dx =      problem_geom.primitives[j].transforms[k].dx;
			dy =      problem_geom.primitives[j].transforms[k].dy;
			dz =      problem_geom.primitives[j].transforms[k].dz;
			theta =   problem_geom.primitives[j].transforms[k].theta;
			phi =     problem_geom.primitives[j].transforms[k].phi;
			cellnum = problem_geom.primitives[j].transforms[k].cellnum;
			cellmat = problem_geom.primitives[j].transforms[k].cellmat;

			//create instances
			ginst = context -> createGeometryInstance();
			ginst -> setGeometry( this_geom );
			ginst -> setMaterialCount( 1u );
			ginst -> setMaterial( 0, material );

			//set cell-specific variables
			cellnum_var = ginst["cellnum"];
			cellmat_var = ginst["cellmat"];
			cellnum_var -> setUint(cellnum);
			cellmat_var -> setUint(cellmat);

			// make geometry group for this primitive (to attach acceleration to)
			this_accel = context->createAcceleration("Sbvh","Bvh");
			this_accel -> markDirty();
			this_geom_group = context -> createGeometryGroup();
			this_geom_group -> setChildCount( 1u );
			this_geom_group -> setAcceleration( this_accel );
        
			//put geom instance into geomgroup
			this_geom_group -> setChild( 0, ginst );
    
        	//make transforms as necessary and attach to root node
        	//printf("cell %d: applying transform %d -  dx=%f dy=%f dz=%f theta=%f phi=%f\n",cellnum,k,dx,dy,dz,theta,phi);

			m[ 0] = cos(theta)*cos(phi);    m[ 1] = -cos(theta)*sin(phi);   m[ 2] = sin(theta);     m[ 3] = dx;
			m[ 4] = sin(phi);               m[ 5] = cos(phi);               m[ 6] = 0.0f;           m[ 7] = dy;
			m[ 8] = -sin(theta)*cos(phi);   m[ 9] = sin(theta)*sin(phi);    m[10] = cos(theta);     m[11] = dz;
			m[12] = 0.0f;                   m[13] = 0.0f;                   m[14] = 0.0f;           m[15] = 1.0f;
  
			this_transform = context -> createTransform();
			this_transform -> setChild(this_geom_group);
			this_transform -> setMatrix( 0, m, 0 );
			top_level_group -> setChild( uniqueindex , this_transform );
			uniqueindex++;

		}

	}

}
void optix_stuff::trace_geometry(unsigned width_in,unsigned height_in,std::string filename){

	std::cout << "\e[1;32m" << "Plotting Geometry... " << "\e[m \n";

	using namespace optix;

	//get aspect ratio and make N-compatible corresponding heights and widths
	float aspect = width_in / height_in;
	unsigned width  = sqrtf(N*aspect); 
	unsigned height = sqrtf(N/aspect);
	std::cout << "width  = " << width << "\n";
	std::cout << "height = " << height << "\n";

	// init the starting points to be across the z=0 plane and pointing downwards
	FILE* positionsfile = fopen("positionsfile","w");
	source_point * positions_local = new source_point[width*height];
	float dx = (42.0-(-42.0))/width;
	float dy = (42.0-(-42.0))/height;
	unsigned index;
	for(int j=0;j<height;j++){
		for(int k=0;k<width;k++){
			index = j * width + k;
			positions_local[index].x = -42.0 + dx/2 + k*dx;
			positions_local[index].y = -42.0 + dy/2 + j*dy;
			positions_local[index].z = 0.0;
			positions_local[index].xhat = 0.0;
			positions_local[index].yhat = 0.0;
			positions_local[index].zhat = -1.0;
			positions_local[index].samp_dist = 50000.0; 
		}
	}
	fclose(positionsfile);

	// copy starting positions data to pointer
	cudaMemcpy((void*)positions_ptr,positions_local,width*height*sizeof(source_point),cudaMemcpyHostToDevice);
	
	// trace with whereami?
	context["trace_type"]->setUint(2);
	context->launch(0,width*height);
	
	//copy to local buffer
	unsigned * image_local = new unsigned[width*height];
	cudaMemcpy(image_local,(void*)cellnum_ptr,width*height*sizeof(unsigned),cudaMemcpyDeviceToHost);

	// make image
	png::image< png::rgb_pixel > image(height, width);
	float * colormap = new float[3];
	for (size_t y = 0; y < image.get_height(); ++y)
	{
	    for (size_t x = 0; x < image.get_width(); ++x)
	    {
	    	make_color(colormap,image_local[y*width+x],mincell,maxcell);
	    	//printf("%u %u %6.3f %6.3f %6.3f\n",mincell,maxcell,colormap[0],colormap[1],colormap[2]);
	        image[y][x] = png::rgb_pixel(colormap[0],colormap[1],colormap[2]);
	    }
	}

	image.write(filename);

	std::cout << "Done.  Written to " << filename << "\n";

	delete image_local;
	delete colormap;
	delete positions_local;

}
void optix_stuff::print(){
	std::cout << "\e[1;32m" << "--- OptiX SUMMARY ---" << "\e[m \n";
	std::cout << "stack  size = " << context->getStackSize() << " bytes\n";
	std::cout << "printf size = " << context->getPrintBufferSize() << " bytes\n";
}
void optix_stuff::make_color(float* color, unsigned x, unsigned min, unsigned max){
	// red linear blue linear green sin colormap
	float normed_value = (float) (x-min+1)/(max+2-min);
	color[0] = normed_value;              //red
	color[1] = sin(normed_value*3.14159); //green
	color[2] = 1.0-normed_value;          //blue

	//bring up to 256 bits?
	color[0]=color[0]*256;
	color[1]=color[1]*256;
	color[2]=color[2]*256;

}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//							History stuff
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


//history struct
class whistory { 
	// cuda parameters
	int 	N;
	int     RNUM_PER_THREAD;
	int 	NUM_THREADS;
	int 	blks;
	// host data
    source_point *  space;
    unsigned *      xs_length_numbers;     // 0=isotopes, 1=main E points, 2=angular cosine points, 3=outgoing energy points
    float *   		xs_MT_numbers;
    float *			xs_data_MT;
	float *			xs_data_main_E_grid;
	float *			xs_data_Ang;
	float *			xs_data_Ang_grid;
	float *			xs_data_Ene;
	float *			xs_data_Ene_grid;
    float *         E;
    float *         Q;
    float *         rn_bank;
    unsigned *      cellnum;
    unsigned *      matnum;
    unsigned *      isonum;
    unsigned *      rxn;
    unsigned *      done;
    unsigned *      yield;
	// device data
	source_point *  d_space;
	unsigned *      d_xs_length_numbers;
	float * 		d_xs_MT_numbers;
    float *			d_xs_data_MT;
	float *			d_xs_data_main_E_grid;
	float *			d_xs_data_Ang;
	float *			d_xs_data_Ang_grid;
	float *			d_xs_data_Ene;
	float *			d_xs_data_Ene_grid;
    float *         d_E;
    float *         d_Q;
    float *         d_rn_bank;
    unsigned *      d_cellnum;
    unsigned *      d_matnum;
    unsigned *      d_isonum;
    unsigned *      d_rxn;
    unsigned *      d_done;
    unsigned *      d_yield;
    // xs data parameters
    std::string xs_isotope_string;
    unsigned 	xs_rows;
    unsigned 	xs_columns;
    unsigned    xs_bytes;
    unsigned 	xs_num_rxns_total;
    unsigned 	xs_num_angles;
    unsigned 	xs_num_distE;
    std::vector<unsigned> 	xs_num_rxns;
    std::vector<unsigned> 	xs_isotope_ints;
public:
     whistory(int,optix_stuff);
    ~whistory();
    void init_RNG();
    void init_CUDPP();
    void init_host();
    void copy_to_device();
    void load_cross_sections(std::string);
    void print_xs_data();
    void print_pointers();
    void write_xs_data(std::string);
};
whistory::whistory(int Nin, optix_stuff optix_obj){
	// CUDA stuff
	NUM_THREADS = 256;
	RNUM_PER_THREAD = 15;
	blks = ( N + NUM_THREADS-1 ) / NUM_THREADS;
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, (size_t) 10*1048576 );
	// device data stuff
	N = Nin;
				 d_space 	= (source_point*) optix_obj.positions_ptr;
				 d_cellnum 	= (unsigned*)     optix_obj.cellnum_ptr;
				 d_matnum 	= (unsigned*)     optix_obj.matnum_ptr;
				 d_rxn 		= (unsigned*)     optix_obj.rxn_ptr;
				 d_done 	= (unsigned*)     optix_obj.done_ptr;
	cudaMalloc( &d_xs_length_numbers	, 4*sizeof(unsigned) );		 
	cudaMalloc( &d_E 					, N*sizeof(float)    );
	cudaMalloc( &d_Q 					, N*sizeof(float)    );
	cudaMalloc( &d_rn_bank  			, N*RNUM_PER_THREAD*sizeof(float)    );
	cudaMalloc( &d_isonum   			, N*sizeof(unsigned) );
	cudaMalloc( &d_yield				, N*sizeof(unsigned) );
	// host data stuff
	xs_length_numbers 	= new unsigned [4];
	space 				= new source_point [N];
	E 					= new float [N];
	Q 					= new float [N];
	rn_bank  			= new float [N*RNUM_PER_THREAD];
	cellnum 			= new unsigned [N];
	matnum 				= new unsigned [N];
	rxn 				= new unsigned [N];
	done 				= new unsigned [N];
	isonum   			= new unsigned [N];
	yield	   			= new unsigned [N];
}
whistory::~whistory(){
	cudaFree( d_xs_length_numbers);
	cudaFree( d_xs_MT_numbers);
    cudaFree( d_xs_data_MT);
	cudaFree( d_xs_data_main_E_grid);
	cudaFree( d_xs_data_Ang);
	cudaFree( d_xs_data_Ang_grid);
	cudaFree( d_xs_data_Ene);
	cudaFree( d_xs_data_Ene_grid );
	cudaFree( E         );
	cudaFree( Q         );
	cudaFree( rn_bank   );
	cudaFree( isonum    );
	cudaFree( yield     );
	delete xs_length_numbers; 
    delete xs_MT_numbers;
    delete xs_data_MT;
	delete xs_data_main_E_grid;
	delete xs_data_Ang;
	delete xs_data_Ang_grid;
	delete xs_data_Ene;
	delete xs_data_Ene_grid;
	delete space;
	delete E;
	delete Q;
	delete rn_bank;
	delete cellnum;
	delete matnum;
	delete rxn;
	delete done;
	delete isonum;
	delete yield; 
}
void whistory::init_host(){

	for(int k=0;k<N;k++){
		space[k].x 			= 0.0;
		space[k].y 			= 0.0;
		space[k].z 			= 0.0;
		space[k].xhat 		= 0.0;
		space[k].yhat 		= 0.0;
		space[k].zhat 		= 0.0;
		space[k].samp_dist 	= 0.0;
		E[k]				= 0.0;
		Q[k]				= 0.0;
		cellnum[k]			= 0;
		matnum[k]			= 0;
		rxn[k]				= 0;
		done[k]				= 0;
		isonum[k]			= 0;
		yield[k]			= 0;
	}

}
void whistory::init_RNG(){
	std::cout << "\e[1;32m" << "Initializing random number bank on device using MTGP32..." << "\e[m \n";
	curandGenerator_t rand_gen ;
	curandCreateGenerator( &rand_gen , CURAND_RNG_PSEUDO_MTGP32 );  //mersenne twister type
	curandSetPseudoRandomGeneratorSeed( rand_gen , 1234ULL );
	curandGenerateUniform( rand_gen , d_rn_bank , N * RNUM_PER_THREAD );
	cudaMemcpy(rn_bank , d_rn_bank , N * RNUM_PER_THREAD , cudaMemcpyDeviceToHost); // copy bank back to keep seeds
}
void whistory::init_CUDPP(){

	CUDPPHandle            theCudpp;
	CUDPPHashTableConfig   hash_config;
	CUDPPConfiguration     compact_config;
	CUDPPConfiguration     redu_int_config;
	CUDPPConfiguration     redu_float_config;
	CUDPPHandle            mate_hash_table_handle;
	CUDPPHandle            fiss_hash_table_handle;
	CUDPPHandle            reduplan_int;
	CUDPPHandle            reduplan_float;
	CUDPPHandle            compactplan;
	CUDPPResult            res = CUDPP_SUCCESS;
	
	std::cout << "\e[1;32m" << "Initializing CUDPP..." << "\e[m \n";
	// global objects
	res = cudppCreate(&theCudpp);
	if (res != CUDPP_SUCCESS){fprintf(stderr, "Error initializing CUDPP Library.\n");}
	
	std::cout << "configuring sort..." << "\n";
	// sort stuff
	compact_config.op = CUDPP_ADD;
	compact_config.datatype = CUDPP_INT;
	compact_config.algorithm = CUDPP_COMPACT;
	compact_config.options = CUDPP_OPTION_FORWARD;
	res = cudppPlan(theCudpp, &compactplan, compact_config, N, 1, 0);
	if (CUDPP_SUCCESS != res){printf("Error creating CUDPPPlan for compact\n");exit(-1);}
	

	std::cout << "configuring reduction..." << "\n";
	// int reduction stuff
	redu_int_config.op = CUDPP_ADD;
	redu_int_config.datatype = CUDPP_INT;
	redu_int_config.algorithm = CUDPP_REDUCE;
	redu_int_config.options = 0;
	res = cudppPlan(theCudpp, &reduplan_int, redu_int_config, N, 1, 0);
	if (CUDPP_SUCCESS != res){printf("Error creating CUDPPPlan for reduction\n");exit(-1);}
	
	// float reduction stuff
	redu_float_config.op = CUDPP_ADD;
	redu_float_config.datatype = CUDPP_FLOAT;
	redu_float_config.algorithm = CUDPP_REDUCE;
	redu_float_config.options = 0;
	res = cudppPlan(theCudpp, &reduplan_float, redu_float_config, N, 1, 0);
	if (CUDPP_SUCCESS != res){printf("Error creating CUDPPPlan for reduction\n");exit(-1);}
	
	std::cout << "configuring hashes..." << "\n";
	// hash config stuff
	//hash_config.type = CUDPP_BASIC_HASH_TABLE;
	//hash_config.kInputSize = all_geom.all_total;
	//hash_config.space_usage = 1.2f;

	//material hash stuff
	//res = cudppHashTable(theCudpp, &mate_hash_table_handle, &hash_config);
	//if (res != CUDPP_SUCCESS){fprintf(stderr, "Error in cudppHashTable call (make sure your device is at least compute version 2.0\n");exit(-1);}
	//printf("\e[0;32m%-6s\e[m \n","  Inserting values into cellnum/matnum hash table...");
	//res = cudppHashInsert(mate_hash_table_handle, d_hash_key, d_hash_val_mate, hash_config.kInputSize);
	//if (res != CUDPP_SUCCESS){fprintf(stderr, "Error in inserting values into hash table\n");exit(-1);}

	//// fissile hash stuff
	//res = cudppHashTable(theCudpp, &fiss_hash_table_handle, &hash_config);
	//if (res != CUDPP_SUCCESS){fprintf(stderr, "Error in cudppHashTable call (make sure your device is at least compute version 2.0\n");exit(-1);}
	//printf("\e[0;32m%-6s\e[m \n","  Inserting values into cellnum/fissile hash table...");
	//res = cudppHashInsert(fiss_hash_table_handle, d_hash_key, d_hash_val_fiss, hash_config.kInputSize);
	//if (res != CUDPP_SUCCESS){fprintf(stderr, "Error in inserting values into hash table\n");exit(-1);}

}
void whistory::copy_to_device(){

	std::cout << "\e[1;32m" << "Copying data to device (number?)...";

	// copy history data
    cudaMemcpy( d_space,		space,		N*sizeof(source_point),	cudaMemcpyHostToDevice );
    cudaMemcpy( d_E,			E,			N*sizeof(float),		cudaMemcpyHostToDevice );
    cudaMemcpy( d_Q,    		Q,			N*sizeof(float),		cudaMemcpyHostToDevice );
    cudaMemcpy( d_done,			done,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );
    cudaMemcpy( d_cellnum,		cellnum,	N*sizeof(unsigned),		cudaMemcpyHostToDevice );
    cudaMemcpy( d_matnum,		matnum,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );
    cudaMemcpy( d_isonum,		isonum,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );
    cudaMemcpy( d_yield,		yield,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );
    cudaMemcpy( d_rxn,			rxn,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );
    // copy xs_data
	//cudaMemcpy( d_xs_data,		xs_data,	xs_bytes,				cudaMemcpyHostToDevice );

	std::cout << " Done." << "\e[m \n";

}
void whistory::load_cross_sections(std::string tope_string){

	printf("\e[1;32m%-6s\e[m \n","Loading cross sections and unionizing...");

	// set the string, make ints list
	xs_isotope_string = tope_string;
	std::istringstream ss(tope_string);
	std::string token;
	unsigned utoken;
	char tope_cstring[256];

	while(std::getline(ss, token, ',')) {
		utoken = std::atoi(token.c_str());
    	xs_isotope_ints.push_back(utoken);
	}

	// get data from python
	PyObject *pName, *pModule, *pDict, *pFunc;
    PyObject *pArgs, *pValue, *pString, *pBuffObj, *pObjList;
    Py_buffer pBuff[9];
    int i;
    char index_string[16];

    Py_Initialize();
    pName = PyString_FromString("unionize");
    /* Error checking of pName left out */

    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, "get_xs_pointers");
        /* pFunc is a new reference */

        if (pFunc && PyCallable_Check(pFunc)) {
        	pArgs = PyTuple_New(1);
            pString = PyString_FromString(xs_isotope_string.c_str());
            PyTuple_SetItem(pArgs, 0, pString);
            pObjList = PyObject_CallObject(pFunc, pArgs);
            for(int f=0;f<9;f++){  // go through list
            	pBuffObj = PyList_GetItem(pObjList,f);
            	if (PyObject_CheckBuffer(pBuffObj)){
            		PyObject_GetBuffer(pBuffObj, &pBuff[f],PyBUF_ND);
            	}
            	else{
            		printf("Object has no buffer\n");
            	}
            }
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", "unionize");
        return;
    }
    /*  list is:
			xs._get_length_numbers_pointer(), \
			xs._get_MT_number_totals_pointer(), \
			xs._get_MT_number_pointer(), \
			xs._get_main_Egrid_pointer(), \
			xs._get_MT_array_pointer(), \
			xs._get_Ang_Egrid_pointer(), \
			xs._get_Ang_array_pointer(), \
			xs._get_Ene_Egrid_pointer(), \
			xs._get_Ene_array_pointer()]
    */

    //set xs_data dimensions from python buffer
    xs_rows    			= pBuff[4].shape[0];
	xs_columns 			= pBuff[4].shape[1];
	xs_bytes   			= pBuff[4].len;
	//xs_num_rxns_total = 
	//xs_num_angles =
	//xs_num_distE =

    // allocate xs_data pointer, copy python buffer contents to pointer
    xs_data_MT  = (float*) malloc(xs_bytes);
    memcpy( xs_data_MT,   pBuff[4].buf , xs_bytes );
    
    //callocate device memory now that we know the size!!!!!
    cudaMalloc(&d_xs_data_MT,xs_bytes);

    Py_Finalize();

}
void whistory::print_xs_data(){
	printf("\e[1;32m%-6s\e[m \n","Cross section data info:");
	std::cout << "xs_data bytes   :" << xs_bytes << "\n";
	std::cout << "xs_data rows    :" << xs_rows << "\n";
	std::cout << "xs_data columns :" << xs_columns << "\n";
	std::cout << "c*r*4           :" << xs_columns*xs_rows*4 << "\n";
}
void whistory::write_xs_data(std::string filename){

	std::cout << "\e[1;32m" << "Writing xs_data to " << filename << "... ";

	FILE* xsfile = fopen(filename.c_str(),"w");

	for (int j=0;j<xs_rows;j++){
		for(int k=0;k<xs_columns;k++){
			fprintf(xsfile,"% 10.8E ",xs_data_MT[j*xs_columns+k]);
		}
		fprintf(xsfile,"\n");
	}

	fclose(xsfile);

	std::cout << "Done." << "\e[m \n";

}
void whistory::print_pointers(){
	std::cout << "\e[1;32m" << "Pointer Info:" << "\e[m \n";
	std::cout << "--- HOST ---" << "\n";
	std::cout << "  space:   " <<   space   << "\n";
	//std::cout << "  xs_data: " <<   xs_data << "\n";
	std::cout << "  E:       " <<   E       << "\n";
	std::cout << "  Q:       " <<   Q       << "\n";
	std::cout << "  rn_bank: " <<   rn_bank << "\n";
	std::cout << "  cellnum: " <<   cellnum << "\n";
	std::cout << "  matnum:  " <<   matnum  << "\n";
	std::cout << "  isonum:  " <<   isonum  << "\n";
	std::cout << "  rxn:     " <<   rxn     << "\n";
	std::cout << "  done:    " <<   done    << "\n";
	std::cout << "  yield:   " <<   yield   << "\n";
	std::cout << "--- DEVICE ---" << "\n";
	std::cout << "d_space:   " << d_space   << "\n";
	//std::cout << "d_xs_data: " << d_xs_data << "\n";
	std::cout << "d_E:       " << d_E       << "\n";
	std::cout << "d_Q:       " << d_Q       << "\n";
	std::cout << "d_rn_bank: " << d_rn_bank << "\n";
	std::cout << "d_cellnum: " << d_cellnum << "\n";
	std::cout << "d_matnum:  " << d_matnum  << "\n";
	std::cout << "d_isonum:  " << d_isonum  << "\n";
	std::cout << "d_rxn:     " << d_rxn     << "\n";
	std::cout << "d_done:    " << d_done    << "\n";
	std::cout << "d_yield:   " << d_yield   << "\n";
}



























