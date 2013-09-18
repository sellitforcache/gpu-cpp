#include <vector> 
#include <iostream>
#include <cmath>
#include <cuda.h>
#include <curand.h>
#include <optixu/optixpp_namespace.h>
#include "datadef.h"
#include <cudpp_hash.h>
#include <Python.h>


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
primitive::primitive(int ptype, unsigned cellnum ,unsigned cellmat ,float xmin,float ymin,float zmin,float xmax,float ymax,float zmax,float x,float y,float z){
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
	unsigned n_box;
	unsigned n_cyl;
	unsigned n_hex;
	unsigned n_primitives;
	unsigned n_transforms;
public:
	 wgeometry();
	~wgeometry();
	unsigned get_primitive_count();
	unsigned get_transform_count();
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
unsigned wgeometry::get_primitive_count(){
	return(n_primitives);
}
unsigned wgeometry::get_transform_count(){
	return(n_transforms);
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
	CUdeviceptr 	positions_ptr; 
	CUdeviceptr 	      rxn_ptr; 
	CUdeviceptr 	     done_ptr;
	CUdeviceptr 	  cellnum_ptr;
	CUdeviceptr 	   matnum_ptr;
	unsigned 			stack_size_multiplier;
	unsigned 			N;
	optix_stuff(unsigned,unsigned);
	void init(wgeometry);
	void trace();
	void trace(unsigned);
	void set_trace_type(unsigned);
};
optix_stuff::optix_stuff(unsigned Nin,unsigned mult){
	//set stack size multiplier
	stack_size_multiplier = mult;
	//set main N
	N=Nin;
}
void optix_stuff::init(wgeometry problem_geom){

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
	
	// Set up context
	context = Context::create();
  	context->setRayTypeCount( 1u );
  	context->setEntryPointCount( 1u );
  	context["radiance_ray_type"]->setUint( 0u );
  	context["scene_epsilon"]->setFloat( 1.e-4f );
	context->setPrintEnabled( 1);
	context->setExceptionEnabled( RT_EXCEPTION_ALL, 1);

	// set stack size
	stack_size = context->getStackSize();
	stack_size = stack_size_multiplier*stack_size;
	context->setStackSize( stack_size );
	printf("OptiX stack size is %d bytes\n",(unsigned) stack_size);
	
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
	
	//set outer cell number as (basically) global
	context["outer_cell"]->setUint(0);

	//set boundary condition for outer cell
	context["boundary_condition"]->setUint(0);

	//set trace type, 1=transport (writes intersection point and next cell), 2=fission (writes origin and current cell)
	context["trace_type"]->setUint(1);

	// make all geometry instances
	make_geom(problem_geom);

	//validate and compile
	context->validate();
    context->compile();
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
		if (problem_geom.primitives[j].type == 0)		{sprintf( path_to_ptx, "%s", "box.ptx" );}
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


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//							History stuff
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


//history struct
class whistory { 
	int 	N;
	int     RNUM_PER_THREAD;
	int 	NUM_THREADS;
	int 	blks;
	source_point *  d_space;
    float *			d_xs_data;
    float *         d_E;
    float *         d_Q;
    float *         d_rn_bank;
    unsigned *      d_cellnum;
    unsigned *      d_matnum;
    unsigned *      d_isonum;
    unsigned *      d_rxn;
    unsigned *      d_done;
    unsigned *      d_yield;
public:
    source_point *  space;
    float *			xs_data;
    float *         E;
    float *         Q;
    float *         rn_bank;
    unsigned *      cellnum;
    unsigned *      matnum;
    unsigned *      isonum;
    unsigned *      rxn;
    unsigned *      done;
    unsigned *      yield;
     whistory(int,optix_stuff);
    ~whistory();
    void init_RNG();
    void init_CUDPP();
    void copy_to_device();
    void load_cross_sections(std::string);
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
	cudaMalloc( &d_E 		, N*sizeof(float)    );
	cudaMalloc( &d_Q 		, N*sizeof(float)    );
	cudaMalloc( &d_rn_bank  , N*sizeof(float)    );
	cudaMalloc( &d_isonum   , N*sizeof(unsigned) );
	cudaMalloc( &d_yield	, N*sizeof(unsigned) );
	// host data stuff
	space 		= new source_point [N];
	E 			= new float [N];
	Q 			= new float [N];
	rn_bank  	= new float [N];
	cellnum 	= new unsigned [N];
	matnum 		= new unsigned [N];
	rxn 		= new unsigned [N];
	done 		= new unsigned [N];
	isonum   	= new unsigned [N];
	yield	   	= new unsigned [N];
}
whistory::~whistory(){
	cudaFree( d_xs_data );
	cudaFree( E         );
	cudaFree( Q         );
	cudaFree( rn_bank   );
	cudaFree( isonum    );
	cudaFree( yield     );
	delete xs_data;
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
void whistory::init_RNG(){
	printf("\e[1;32m%-6s\e[m \n","Initializing random number bank on device using MTGP32...");
	curandGenerator_t rand_gen ;
	curandCreateGenerator( &rand_gen , CURAND_RNG_PSEUDO_MTGP32 );  //mersenne twister type
	curandSetPseudoRandomGeneratorSeed( rand_gen , 1234ULL );
	curandGenerateUniform( rand_gen , rn_bank , N * RNUM_PER_THREAD );
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
	

	printf("\e[0;32m%-6s\e[m \n","  Configuring reduction...");
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
	
	printf("\e[0;32m%-6s\e[m \n","  Configuring hashes...");
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

    cudaMemcpy( d_space,		space,		N*sizeof(source_point),	cudaMemcpyHostToDevice );
    cudaMemcpy( d_E,			E,			N*sizeof(float),		cudaMemcpyHostToDevice );
    cudaMemcpy( d_Q,    		Q,			N*sizeof(float),		cudaMemcpyHostToDevice );
    cudaMemcpy( d_done,			done,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );
    cudaMemcpy( d_cellnum,		cellnum,	N*sizeof(unsigned),		cudaMemcpyHostToDevice );
    cudaMemcpy( d_matnum,		matnum,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );
    cudaMemcpy( d_isonum,		isonum,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );
    cudaMemcpy( d_yield,		yield,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );
    cudaMemcpy( d_rxn,			rxn,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );

}
void whistory::load_cross_sections(std::string tope_string){

	PyObject *pName, *pModule, *pDict, *pFunc;
    PyObject *pArgs, *pValue, *pString, *pBuffObj;
    Py_buffer pBuff;
    int i;

    Py_Initialize();
    pName = PyString_FromString("unionize");
    /* Error checking of pName left out */

    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, "get_xs_pointer");
        /* pFunc is a new reference */

        if (pFunc && PyCallable_Check(pFunc)) {
        	pArgs = PyTuple_New(1);
            pString = PyString_FromString("92235,92238,8016,1001");
            PyTuple_SetItem(pArgs, 0, pString);
            pBuffObj = PyObject_CallObject(pFunc, pArgs);
            if (PyObject_CheckBuffer(pBuffObj)){
            	PyObject_GetBuffer(pBuffObj, &pBuff,PyBUF_ND);
            }
            else{
            	printf("Object has no buffer\n");
            }
            //Py_DECREF(pArgs);
            //if (pValue != NULL) {
            //    printf("Result of call: %p\n", pBuff.buf );
            //    Py_DECREF(pValue);
            //}
            //else {
            //    Py_DECREF(pFunc);
            //    Py_DECREF(pModule);
            //    PyErr_Print();
            //    fprintf(stderr,"Call failed\n");
            //    return;
            //}
        //}
        //else {
        //    if (PyErr_Occurred())
        //        PyErr_Print();
        //    fprintf(stderr, "Cannot find function \"%s\"\n","get_xs_pointer");
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", "unionize");
        return;
    }

    // allocate xs_data pointer, copy python buffer contents to pointer
    unsigned nbytes = pBuff.len;
    printf("length of array is %u bytes\n",nbytes);
    xs_data = (float*) malloc(nbytes);
    memcpy( xs_data,   pBuff.buf , nbytes );

    //set xs_data dimensions from python buffer

    Py_Finalize();

}


























