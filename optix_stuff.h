#include <optix_world.h>

class optix_stuff{
	optix::Context 	context;
	std::string accel_type, traverse_type;
	unsigned mincell;
	unsigned maxcell;
	unsigned compute_device;
	unsigned GEOM_FLAG;
	void make_geom_xform(wgeometry);
	void make_geom_prim(wgeometry);
	void init_internal(wgeometry, unsigned, std::string);
public:
	CUdeviceptr 	positions_ptr; 
	CUdeviceptr 	      rxn_ptr; 
	CUdeviceptr 	     done_ptr;
	CUdeviceptr 	  cellnum_ptr;
	CUdeviceptr 	   matnum_ptr;
	unsigned 			stack_size_multiplier;
	unsigned 			N;
	float 				outer_cell_dims[6];
	unsigned 			outer_cell_type;
	optix_stuff(unsigned,unsigned);
	optix_stuff();
	~optix_stuff();
	void init(wgeometry, unsigned, std::string);
	void trace();
	void trace(unsigned);
	void set_trace_type(unsigned);
	void print();
	void trace_geometry(unsigned,unsigned,std::string);
	void make_color(float*,unsigned,unsigned,unsigned);
};