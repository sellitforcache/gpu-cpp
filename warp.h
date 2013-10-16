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
#include <assert.h>
#include <time.h>


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//							Prototype declaration stuff
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

// host calls
void print_banner();
// device calls
void set_positions_rand(unsigned , unsigned, unsigned, unsigned, source_point * , float *  , float  * );
void copy_points(unsigned , unsigned , unsigned , unsigned*  , unsigned  , unsigned *  , source_point *  , source_point * );
void sample_fission_spectra(unsigned,unsigned,unsigned,float*,float*);
void sample_isotropic_directions(unsigned, unsigned, unsigned, unsigned, source_point* , float*);
void macroscopic(unsigned , unsigned ,  unsigned, unsigned, unsigned , source_point * , unsigned* , unsigned * , unsigned * , float * , float * , float * , float *  , float* , unsigned*);
void microscopic(unsigned , unsigned ,  unsigned , unsigned , unsigned , unsigned* , unsigned * , float * , float * , float * , float *  , unsigned * , unsigned * ,  float* , unsigned * , float*, unsigned* );
void tally_spec(unsigned , unsigned ,  unsigned , unsigned , source_point * , float* , float * , unsigned * , unsigned*, unsigned*);
void escatter(unsigned , unsigned , unsigned, unsigned , unsigned* , unsigned* , float* , float*, source_point* , unsigned*, float*, unsigned*, float**);
void iscatter(unsigned , unsigned , unsigned , unsigned , unsigned* , unsigned * , float * , float *, source_point *  ,unsigned * , float* , float* , unsigned* , float**);
void fission(unsigned , unsigned , unsigned, unsigned , unsigned*  , unsigned*  , float * , unsigned* );
void absorb(unsigned , unsigned , unsigned , unsigned*  , unsigned* );
void find_E_grid_index(unsigned , unsigned , unsigned , unsigned , float * , float* , unsigned *, unsigned* );
void make_mask(unsigned, unsigned, unsigned, unsigned*, unsigned*, unsigned, unsigned);
void print_histories(unsigned, unsigned, unsigned, unsigned *, unsigned*, source_point*, float*, unsigned*);


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
	void add_transform(unsigned,float,float,float,float,float);  //defaults to primitive material
	void add_transform(unsigned,unsigned,float,float,float,float,float); //set own material
	void print_transform();
	void print_transform(int);
	void make_hex_array(int,float,float,unsigned);
	void make_hex_array(int,float,float,float,unsigned);
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
void primitive::add_transform(unsigned cellnum ,unsigned cellmat, float dx , float dy , float dz , float theta , float phi ){
	wtransform this_transform;
	this_transform.cellnum = cellnum;
	this_transform.cellmat = cellmat;
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

void primitive::make_hex_array(int n, float x, float y, float phi, unsigned starting_index){

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
    		this_transform.phi=phi;
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
	unsigned 	n_box;
	unsigned 	n_cyl;
	unsigned 	n_hex;
	unsigned 	n_primitives;
	unsigned 	n_transforms;
	unsigned 	outer_cell;
	unsigned 	n_materials;
	unsigned 	n_isotopes;
	unsigned * 	material_num_list;
	unsigned * 	cell_num_list;
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
	void add_material(unsigned , unsigned, unsigned , float, unsigned * , float * );
	int check();
	unsigned get_outer_cell_dims(float*);
	unsigned get_material_count();
	void make_material_table();
	void get_material_table(unsigned*,unsigned*,unsigned**,unsigned**,float**);
	void print_materials_table();
	std::vector<primitive>   	primitives;
	std::vector<material_def>	materials;
	std::vector<unsigned>		isotopes;
	std::string 				isotope_list;
	unsigned *	isotope_list_array;
	unsigned *	material_list_array;
	float * 	concentrations_matrix;
	float * 	awr_list;
};

wgeometry::wgeometry(){
	n_box        = 0;
	n_cyl        = 0;
	n_hex        = 0;
	n_primitives = 0;
	n_transforms = 0;
	outer_cell   = 0;
	n_materials  = 0;
	n_isotopes   = 0;
}
wgeometry::~wgeometry(){
	//material destructor
	//for(int k=0;k<n_materials;k++){
	//	delete materials[k].fractions;
	//	delete materials[k].isotopes;
	//}
	//delete cell_num_list;
	//delete material_num_list;
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

	// allocate arrays for lists
	cell_num_list = new unsigned [n_transforms];
	material_num_list = new unsigned [n_transforms]; // allocate enough for every cell to have its own material

	// compile list of all isotopes
	unsigned this_mat  = 0;
	unsigned n_topes   = 0;
	unsigned this_tope = 0;
	std::vector<unsigned>  all_isotopes;
	for(this_mat=0 ; this_mat<n_materials ; this_mat++){
		n_topes = materials[this_mat].num_isotopes;
		for(int k=0;k<n_topes;k++){
			all_isotopes.push_back(materials[this_mat].isotopes[k]);
		}
	}
	// go through list, get rid of extra copies
	n_isotopes = 0;
	unsigned notfound=0;
	//std::cout << "all_isotopes.size() = " << all_isotopes.size() << "\n";
	for(int k=0;k<all_isotopes.size();k++){
		notfound=1;
		for(int j=0;j<isotopes.size();j++){
			if(isotopes[j]==all_isotopes[k])
				notfound=0; 
		}
		if(notfound){
			isotopes.push_back(all_isotopes[k]);  //add if not found already
		}
	}
	n_isotopes = isotopes.size();

	//make string from isotope table
	char numstr[16];
	for(int k =0;k<n_isotopes;k++){
		sprintf(numstr,"%u",isotopes[k]);
		isotope_list += numstr;
		if(k<n_isotopes-1){
			isotope_list += ",";
		}
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
	std::cout << "\e[1;32m" << "--- INPUT MATERIAL SUMMARY ---" << "\e[m \n";
	std::cout << "materials          = " << n_materials << "\n";
	std::cout << "isotopes           = " << n_isotopes << "\n";
	std::cout << "isotope list:    " << isotope_list << "\n";
	std::cout << "  --------------   " << n_materials << "\n";
	for(int k=0;k<n_materials;k++){
		std::cout << "material #       = " << materials[k].matnum << "\n";
		std::cout << "density (g/cc)   = " << materials[k].density << "\n";
		std::cout << "is fissile       = " << materials[k].is_fissile << "\n";
		std::cout << "isotopes         = " << materials[k].num_isotopes << "\n";
		for(int j=0;j<materials[k].num_isotopes;j++){
			std::cout << "  number "<< j << ":  isotope " << materials[k].isotopes[j] << " frac = " << materials[k].fractions[j] << "\n";
		}
	}
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
void wgeometry::add_material(unsigned matnum , unsigned is_fissile, unsigned num_topes, float density, unsigned * isotopes, float * fractions){
	
	// get current material index
	unsigned dex = materials.size(); 

	material_def this_material_def;

	this_material_def.fractions = new float    [num_topes];
	this_material_def.isotopes  = new unsigned [num_topes];
	
	this_material_def.num_isotopes  = num_topes;
	this_material_def.matnum        = matnum;
	this_material_def.id 			= dex;
	this_material_def.density       = density;
	this_material_def.is_fissile    = is_fissile;
	memcpy(this_material_def.fractions,  fractions,   num_topes*sizeof(float));
	memcpy(this_material_def.isotopes,   isotopes,    num_topes*sizeof(unsigned));
	
	materials.push_back(this_material_def);

	n_materials++;
}
int wgeometry::check(){

	std::cout << "\e[1;32m" << "Checking cell numbers and materials..." << "\e[m \n";

	unsigned cellnum,matnum;
	unsigned cell_list_index = 0;
	unsigned mat_list_index  = 0;
	unsigned z,notfound,found_cell;
	// check that all cells have their own ID
	for (int k=0;k<n_primitives;k++){
		for (int j=0;j<primitives[k].n_transforms;j++){	
			cellnum = primitives[k].transforms[j].cellnum;
			matnum  = primitives[k].transforms[j].cellmat;
			// scan the cell list 
			for (z = 0 ; z<cell_list_index; z++){
				if (cell_num_list[z]==cellnum){
					std::cout << "cell number " << cellnum << " has duplicate entries!\n";
					return 1;
				}
			}
			cell_num_list[z]=cellnum; //append this cell number
			cell_list_index++;

			// scan the material list
			notfound=1;
			for (z = 0 ; z<mat_list_index ; z++){
				if (material_num_list[z]==matnum){
					notfound=0;
					break;   //break on this index if found
				}
			}
			if(notfound){
				material_num_list[mat_list_index]=matnum;  // append this material and increment index counter
				mat_list_index++;
			}
		}
	}

	// check that there are materials for each number specified in the geom

	for(int k=0;k<mat_list_index;k++){
		notfound=1;
		for(int j=0;j<n_materials;j++){
			if(material_num_list[k]==materials[j].matnum){
				notfound=0;
				break;
			}
		}
		if(notfound){
			std::cout << "material " << material_num_list[k] << " not defined!\n";
			return 1;
		}
	}

	// check to make sure the outer cell exists
	notfound = 1;
	for (int k=0;k<n_primitives;k++){
		for (int j=0;j<primitives[k].n_transforms;j++){	
			if(primitives[k].transforms[j].cellnum==outer_cell & notfound){
				notfound=0;
			}
		}
	}
	if(notfound){
		std::cout << "Cell " << outer_cell << " not found, cannot set it as the outer cell!\n";
		return 1;
	}

	std::cout << "They check out.\n";
	return 0;

}
unsigned wgeometry::get_outer_cell_dims(float * input_array){

	float this_min[3];
	float this_max[3];

	for (int k=0;k<n_primitives;k++){
		for (int j=0;j<primitives[k].n_transforms;j++){	
			if(primitives[k].transforms[j].cellnum==outer_cell){
				// apply transform to base primitive, just translation now, maybe add rotation later?  no this is a maximum extent projection onto the axes, should always be a box.
				memcpy(this_min , primitives[k].min , 3*sizeof(float));
				memcpy(this_max , primitives[k].max , 3*sizeof(float));
				this_min[0] += primitives[k].transforms[j].dx;
				this_min[1] += primitives[k].transforms[j].dy;
				this_min[2] += primitives[k].transforms[j].dz;
				this_max[0] += primitives[k].transforms[j].dx;
				this_max[1] += primitives[k].transforms[j].dy;
				this_max[2] += primitives[k].transforms[j].dz;
				// copy and return type
				memcpy(&input_array[0] , this_min , 3*sizeof(float));
				memcpy(&input_array[3] , this_max , 3*sizeof(float));
				return primitives[k].type;
			}
		}
	}

}
unsigned wgeometry::get_material_count(){
	return n_materials;
}
void wgeometry::make_material_table(){

	// allocate and copy the insotope list to the array
	isotope_list_array = new unsigned [n_isotopes];
	memcpy(isotope_list_array,isotopes.data(),n_isotopes*sizeof(unsigned));

	// allocate and copy the material number list to the array
	material_list_array = new unsigned [n_materials];
	for(int k=0;k<n_materials;k++){
		material_list_array[k]=materials[k].matnum;
	}

	// allocate and copy the fractions to the matrix
	unsigned notfound=1;
	int z=0;
	concentrations_matrix = new float [n_materials*n_isotopes];
	for(int j=0;j<n_materials;j++){     // isotope in a column
		for(int k=0;k<n_isotopes;k++){  // material in a row
			
			notfound=1;
			//scan the material object to see if the isotope is there
			for(z=0;z<materials[j].num_isotopes;z++){
				if(materials[j].isotopes[z] == isotope_list_array[k]){
					notfound=0;
					break;
				}
			}

			// use the internal index to copy to matrix
			if(notfound){
				concentrations_matrix[j*n_isotopes + k] = 0.0;
			}
			else{
				concentrations_matrix[j*n_isotopes + k] = materials[j].fractions[z];
			}
		}
	}

	// now convert fractions into number densities
	float frac   = 0.0;
	float m_avg  = 0.0;
	float N_avg  = 0.0;
	float awr    = 0.0;
	float dens   = 0.0;
	float u_to_g = 1.66053892e-24; // grams
	float m_n    = 1.008664916;    // u
	float barns  = 1e24;

	for(int j=0;j<n_materials;j++){

		m_avg = 0.0;
		frac  = 0.0;

		//normalize the fractions for this material and calculate average mass
		for(int k=0;k<n_isotopes;k++){
			frac += concentrations_matrix[j*n_isotopes+k];
		}
		for(int k=0;k<n_isotopes;k++){
			concentrations_matrix[j*n_isotopes+k] = concentrations_matrix[j*n_isotopes+k]/frac;
			m_avg += concentrations_matrix[j*n_isotopes+k] * awr_list[k] * m_n;
			//std::cout << "awr["<<k<<"] = "<<awr_list[k]<<"\n";
		}

		//get density
		dens = materials[j].density;

		// average num density
		N_avg = dens/(m_avg * u_to_g * barns);

		//  multiply normalized fractions by average number density to get topes number density
		for(int k=0;k<n_isotopes;k++){
			concentrations_matrix[j*n_isotopes+k] = concentrations_matrix[j*n_isotopes+k] * N_avg;
		}
	}
}
void wgeometry::get_material_table(unsigned* n_mat_in, unsigned * n_tope_in, unsigned** material_list_in, unsigned** isotope_list_in, float** conc_mat_in){

	*n_mat_in  = n_materials;
	*n_tope_in = n_isotopes;

	*material_list_in 	= new unsigned [n_materials];
	*isotope_list_in 	= new unsigned [n_isotopes];
	*conc_mat_in 		= new float    [n_materials*n_isotopes];

	memcpy(*material_list_in,  material_list_array,    n_materials*sizeof(unsigned)         );
	memcpy(*isotope_list_in,   isotope_list_array,     n_isotopes *sizeof(unsigned)         );
	memcpy(*conc_mat_in,       concentrations_matrix,  n_materials*n_isotopes*sizeof(float) );
}
void wgeometry::print_materials_table(){

	std::cout << "\e[1;32m" << "--- MATERIALS SUMMARY ---" << "\e[m \n";

	for(int j=0;j<n_materials;j++){

		assert(j==materials[j].id);
		std::cout <<  "material index " << j << " = material " << material_list_array[j] << "\n";
		std::cout <<  " (isotope index, ZZZAAA) \n";
		std::cout <<  " (number density #/bn-cm) \n";
		
		for(int k=0;k<n_isotopes;k++){

			if (k==n_isotopes-1){
				std::cout << "( "<< k << " , "<< isotope_list_array[k] << " ) \n";
			}
			else{
				std::cout << "  ( "<< k << " , "<< isotope_list_array[k] << " )     ";
			}
		}

		for(int k=0;k<n_isotopes;k++){

			if (k==n_isotopes-1){
				std::cout << "( " <<concentrations_matrix[j*n_isotopes+k] << " )\n";
			}
			else{
				std::cout << "  ( " <<concentrations_matrix[j*n_isotopes+k] << " )     ";
			}
		}

		
	}

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
	float 				outer_cell_dims[6];
	unsigned 			outer_cell_type;
	optix_stuff(unsigned,unsigned);
	optix_stuff();
	~optix_stuff();
	void init(wgeometry);
	void trace();
	void trace(unsigned);
	void set_trace_type(unsigned);
	void print();
	void trace_geometry(unsigned,unsigned,std::string);
	void make_color(float*,unsigned,unsigned,unsigned);
};
optix_stuff::optix_stuff(){}
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

	//set outer cell adn get its dimensions
	context["outer_cell"]->setUint(problem_geom.get_outer_cell());
	outer_cell_type = problem_geom.get_outer_cell_dims(outer_cell_dims);

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
	Variable 			cellfissile_var;

	char 				path_to_ptx[512];
	unsigned 			cellnum,cellmat;
	float 				dx,dy,dz,theta,phi;
	float 				m[16];
	unsigned 			uniqueindex = 0;
	unsigned 			is_fissile = 0;

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

			dx =          problem_geom.primitives[j].transforms[k].dx;
			dy =          problem_geom.primitives[j].transforms[k].dy;
			dz =          problem_geom.primitives[j].transforms[k].dz;
			theta =       problem_geom.primitives[j].transforms[k].theta;
			phi =         problem_geom.primitives[j].transforms[k].phi;
			cellnum =     problem_geom.primitives[j].transforms[k].cellnum;
			cellmat =     problem_geom.primitives[j].transforms[k].cellmat;
			for(int z=0;z<problem_geom.get_material_count();z++){
				if (cellmat == problem_geom.materials[z].matnum){
					is_fissile =  problem_geom.materials[z].is_fissile;   // set fissile flag
					cellmat    = problem_geom.materials[z].id;            // hash the material number to the ID, which is the matrix index, not that user-set number
					break;
				}
			}
			//std::cout << "cellnum " << cellnum << " matnum " << cellmat << " is fissile " << is_fissile << "\n";

			//create instances
			ginst = context -> createGeometryInstance();
			ginst -> setGeometry( this_geom );
			ginst -> setMaterialCount( 1u );
			ginst -> setMaterial( 0, material );

			//set cell-specific variables
			cellnum_var 	= ginst["cellnum"];
			cellmat_var 	= ginst["cellmat"];
			cellfissile_var = ginst["cellfissile"];
			cellnum_var 	-> setUint(cellnum);
			cellmat_var 	-> setUint(cellmat);
			cellfissile_var -> setUint(is_fissile);
			//std::cout << "cellnum,matnum,isfiss " << cellnum << " " << cellmat << " " << is_fissile << "\n";

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
	    	//mincell=0;
	    	//maxcell=3;
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
	// geommetry object
	wgeometry 			   problem_geom;
	// optix object 
	optix_stuff 		   optix_obj;
	// CUDPP
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
	CUDPPResult            res;
	unsigned * 				d_valid_result;
	unsigned * 				d_valid_N;
	unsigned * 				d_remap;
	// CURAND
	curandGenerator_t rand_gen;
	// cuda parameters
	unsigned 	N;
	unsigned    RNUM_PER_THREAD;
	unsigned 	NUM_THREADS;
	unsigned 	blks;
	// host data
	unsigned 		n_materials;
	unsigned 		n_isotopes;
	unsigned 		n_tally;
    source_point *  space;
    unsigned *      xs_length_numbers;     // 0=isotopes, 1=main E points, 2=total numer of reaction channels, 3=matrix E points, 4=angular cosine points, 5=outgoing energy points 
    unsigned *      xs_MT_numbers_total;
    unsigned *   	xs_MT_numbers;
    float *			xs_data_MT;
	float *			xs_data_main_E_grid;
	float **		xs_data_scatter;
	float **		xs_data_energy;
	float **		xs_data_scatter_host;
	float **		xs_data_energy_host;
	float * 		xs_data_Q;
    float *         E;
    float *         Q;
    float *         rn_bank;
    float * 		awr_list;
    float * 		tally_score;
    unsigned * 		tally_count;
    unsigned 		tally_cell;
    unsigned * 		index;
    unsigned *      cellnum;
    unsigned *      matnum;
    unsigned *      isonum;
    unsigned *      rxn;
    unsigned *      done;
    unsigned *      yield;
    unsigned * 		material_list;
    unsigned * 		isotope_list;
    float *  		number_density_matrix;
    unsigned 		reduced_yields;
    unsigned * 		remap;
    unsigned * 		zeros;
    unsigned * 		ones;
	// device data
	source_point *  d_space;
	unsigned *      d_xs_length_numbers;
	unsigned * 		d_xs_MT_numbers_total;
	unsigned * 		d_xs_MT_numbers;
    float *			d_xs_data_MT;
	float *			d_xs_data_main_E_grid;
	float **		d_xs_data_scatter;
	float ** 		d_xs_data_energy;
	float * 		d_xs_data_Q;
	float *         d_E;
	float *         d_Q;
	float *         d_rn_bank;
	float * 		d_awr_list;
	float * 		d_tally_score;
	unsigned * 		d_tally_count;
	unsigned * 		d_index;
	unsigned *      d_cellnum;
	unsigned *      d_matnum;
	unsigned *      d_isonum;
	unsigned *      d_rxn;
	unsigned *      d_done;
	unsigned *      d_yield;
	unsigned * 		d_material_list;
	unsigned * 		d_isotope_list;
    float *  		d_number_density_matrix;
    unsigned * 		d_reduced_yields;
    unsigned * 		d_reduced_done;
    source_point * 	d_fissile_points;
    unsigned * 		d_mask;
    // xs data parameters
    std::string xs_isotope_string;
    std::vector<unsigned> 	xs_num_rxns;
    std::vector<unsigned> 	xs_isotope_ints;
    unsigned 		total_bytes_scatter;
    unsigned 		total_bytes_energy ;
    unsigned 		MT_rows;
    unsigned 		MT_columns;
    //geom parameters
    float 			outer_cell_dims [6];
    // private transport functions
public:
     whistory(int,wgeometry);
    ~whistory();
    void init_RNG();
    void update_RNG();
    void init_CUDPP();
    void init_host();
    void copy_to_device();
    void load_cross_sections();
    void print_xs_data();
    void print_pointers();
    void print_materials_table();
    void converge(unsigned);
    void sample_fissile_points();
    float reduce_yield();
    void run(unsigned);
    unsigned reduce_done();
    unsigned reset_cycle(unsigned);
    void reset_fixed();
    void trace(unsigned);
    float get_time();
    void write_xs_data(std::string);
    void write_tally(unsigned, std::string);
    void set_tally_cell(unsigned);
};
whistory::whistory(int Nin, wgeometry problem_geom_in){
	// do problem gemetry stuff first
	problem_geom = problem_geom_in;
	// init optix stuff second
	optix_obj.N=Nin;
	optix_obj.stack_size_multiplier=12;
	optix_obj.init(problem_geom);
	optix_obj.print();
	// CUDA stuff
	N=Nin;
	NUM_THREADS = 256;
	RNUM_PER_THREAD = 15;
	blks = ( N + NUM_THREADS - 1 ) / NUM_THREADS;
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, (size_t) 10*1048576 );
	// set tally vector length
	n_tally = 1024;
	// device data stuff
	N = Nin;
				 d_space 	= (source_point*) optix_obj.positions_ptr;
				 d_cellnum 	= (unsigned*)     optix_obj.cellnum_ptr;
				 d_matnum 	= (unsigned*)     optix_obj.matnum_ptr;
				 d_rxn 		= (unsigned*)     optix_obj.rxn_ptr;
				 d_done 	= (unsigned*)     optix_obj.done_ptr;
	cudaMalloc( &d_xs_length_numbers	, 6*sizeof(unsigned) );		 
	cudaMalloc( &d_E 					, N*sizeof(float)    );
	cudaMalloc( &d_Q 					, N*sizeof(float)    );
	cudaMalloc( &d_rn_bank  			, N*RNUM_PER_THREAD*sizeof(float)    );
	cudaMalloc( &d_isonum   			, N*sizeof(unsigned) );
	cudaMalloc( &d_yield				, N*sizeof(unsigned) );
	cudaMalloc( &d_index				, N*sizeof(unsigned) );
	cudaMalloc( &d_tally_score  		, n_tally*sizeof(float));
	cudaMalloc( &d_tally_count  		, n_tally*sizeof(unsigned));
	cudaMalloc( &d_reduced_yields 		, 1*sizeof(unsigned));
	cudaMalloc( &d_reduced_done 		, 1*sizeof(unsigned));
	cudaMalloc(	&d_valid_result			, N*sizeof(unsigned));
	cudaMalloc(	&d_valid_N				, 1*sizeof(unsigned));
	cudaMalloc(	&d_remap				, N*sizeof(unsigned));
	cudaMalloc(	&d_fissile_points		, N*sizeof(source_point));
	cudaMalloc( &d_mask 				, N*sizeof(unsigned));
	// host data stuff
	//xs_length_numbers 	= new unsigned [6];
	space 				= new source_point [N];
	E 					= new float [N];
	Q 					= new float [N];
	rn_bank  			= new float [N*RNUM_PER_THREAD];
	tally_score 		= new float [n_tally];
	tally_count 		= new unsigned [n_tally];
	index     			= new unsigned [N];
	cellnum 			= new unsigned [N];
	matnum 				= new unsigned [N];
	rxn 				= new unsigned [N];
	done 				= new unsigned [N];
	isonum   			= new unsigned [N];
	yield	   			= new unsigned [N];
	remap 				= new unsigned [N];
	zeros 				= new unsigned [N];
	ones 				= new unsigned [N];
	// init counters to 0
	total_bytes_scatter = 0;
	total_bytes_energy  = 0;
	//copy any info needed
	memcpy(outer_cell_dims,optix_obj.outer_cell_dims,6*sizeof(float));
	xs_isotope_string = problem_geom.isotope_list;
	// init remapping stuff for compaction
	for(int k =0;k<N;k++){remap[k]=k;}
	// zero out the zeros vector
	for(int k =0;k<N;k++){zeros[k]=0;ones[k]=1;}
}
whistory::~whistory(){
	cudaFree( d_xs_length_numbers 	);
	cudaFree( d_xs_MT_numbers_total );
	cudaFree( d_xs_MT_numbers 		);
	cudaFree( d_xs_data_MT 			);
	cudaFree( d_xs_data_main_E_grid );
	cudaFree( d_xs_data_scatter     );
	cudaFree( d_xs_data_energy      );
	cudaFree( d_tally_score 		);
    cudaFree( d_tally_count 		);
    cudaFree( d_xs_data_Q    		);
	cudaFree( d_index   );
	cudaFree( d_E         );
	cudaFree( d_Q         );
	cudaFree( d_rn_bank   );
	cudaFree( d_isonum    );
	cudaFree( d_yield     );
	cudaFree( d_awr_list);
	delete xs_length_numbers; 
	delete xs_MT_numbers_total;
    delete xs_MT_numbers;
    delete xs_data_MT;
	delete xs_data_main_E_grid;
	delete xs_data_Q;
	delete space;
	delete index;
	delete awr_list;
	delete E;
	delete Q;
	delete rn_bank;
	delete cellnum;
	delete matnum;
	delete rxn;
	delete done;
	delete isonum;
	delete yield; 
	delete tally_count;
	delete tally_score;
	delete zeros;
	delete ones;
	// for loops to deallocate everything in the pointer arrays
	for (int j=0 ; j < MT_columns ; j++){  //start after the total xs and total abs vectors
		for (int k=0 ; k < MT_rows ; k++){
			// scatter
			//std::cout << "j,k = " << j << ", " << k << " colums,rows = " << MT_columns << ", " << MT_rows << "\n";
			float * this_pointer =   xs_data_scatter     [k*MT_columns + j];
			float * cuda_pointer =   xs_data_scatter_host[k*MT_columns + j];
			if(this_pointer!=NULL){
				while(xs_data_scatter[(k+1)*MT_columns + j ]==this_pointer){
					k++; //push k to the end of the copies so don't try to free it twice
				}
				//std::cout << "j,k " << j << ", " << k << " - " ;
				//std::cout << "freeing " << this_pointer << " " << cuda_pointer << "\n";
				delete this_pointer;
				cudaFree(cuda_pointer);
			}
		}
	}
	//delete pointer arrays themselves
	delete xs_data_scatter;
	delete xs_data_energy;
	delete xs_data_scatter_host;
	delete xs_data_energy_host;
}
void whistory::init_host(){

	for(int k=0;k<N;k++){
		space[k].x 			= 0.0;
		space[k].y 			= 0.0;
		space[k].z 			= 0.0;
		space[k].xhat 		= 0.0;
		space[k].yhat 		= 0.0;
		space[k].zhat 		= 0.0;
		space[k].samp_dist 	= 10000.0;
		space[k].macro_t 	= 0.0;
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
	curandCreateGenerator( &rand_gen , CURAND_RNG_PSEUDO_MTGP32 );  //mersenne twister type
	curandSetPseudoRandomGeneratorSeed( rand_gen , 1234ULL );
	curandGenerateUniform( rand_gen , d_rn_bank , N * RNUM_PER_THREAD );
	cudaMemcpy(rn_bank , d_rn_bank , N * RNUM_PER_THREAD , cudaMemcpyDeviceToHost); // copy bank back to keep seeds
}
void whistory::update_RNG(){

	curandGenerateUniform( rand_gen , d_rn_bank , N * RNUM_PER_THREAD );

}
void whistory::init_CUDPP(){
	
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
	
	//std::cout << "configuring hashes..." << "\n";
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
float whistory::reduce_yield(){

	res = cudppReduce(reduplan_int, d_reduced_yields, d_yield, N);
	if (res != CUDPP_SUCCESS){fprintf(stderr, "Error in reducing yield values\n");exit(-1);}
	cudaMemcpy(&reduced_yields, d_reduced_yields, 1*sizeof(unsigned), cudaMemcpyDeviceToHost);

	float keff = (float)reduced_yields / (float)N ;

	return keff;

}
unsigned whistory::reduce_done(){

	unsigned reduced_done = 0;

	res = cudppReduce(reduplan_int, d_reduced_done, d_done, N);
	if (res != CUDPP_SUCCESS){fprintf(stderr, "Error in reducing done values\n");exit(-1);}
	cudaMemcpy(&reduced_done, d_reduced_done, 1*sizeof(unsigned), cudaMemcpyDeviceToHost);

	return reduced_done;

}
void whistory::copy_to_device(){

	float * this_pointer;
	float * cuda_pointer;
	unsigned vlen;
	float * temp = new float [128];
	for(int g=0;g<128;g++){temp[g]=123456789;}

	std::cout << "\e[1;32m" << "Copying data to device (number?)..." << "\e[m \n";

	// copy history data
	std::cout << "History data... ";
    cudaMemcpy( d_space,		space,		N*sizeof(source_point),	cudaMemcpyHostToDevice );
    cudaMemcpy( d_E,			E,			N*sizeof(float),		cudaMemcpyHostToDevice );
    cudaMemcpy( d_Q,    		Q,			N*sizeof(float),		cudaMemcpyHostToDevice );
    cudaMemcpy( d_done,			done,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );
    cudaMemcpy( d_cellnum,		cellnum,	N*sizeof(unsigned),		cudaMemcpyHostToDevice );
    cudaMemcpy( d_matnum,		matnum,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );
    cudaMemcpy( d_isonum,		isonum,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );
    cudaMemcpy( d_yield,		yield,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );
    cudaMemcpy( d_rxn,			rxn,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );
    cudaMemcpy( d_remap, 		remap,    	N*sizeof(unsigned),		cudaMemcpyHostToDevice );
    std::cout << "Done.\n";
    std::cout << "Unionized cross sections... ";
    // copy xs_data,  0=isotopes, 1=main E points, 2=total numer of reaction channels
    cudaMemcpy( d_xs_length_numbers, 	xs_length_numbers,		6 																*sizeof(unsigned), 	cudaMemcpyHostToDevice );
    cudaMemcpy( d_xs_MT_numbers_total, 	xs_MT_numbers_total,	xs_length_numbers[0]											*sizeof(unsigned), 	cudaMemcpyHostToDevice );
    cudaMemcpy( d_xs_MT_numbers,		xs_MT_numbers,			(xs_length_numbers[2]+xs_length_numbers[0])						*sizeof(unsigned), 	cudaMemcpyHostToDevice );
    cudaMemcpy(	d_xs_data_MT,			xs_data_MT,				MT_rows*MT_columns 												*sizeof(float), 	cudaMemcpyHostToDevice );
	cudaMemcpy(	d_xs_data_main_E_grid,	xs_data_main_E_grid,	xs_length_numbers[1]											*sizeof(float), 	cudaMemcpyHostToDevice );
	cudaMemcpy( d_awr_list, 			awr_list,   			xs_length_numbers[0]       										*sizeof(float), 	cudaMemcpyHostToDevice );
	cudaMemcpy( d_material_list,        material_list,         	n_materials     												*sizeof(unsigned), 	cudaMemcpyHostToDevice );
	cudaMemcpy( d_isotope_list,         isotope_list,          	xs_length_numbers[0]            								*sizeof(unsigned), 	cudaMemcpyHostToDevice );
	cudaMemcpy( d_number_density_matrix,number_density_matrix, 	n_materials*xs_length_numbers[0] 								*sizeof(float),    	cudaMemcpyHostToDevice );
	cudaMemcpy( d_xs_data_Q,			xs_data_Q, 				(xs_length_numbers[2]+xs_length_numbers[0])						*sizeof(float), 	cudaMemcpyHostToDevice );
	std::cout << "Done.\n";
	// copy device pointer array to device array
	std::cout << "Pointer arrays... ";
	cudaMemcpy( d_xs_data_scatter, 	xs_data_scatter_host,	MT_rows*MT_columns*sizeof(float), cudaMemcpyHostToDevice); 	
	cudaMemcpy( d_xs_data_energy, 	xs_data_energy_host,	MT_rows*MT_columns*sizeof(float), cudaMemcpyHostToDevice); 	
	std::cout << "Done.\n";
	// copy scattering data to device array pointers
	std::cout << "Scattering data... ";
	for (int j=0 ; j < MT_columns ; j++){  //start after the total xs and total abs vectors
		for (int k=0 ; k < MT_rows ; k++){
			// scatter
			this_pointer =   xs_data_scatter     [k*MT_columns + j];
			cuda_pointer =   xs_data_scatter_host[k*MT_columns + j];
			if(this_pointer!=NULL & k<MT_rows-1){
				while(xs_data_scatter[(k+1)*MT_columns + j ]==this_pointer){
					k++; //push k to the end of the copies so don't try to free it twice
				}
				memcpy(&vlen,&this_pointer[0],sizeof(float));
				cudaMemcpy(cuda_pointer,this_pointer,(2*vlen+1)*sizeof(float),cudaMemcpyHostToDevice);
			}
		}
	}
	std::cout << " Done.\n";
	// zero out tally arrays
	std::cout << "Zeroing tally arrays... ";
	cudaMemcpy( d_tally_score, 	zeros,	n_tally*sizeof(float),    cudaMemcpyHostToDevice); 	
	cudaMemcpy( d_tally_count,	zeros,	n_tally*sizeof(unsigned), cudaMemcpyHostToDevice); 	
	std::cout << "Done.\n";

}
void whistory::load_cross_sections(){
	
	printf("\e[1;32m%-6s\e[m \n","Loading cross sections and unionizing...");

	// set the string, make ints list
	std::istringstream ss(xs_isotope_string);
	std::string token;
	unsigned utoken;
	unsigned bytes,rows,columns;

	while(std::getline(ss, token, ',')) {
		utoken = std::atoi(token.c_str());
    	xs_isotope_ints.push_back(utoken);
	}

	// get data from python
	/* 	 need to do
	xs = unionize.cross_section_data()
	xs._init_from_string(this_string)
	xs._read_tables()
	xs._unionize()
	xs._insert_reactions()
	xs._allocate_arrays()
	xs._interpolate() */

	// variables
	PyObject *pName, *pModule, *pDict, *pFunc;
    PyObject *pArgs, *pValue, *pString, *pBuffObj, *pObjList;
    PyObject *call_result;
    PyObject *call_string,*arg_string;
    PyObject *xsdat_instance;
    PyObject *pClass;
    Py_buffer pBuff;
    int i;

    Py_Initialize();

    pName = PyString_FromString("unionize");
    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if ( pModule==NULL ){
        PyErr_Print();
        fprintf(stderr, "Failed to import \"%s\"\n", "unionize");
        return;	
    }

    pName = PyString_FromString("cross_section_data");
    xsdat_instance = PyObject_CallMethodObjArgs(pModule,pName,NULL);
    PyErr_Print();
    Py_DECREF(pName);


    if (xsdat_instance != NULL) {

		// init the libraries wanted
		char tope_string_c[256];
		call_string = PyString_FromString("_init_from_string");
		arg_string  = PyString_FromString(xs_isotope_string.c_str());
		call_result = PyObject_CallMethodObjArgs(xsdat_instance, call_string, arg_string, NULL);
		PyErr_Print();
		Py_DECREF(arg_string);
		Py_DECREF(call_string);
		Py_DECREF(call_result);
	
		// read the tables
		call_string = PyString_FromString("_read_tables");
		call_result = PyObject_CallMethodObjArgs(xsdat_instance, call_string, NULL);
		PyErr_Print();
		Py_DECREF(call_string);
		Py_DECREF(call_result);

		// unionize the main energy grid across all isotopes
		call_string = PyString_FromString("_unionize");
		call_result = PyObject_CallMethodObjArgs(xsdat_instance, call_string, NULL);
		PyErr_Print();
		Py_DECREF(call_string);
		Py_DECREF(call_result);

		// make the total MT reaction list from all isotopes
		call_string = PyString_FromString("_insert_reactions");
		call_result = PyObject_CallMethodObjArgs(xsdat_instance, call_string, NULL);
		PyErr_Print();
		Py_DECREF(call_string);
		Py_DECREF(call_result);

		// allocate the unionized array
		call_string = PyString_FromString("_allocate_arrays");
		call_result = PyObject_CallMethodObjArgs(xsdat_instance, call_string, NULL);
		PyErr_Print();
		Py_DECREF(call_string);
		Py_DECREF(call_result);

		// insert and interpolate the cross sections
		call_string = PyString_FromString("_interpolate");
		call_result = PyObject_CallMethodObjArgs(xsdat_instance, call_string, NULL);
		PyErr_Print();
		Py_DECREF(call_string);
		Py_DECREF(call_result);

    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to instanciate \"%s\"\n", "unionize.cross_section_data");
        return;
    }


    // get the MT array buffer
    call_string = PyString_FromString("_get_MT_array_pointer");
	call_result = PyObject_CallMethodObjArgs(xsdat_instance, call_string, NULL);
	Py_DECREF(call_string);
	if (PyObject_CheckBuffer(call_result)){
		PyObject_GetBuffer(call_result, &pBuff,PyBUF_ND);
	}
	else{
		PyErr_Print();
        fprintf(stderr, "Returned object does not support buffer interface\n");
        return;
	}

    //
    // get and copy the unionized MT array
    //
	MT_rows    = pBuff.shape[0];
	MT_columns = pBuff.shape[1];
	bytes   = pBuff.len;
	//std::cout << "unionized MT array " << MT_rows << " " << MT_columns << " " << bytes << "\n";
    // allocate xs_data pointer arrays
    xs_data_MT       = new float  [MT_rows*MT_columns];
    // check to make sure bytes *= elements
    assert(bytes==MT_rows*MT_columns*4);
    // copy python buffer contents to pointer
    memcpy( xs_data_MT,   pBuff.buf , bytes );
    // cudaallocate device memory now that we know the size!
    cudaMalloc(&d_xs_data_MT,bytes);
    // release python variable to free memory
    Py_DECREF(call_result);




    // get the unionized main energy grid buffer
    call_string = PyString_FromString("_get_main_Egrid_pointer");
	call_result = PyObject_CallMethodObjArgs(xsdat_instance, call_string, NULL);
	Py_DECREF(call_string);
	if (PyObject_CheckBuffer(call_result)){
		PyObject_GetBuffer(call_result, &pBuff,PyBUF_ND);
	}
	else{
		PyErr_Print();
        fprintf(stderr, "Returned object does not support buffer interface\n");
        return;
	}

    //
    // get and copy unionized main energy grid
    //
	rows    = pBuff.shape[0];
	columns = pBuff.shape[1];
	bytes   = pBuff.len;
	//std::cout << "main e grid " << rows << " " << columns << " " << bytes << "\n";
    // allocate xs_data pointer arrays
    xs_data_main_E_grid  = new float  [rows];
    // check to make sure bytes *= elements
    assert(bytes==rows*4);
    // copy python buffer contents to pointer
    memcpy( xs_data_main_E_grid,   pBuff.buf , bytes );
    // cudaallocate device memory now that we know the size!
    cudaMalloc(&d_xs_data_main_E_grid,bytes);
    // release python variable to free memory
    Py_DECREF(call_result);




    // mt number vector
    call_string = PyString_FromString("_get_MT_numbers_pointer");
	call_result = PyObject_CallMethodObjArgs(xsdat_instance, call_string, NULL);
	Py_DECREF(call_string);
	if (PyObject_CheckBuffer(call_result)){
		PyObject_GetBuffer(call_result, &pBuff,PyBUF_ND);
	}
	else{
		PyErr_Print();
        fprintf(stderr, "Returned object does not support buffer interface\n");
        return;
	}

    //
    // get and copy mt number vector
    //
	rows    = pBuff.shape[0];
	columns = pBuff.shape[1];
	bytes   = pBuff.len;
	//std::cout << "mt nums " << rows << " " << columns << " " << bytes << "\n";
    // allocate xs_data pointer arrays
    xs_MT_numbers      = new unsigned  [rows];
    // check to make sure bytes *= elements
    assert(bytes==rows*4);
    // copy python buffer contents to pointer
    memcpy( xs_MT_numbers,   pBuff.buf , bytes );
    // cudaallocate device memory now that we know the size!
    cudaMalloc(&d_xs_MT_numbers,bytes);
    // release python variable to free memory
    Py_DECREF(call_result);



    // mt number total vector
    call_string = PyString_FromString("_get_MT_numbers_total_pointer");
	call_result = PyObject_CallMethodObjArgs(xsdat_instance, call_string, NULL);
	Py_DECREF(call_string);
	if (PyObject_CheckBuffer(call_result)){
		PyObject_GetBuffer(call_result, &pBuff,PyBUF_ND);
	}
	else{
		PyErr_Print();
        fprintf(stderr, "Returned object does not support buffer interface\n");
        return;
	}

    //
    // get and copy unionized totals vector
    //
	rows    = pBuff.shape[0];
	columns = pBuff.shape[1];
	bytes   = pBuff.len;
	//std::cout << "totals " << rows << " " << columns << " " << bytes << "\n";
    // allocate xs_data pointer arrays
    xs_MT_numbers_total      = new unsigned  [rows];
    // check to make sure bytes *= elements
    assert(bytes==rows*4);
    // copy python buffer contents to pointer
    memcpy( xs_MT_numbers_total,   pBuff.buf , bytes );
    // cudaallocate device memory now that we know the size!
    cudaMalloc(&d_xs_MT_numbers_total,bytes);
    // release python variable to free memory
    Py_DECREF(call_result);


    // lengths vector
    call_string = PyString_FromString("_get_length_numbers_pointer");
	call_result = PyObject_CallMethodObjArgs(xsdat_instance, call_string, NULL);
	Py_DECREF(call_string);
	if (PyObject_CheckBuffer(call_result)){
		PyObject_GetBuffer(call_result, &pBuff,PyBUF_ND);
	}
	else{
		PyErr_Print();
        fprintf(stderr, "Returned object does not support buffer interface\n");
        return;
	}

	//
	// get and copy lengths vector
	//
	rows    = pBuff.shape[0];
	columns = pBuff.shape[1];
	bytes   = pBuff.len;
	//std::cout << "lengths " << rows << " " << columns << " " << bytes << "\n";
	// allocate xs_data pointer arrays
	xs_length_numbers     = new unsigned  [rows];
	// check to make sure bytes *= elements
	assert(bytes==rows*4);
	// copy python buffer contents to pointer
	memcpy( xs_length_numbers,   pBuff.buf , bytes );
	// cudaallocate device memory now that we know the size!
	cudaMalloc(&d_xs_length_numbers,bytes);
	// release python variable to free memory
	Py_DECREF(call_result);





	// AWR vector
	call_string = PyString_FromString("_get_awr_pointer");
	call_result = PyObject_CallMethodObjArgs(xsdat_instance, call_string, NULL);
	Py_DECREF(call_string);
	if (PyObject_CheckBuffer(call_result)){
		PyObject_GetBuffer(call_result, &pBuff,PyBUF_ND);
	}
	else{
		PyErr_Print();
	    fprintf(stderr, "Returned object does not support buffer interface\n");
	    return;
	}

	//
	// get and copy AWR vector
	//
	rows    = pBuff.shape[0];
	columns = pBuff.shape[1];
	bytes   = pBuff.len;
	//std::cout << "lengths " << rows << " " << columns << " " << bytes << "\n";
	// allocate xs_data pointer arrays
	awr_list     = new float  [rows];
	// check to make sure bytes *= elements
	assert(bytes==rows*4);
	// copy python buffer contents to pointer
	memcpy( awr_list,   pBuff.buf , bytes );
	// cudaallocate device memory now that we know the size!
	cudaMalloc(&d_awr_list,bytes);
	// release python variable to free memory
	Py_DECREF(call_result);



	// Q vector
	call_string = PyString_FromString("_get_Q_pointer");
	call_result = PyObject_CallMethodObjArgs(xsdat_instance, call_string, NULL);
	Py_DECREF(call_string);
	if (PyObject_CheckBuffer(call_result)){
		PyObject_GetBuffer(call_result, &pBuff,PyBUF_ND);
	}
	else{
		PyErr_Print();
	    fprintf(stderr, "Returned object does not support buffer interface\n");
	    return;
	}

	//
	// get and copy Q vector
	//
	rows    = pBuff.shape[0];
	columns = pBuff.shape[1];
	bytes   = pBuff.len;
	//std::cout << "lengths " << rows << " " << columns << " " << bytes << "\n";
	// allocate xs_data pointer arrays
	xs_data_Q     = new float  [rows];
	// check to make sure bytes *= elements
	assert(bytes==rows*4);
	// copy python buffer contents to pointer
	memcpy( xs_data_Q,   pBuff.buf , bytes );
	// cudaallocate device memory now that we know the size!
	cudaMalloc(&d_xs_data_Q,bytes);
	// release python variable to free memory
	Py_DECREF(call_result);









    ////////////////////////////////////
    // do scattering stuff
    ////////////////////////////////////

    float * temp = new float [128];
	for(int g=0;g<128;g++){temp[g]=123456789;}
	unsigned vlen;

    //ALLOCATE THE ARRAYS.
    xs_data_scatter      = new float* [MT_rows*MT_columns];
    xs_data_energy       = new float* [MT_rows*MT_columns];
    xs_data_scatter_host = new float* [MT_rows*MT_columns];
    xs_data_energy_host  = new float* [MT_rows*MT_columns];
    cudaMalloc(&d_xs_data_scatter,MT_rows*MT_columns*sizeof(float));
    cudaMalloc(&d_xs_data_energy, MT_rows*MT_columns*sizeof(float));
    // python variables for arguments
    PyObject 	*E_obj, *MT_obj, *tope_obj;
    PyObject 	*cdf_vector_obj, *mu_vector_obj , *vector_length_obj, *nextE_obj; 
    PyObject 	*obj_list;
    Py_buffer 	muBuff, cdfBuff;
    float 		*this_pointer,*cuda_pointer;
    float  		nextE;
    float       this_energy;
    unsigned	this_MT, this_tope, vector_length_L;
    unsigned 	vector_length;
    unsigned 	 muRows,  muColumns,  muBytes;
    unsigned 	cdfRows, cdfColumns, cdfBytes;

    //set total cross sections to NULL
    for (int j=0 ; j<1*xs_length_numbers[0] ; j++){  //start after the total xs vectors
    	for (int k=0 ; k<MT_rows ; k++){
    		xs_data_scatter     [k*MT_columns + j] = NULL;
			xs_data_scatter_host[k*MT_columns + j] = NULL;
		}
	}

    // do the rest of the MT numbers
    for (int j=1*xs_length_numbers[0] ; j<MT_columns ; j++){  //start after the total xs vectors
    	
    	// get MT number and isotope
    	this_MT     = xs_MT_numbers[j];  //adjust for the first total xs when accessing this array
    	for (int z=0 ; z<xs_length_numbers[0] ; z++){ // see what isotope were in
    		if(j< (xs_MT_numbers_total[z] + 1*xs_length_numbers[0] )){
    			this_tope=z;
    			break;
    		}
    	}
    	std::cout << "tope = " << this_tope << " MT = " << this_MT << "\n";

    	for (int k=0 ; k<MT_rows ; k++){

    		// get this energy point
    		this_energy = xs_data_main_E_grid[k];

    		//SHOULD TRY TO FIX THIS CRAP FROM PRECISION LOSS
    		if (k==0){this_energy=this_energy*1.0000001;}
    		//////

    		//std::cout << "j,k = " << j << ", " << k  <<" this_energy " << this_energy << " this_MT " << this_MT << " this_tope " << this_tope << "\n";
    		//printf("this_energy = %12.10E\n",this_energy);

    		// call cross_section_data instance to get buffer
    		E_obj       = PyFloat_FromDouble (this_energy);
    		MT_obj      = PyInt_FromLong     (this_MT);
    		tope_obj    = PyInt_FromLong     (this_tope);
    		call_string = PyString_FromString("_get_scattering_data");
			obj_list    = PyObject_CallMethodObjArgs(xsdat_instance, call_string, tope_obj, MT_obj, E_obj, NULL);
			PyErr_Print();

			// get objects in the returned list
			nextE_obj  			= PyList_GetItem(obj_list,0);
			vector_length_obj 	= PyList_GetItem(obj_list,1);
			mu_vector_obj 		= PyList_GetItem(obj_list,2);
			cdf_vector_obj 		= PyList_GetItem(obj_list,3);
			PyErr_Print();

			// expand list to c variables
			nextE         = PyFloat_AsDouble(nextE_obj);
			vector_length = PyInt_AsLong    (vector_length_obj);
			PyErr_Print();

			//std::cout << " nextE " << nextE << " vector_length " << vector_length << "\n";

			if(vector_length==0){
				//std::cout << "set as NULL \n";
				xs_data_scatter     [k*MT_columns + j] = NULL;
				xs_data_scatter_host[k*MT_columns + j] = NULL;
				// free python variables
				//std::cout << "freeing python stuff... ";
				//Py_DECREF(call_string);
				//Py_DECREF(E_obj);
				//Py_DECREF(MT_obj); 
				//Py_DECREF(tope_obj);
				//Py_DECREF(vector_length_obj); 
				//Py_DECREF(nextE_obj);
				//Py_DECREF(obj_list);
				//std::cout << "done. \n";
				PyErr_Print();
			}
			else{
				//std::cout << "getting buffer...\n";
				// get data buffer from numpy array
				if (PyObject_CheckBuffer(mu_vector_obj) & PyObject_CheckBuffer(cdf_vector_obj)){
					PyObject_GetBuffer( mu_vector_obj,  &muBuff, PyBUF_ND);
					PyObject_GetBuffer(cdf_vector_obj, &cdfBuff, PyBUF_ND);
					PyErr_Print();
				}
				else{
					PyErr_Print();
    			    fprintf(stderr, "Returned object does not support buffer interface\n");
    			    return;
				}
	
				// shape info
				muRows     =  muBuff.shape[0];
				muColumns  =  muBuff.shape[1];
				muBytes    =  muBuff.len;
				cdfRows    = cdfBuff.shape[0];
				cdfColumns = cdfBuff.shape[1];
				cdfBytes   = cdfBuff.len;
	
				//make sure every is ok
				assert(muRows==cdfRows);
				assert(muColumns==cdfColumns);
				assert(muBytes==cdfBytes);
	
				//allocate pointer, write into array
				//for cuda too
				//std::cout << "muRows, cdfRows = " << muRows << ", " << cdfRows << "\n";
				//std::cout << "new pointer size = " << muRows+cdfRows+1 << "\n";
				this_pointer = new float [muRows+cdfRows+1];
				cudaMalloc(&cuda_pointer,(muRows+cdfRows+1)*sizeof(float));
				total_bytes_scatter += (muRows+cdfRows+1)*sizeof(float);  // add to total count
				//std::cout <<"here "<< k*MT_columns + j <<"\n";
				xs_data_scatter     [k*MT_columns + j] = this_pointer;
				xs_data_scatter_host[k*MT_columns + j] = cuda_pointer;
	
				//copy data from python buffer to pointer in array
				//std::cout <<this_pointer<<" "<<&this_pointer[1]<<" "<<&this_pointer[1+muRows] << "\n";
				memcpy(this_pointer, 			&muRows,   		     sizeof(unsigned));  // to first position
				memcpy(&this_pointer[1],		muBuff.buf,  	 muRows*sizeof(float));     // to len bytes after
				memcpy(&this_pointer[1+muRows],	cdfBuff.buf, 	cdfRows*sizeof(float));     // to len bytes after that
				//COPY TO CUDA POINTER
				//memcpy(&vlen,&this_pointer[0],4);
				//std::cout << vlen << "\n";
				//std::cout << this_pointer[1] << " " << this_pointer[2] << " " << this_pointer[3] << "\n";
				//std::cout << this_pointer[4] << " " << this_pointer[5] << " " << this_pointer[6] << "\n";
				//cudaMemcpy(cuda_pointer,this_pointer,(muRows+cdfRows+1)*sizeof(float),cudaMemcpyHostToDevice);
				//cudaMemcpy(temp,cuda_pointer,(2*vlen+1)*sizeof(float),cudaMemcpyDeviceToHost);
				//memcpy(&vlen,&temp[0],4);
				//std::cout << vlen << "\n";
				//std::cout << temp[1] << " " << temp[2] << " " << temp[3] << "\n";
				//std::cout << temp[4] << " " << temp[5] << " " << temp[6] << "\n";
			
				//std::cout << "done copying\n";

				//std::cout << muRows << " " << muColumns << " " << muBytes << this_energy <<"\n";
				//std::cout << this_pointer[0] << "\n";
				//std::cout << this_pointer[1] << " " << this_pointer[2] << " " << this_pointer[3] << "\n";
				//std::cout << this_pointer[4] << " " << this_pointer[5] << " " << this_pointer[6] << "\n";
				
				// free python variables
				//Py_DECREF(call_string);
				//Py_DECREF(E_obj);
				//Py_DECREF(MT_obj); 
				//Py_DECREF(tope_obj);
				//Py_DECREF(cdf_vector_obj); 
				//Py_DECREF(mu_vector_obj);
				//Py_DECREF(vector_length_obj); 
				//Py_DECREF(nextE_obj);
				//Py_DECREF(obj_list);
				PyErr_Print();
			}

			// replicate this pointer into array until nextE
			// for cuda too
			//std::cout << "replicating...\n";
			//std::cout << "k now = " << xs_data_main_E_grid[k] << " nextE = " << nextE << "\n";
			if (k < (MT_rows-1) ){
				while(xs_data_main_E_grid[k+1]<nextE){
					//std::cout << k << " " << MT_rows*MT_columns << " " << xs_data_main_E_grid[k+1] << "\n";
					xs_data_scatter     [(k+1)*MT_columns + j] = this_pointer;
					xs_data_scatter_host[(k+1)*MT_columns + j] = cuda_pointer;
					k++;
				}
			}

			this_pointer = NULL;
			cuda_pointer = NULL;

		}
	}



    ////////////////////////////////////
    // do energy stuff
    ////////////////////////////////////

    //set total cross sections to NULL
    for (int j=0 ; j<1*xs_length_numbers[0] ; j++){  //start after the total xs vectors
    	for (int k=0 ; k<MT_rows ; k++){
    		xs_data_energy     [k*MT_columns + j] = NULL;
			xs_data_energy_host[k*MT_columns + j] = NULL;
		}
	}

    // do the rest of the MT numbers
    for (int j=1*xs_length_numbers[0] ; j<MT_columns ; j++){  //start after the total xs vectors
    	
    	// get MT number and isotope
    	this_MT     = xs_MT_numbers[j];  //adjust for the first total xs when accessing this array
    	for (int z=0 ; z<xs_length_numbers[0] ; z++){ // see what isotope were in
    		if(j< (xs_MT_numbers_total[z] + 1*xs_length_numbers[0] )){
    			this_tope=z;
    			break;
    		}
    	}
    	std::cout << "tope = " << this_tope << " MT = " << this_MT << "\n";

    	for (int k=0 ; k<MT_rows ; k++){

    		// get this energy point
    		this_energy = xs_data_main_E_grid[k];

    		//SHOULD TRY TO FIX THIS CRAP FROM PRECISION LOSS
    		if (k==0){this_energy=this_energy*1.0000001;}
    		//////

    		//std::cout << "j,k = " << j << ", " << k  <<" this_energy " << this_energy << " this_MT " << this_MT << " this_tope " << this_tope << "\n";
    		//printf("this_energy = %12.10E\n",this_energy);

    		// call cross_section_data instance to get buffer
    		E_obj       = PyFloat_FromDouble (this_energy);
    		MT_obj      = PyInt_FromLong     (this_MT);
    		tope_obj    = PyInt_FromLong     (this_tope);
    		call_string = PyString_FromString("_get_energy_data");
			obj_list    = PyObject_CallMethodObjArgs(xsdat_instance, call_string, tope_obj, MT_obj, E_obj, NULL);
			PyErr_Print();

			// get objects in the returned list
			nextE_obj  			= PyList_GetItem(obj_list,0);
			vector_length_obj 	= PyList_GetItem(obj_list,1);
			mu_vector_obj 		= PyList_GetItem(obj_list,2);
			cdf_vector_obj 		= PyList_GetItem(obj_list,3);
			PyErr_Print();

			// expand list to c variables
			nextE         = PyFloat_AsDouble(nextE_obj);
			vector_length = PyInt_AsLong    (vector_length_obj);
			PyErr_Print();

			//std::cout << " nextE " << nextE << " vector_length " << vector_length << "\n";

			if(vector_length==0){
				//std::cout << "set as NULL \n";
				xs_data_energy     [k*MT_columns + j] = NULL;
				xs_data_energy_host[k*MT_columns + j] = NULL;
				// free python variables
				//std::cout << "freeing python stuff... ";
				//Py_DECREF(call_string);
				//Py_DECREF(E_obj);
				//Py_DECREF(MT_obj); 
				//Py_DECREF(tope_obj);
				//Py_DECREF(vector_length_obj); 
				//Py_DECREF(nextE_obj);
				//Py_DECREF(obj_list);
				//std::cout << "done. \n";
				PyErr_Print();
			}
			else{
				//std::cout << "getting buffer...\n";
				// get data buffer from numpy array
				if (PyObject_CheckBuffer(mu_vector_obj) & PyObject_CheckBuffer(cdf_vector_obj)){
					PyObject_GetBuffer( mu_vector_obj,  &muBuff, PyBUF_ND);
					PyObject_GetBuffer(cdf_vector_obj, &cdfBuff, PyBUF_ND);
					PyErr_Print();
				}
				else{
					PyErr_Print();
    			    fprintf(stderr, "Returned object does not support buffer interface\n");
    			    return;
				}
	
				// shape info
				muRows     =  muBuff.shape[0];
				muColumns  =  muBuff.shape[1];
				muBytes    =  muBuff.len;
				cdfRows    = cdfBuff.shape[0];
				cdfColumns = cdfBuff.shape[1];
				cdfBytes   = cdfBuff.len;
	
				//std::cout << muRows << " " << muColumns << " " << muBytes << "\n";
	
				//make sure every is ok
				assert(muRows==cdfRows);
				assert(muColumns==cdfColumns);
				assert(muBytes==cdfBytes);
	
				//allocate pointer, write into array
				//for cuda too
				//std::cout << "muRows, cdfRows = " << muRows << ", " << cdfRows << "\n";
				//std::cout << "new pointer size = " << muRows+cdfRows+1 << "\n";
				this_pointer = new float [muRows+cdfRows+1];
				cudaMalloc(&cuda_pointer,sizeof(float)*muRows+cdfRows+1);
				total_bytes_energy += 2*cdfBytes+4;  // add to total count
				//std::cout <<"here "<< k*MT_columns + j <<"\n";
				xs_data_energy     [k*MT_columns + j] = this_pointer;
				xs_data_energy_host[k*MT_columns + j] = cuda_pointer;
	
				//copy data from python buffer to pointer in array
				//std::cout <<this_pointer<<" "<<&this_pointer[1]<<" "<<&this_pointer[1+muRows] << "\n";
				memcpy(this_pointer, 			&muRows,   		sizeof(unsigned));  // to first position
				memcpy(&this_pointer[1],		muBuff.buf,  	muRows*sizeof(float));     // to len bytes after
				memcpy(&this_pointer[1+muRows],	cdfBuff.buf, 	cdfRows*sizeof(float));     // to len bytes after that
				//std::cout << "done copying\n";
				
				// free python variables
				//Py_DECREF(call_string);
				//Py_DECREF(E_obj);
				//Py_DECREF(MT_obj); 
				//Py_DECREF(tope_obj);
				//Py_DECREF(cdf_vector_obj); 
				//Py_DECREF(mu_vector_obj);
				//Py_DECREF(vector_length_obj); 
				//Py_DECREF(nextE_obj);
				//Py_DECREF(obj_list);
				PyErr_Print();
			}

			// replicate this pointer into array until nextE
			// for cuda too
			//std::cout << "replicating...\n";
			//std::cout << "k now = " << xs_data_main_E_grid[k] << " nextE = " << nextE << "\n";
			if (k < (MT_rows-1) ){
				while(xs_data_main_E_grid[k+1]<nextE){
					//std::cout << k << " " << MT_rows*MT_columns << " " << xs_data_main_E_grid[k+1] << "\n";
					xs_data_energy     [(k+1)*MT_columns + j] = this_pointer;
					xs_data_energy_host[(k+1)*MT_columns + j] = cuda_pointer;
					k++;
				}
			}

			this_pointer = NULL;
			cuda_pointer = NULL;

		}
	}



    Py_Finalize();



    //pass awr pointer to geometry object, make the number density table, copy pointers back
    problem_geom.awr_list = awr_list;
    problem_geom.make_material_table();
    problem_geom.get_material_table(&n_materials,&n_isotopes,&material_list,&isotope_list,&number_density_matrix);  

    assert(n_isotopes == xs_length_numbers[0]);

    //do cudamalloc for these arrays
    cudaMalloc(&d_material_list , 			n_materials*sizeof(unsigned) );
    cudaMalloc(&d_isotope_list , 			n_isotopes*sizeof(unsigned) );
    cudaMalloc(&d_number_density_matrix , 	n_materials*n_isotopes*sizeof(unsigned) );

}
void whistory::print_xs_data(){  // 0=isotopes, 1=main E points, 2=total numer of reaction channels, 3=matrix E points, 4=angular cosine points, 5=outgoing energy points
	unsigned dsum = 0;
	printf("\e[1;32m%-6s\e[m \n","Cross section data info:");
	std::cout << "--- Bytes ---" << "\n";
	std::cout << "  xs_length_numbers:        " << 6 															*sizeof(unsigned) 		<< "\n";  dsum += (6 															*sizeof(unsigned) );
	std::cout << "  xs_MT_numbers_total:      " << xs_length_numbers[0]											*sizeof(unsigned) 		<< "\n";  dsum += (xs_length_numbers[0]											*sizeof(unsigned) );
	std::cout << "  xs_MT_numbers:            " << (xs_length_numbers[2]+xs_length_numbers[0])					*sizeof(unsigned) 		<< "\n";  dsum += (xs_length_numbers[2]											*sizeof(unsigned) );
	std::cout << "  xs_data_main_E_grid:      " << xs_length_numbers[1]											*sizeof(float)	  		<< "\n";  dsum += (xs_length_numbers[1]											*sizeof(float)	  );
	std::cout << "  xs_data_MT:               " << MT_rows*MT_columns*sizeof(float)		<< "\n";  dsum += (MT_rows*MT_columns)*sizeof(float);
	std::cout << "  xs_data_scatter_pointers: " << MT_rows*MT_columns*sizeof(float)		<< "\n";  dsum += (MT_rows*MT_columns)*sizeof(float);
	std::cout << "  xs_data_energy_pointers:  " << MT_rows*MT_columns*sizeof(float)		<< "\n";  dsum += (MT_rows*MT_columns)*sizeof(float);
	std::cout << "  scatter data:             " << total_bytes_scatter																	<< "\n";  dsum += (total_bytes_scatter);
	std::cout << "  energy data:              " << total_bytes_energy																	<< "\n";  dsum += (total_bytes_energy);
	std::cout << "  TOTAL:                    " << dsum << " bytes \n";
	std::cout << "  TOTAL:                    " << dsum/1048576 << " MB \n";
}
void whistory::write_xs_data(std::string filename){

	std::cout << "\e[1;32m" << "Writing xs_data to " << filename << "... ";

	std::string this_name;
	// write MT array
	this_name = filename + ".MTarray";
	FILE* xsfile = fopen(this_name.c_str(),"w");
	for (int k=0;k<MT_rows;k++){
		for(int j=0;j<MT_columns;j++){
			fprintf(xsfile,"% 10.8E ",xs_data_MT[k*MT_columns+j]);
		}
		fprintf(xsfile,"\n");
	}
	fclose(xsfile);

	// write unionized E grid
	this_name = filename + ".Egrid";
	xsfile = fopen(this_name.c_str(),"w");
	for (int k=0;k<MT_rows;k++){
		fprintf(xsfile,"%10.8E\n",xs_data_main_E_grid[k]);
	}
	fclose(xsfile);

	// write MT number array
	this_name = filename + ".MTnums";
	xsfile = fopen(this_name.c_str(),"w");
	for(int j=0;j<MT_columns;j++){
		fprintf(xsfile,"%u\n",xs_MT_numbers[j]);
	}
	fclose(xsfile);

	std::cout << "Done." << "\e[m \n";

}
void whistory::print_pointers(){
	std::cout << "\e[1;32m" << "Pointer Info:" << "\e[m \n";
	std::cout << "--- HOST ---" << "\n";
	std::cout << "  space:               " <<   space   << "\n";
	std::cout << "  E:                   " <<   E       << "\n";
	std::cout << "  Q:                   " <<   Q       << "\n";
	std::cout << "  rn_bank:             " <<   rn_bank << "\n";
	std::cout << "  cellnum:             " <<   cellnum << "\n";
	std::cout << "  matnum:              " <<   matnum  << "\n";
	std::cout << "  isonum:              " <<   isonum  << "\n";
	std::cout << "  rxn:                 " <<   rxn     << "\n";
	std::cout << "  done:                " <<   done    << "\n";
	std::cout << "  yield:               " <<   yield   << "\n";
	std::cout << "  xs_length_numbers:   " << xs_length_numbers   << "\n"; 
	std::cout << "  xs_MT_numbers_total: " << xs_MT_numbers_total << "\n";
	std::cout << "  xs_MT_numbers:       " << xs_MT_numbers       << "\n";
	std::cout << "  xs_data_MT:          " << xs_data_MT          << "\n";
	std::cout << "  xs_data_main_E_grid: " << xs_data_main_E_grid << "\n";
	std::cout << "--- DEVICE ---" << "\n";
	std::cout << "d_space:               " << d_space   << "\n";
	std::cout << "d_E:                   " << d_E       << "\n";
	std::cout << "d_Q:                   " << d_Q       << "\n";
	std::cout << "d_rn_bank:             " << d_rn_bank << "\n";
	std::cout << "d_cellnum:             " << d_cellnum << "\n";
	std::cout << "d_matnum:              " << d_matnum  << "\n";
	std::cout << "d_isonum:              " << d_isonum  << "\n";
	std::cout << "d_rxn:                 " << d_rxn     << "\n";
	std::cout << "d_done:                " << d_done    << "\n";
	std::cout << "d_yield:               " << d_yield   << "\n";
	std::cout << "d_xs_length_numbers:   " << d_xs_length_numbers   << "\n"; 
	std::cout << "d_xs_MT_numbers_total: " << d_xs_MT_numbers_total << "\n";
	std::cout << "d_xs_MT_numbers:       " << d_xs_MT_numbers       << "\n";
	std::cout << "d_xs_data_MT:          " << d_xs_data_MT          << "\n";
	std::cout << "d_xs_data_main_E_grid: " << d_xs_data_main_E_grid << "\n";
}
void whistory::trace(unsigned type){

	optix_obj.trace(type);

}
void whistory::print_materials_table(){

	problem_geom.print_materials_table();

}
void whistory::sample_fissile_points(){

	std::cout << "\e[1;32m" << "Sampling initial fissile starting points uniformly... " << "\e[m \n";

	// iterate
	unsigned current_index = 0;
	unsigned valid_N = 0;
	while (current_index < N){
		
		// advance RN bank
		curandGenerateUniform( rand_gen , d_rn_bank , N*RNUM_PER_THREAD );
		
		// set uniformly random positions on GPU
		set_positions_rand ( blks, NUM_THREADS, N , RNUM_PER_THREAD, d_space , d_rn_bank, outer_cell_dims);
		
		//run OptiX to get cell number, set as a hash run for fissile, writes 1/0 to matnum, trace_type=4
		trace(3);
		
		// compact
		res = cudppCompact(compactplan, d_valid_result, (size_t*)d_valid_N , d_remap , d_matnum , N);
		if (res != CUDPP_SUCCESS){fprintf(stderr, "Error in compacting\n");exit(-1);}

		//copy in new values, keep track of index, copies positions and direction
		copy_points(blks, NUM_THREADS, N, d_valid_N, current_index, d_valid_result, d_fissile_points, d_space); 
		
		// copy back and add
		cudaMemcpy( &valid_N, d_valid_N, 1*sizeof(unsigned), cudaMemcpyDeviceToHost);
		current_index += valid_N;
		
		// print how far along we are
		std::cout << (float)current_index/(float)N*100.0 <<" \% done\r";

		if((float)current_index/(float)N > 1){
			std::cout << "100.00 \% done     \n";
		} 
	}

	std::cout << "Copying to starting points...\n";

	cudaMemcpy(d_space,d_fissile_points,N*sizeof(source_point),cudaMemcpyDeviceToDevice);
	//cudaFree(d_fissile_points);

	std::cout << "Done.\n";

	//write starting positions to file
	cudaMemcpy(space,d_space,N*sizeof(source_point),cudaMemcpyDeviceToHost);
	FILE* positionsfile = fopen("starting_positions","w");
	for(int k=0;k<N;k++){
		fprintf(positionsfile,"% 10.8E % 10.8E % 10.8E % 10.8E % 10.8E % 10.8E\n",space[k].x,space[k].y,space[k].z,space[k].xhat,space[k].yhat,space[k].zhat);
	}
	fclose(positionsfile);

	// advance RN bank
	curandGenerateUniform( rand_gen , d_rn_bank , N*RNUM_PER_THREAD );

}
void whistory::converge(unsigned num_cycles){

	float keff = 0.0;
	unsigned completed_hist = 0;
	unsigned current_fission_index = 0;

	//intital samples
	sample_fissile_points();

	// print
	std::cout << "\e[1;32m" << "--- Running "<< num_cycles << " INACTIVE CYCLES --- " << "\e[m \n";

	//set fission spectra
	sample_fission_spectra(blks, NUM_THREADS,N,d_rn_bank,d_E);

	//make directions isotropic
	sample_isotropic_directions(blks, NUM_THREADS, N , RNUM_PER_THREAD, d_space , d_rn_bank);

	for(int iteration = 0 ; iteration<num_cycles ; iteration++){

		while(completed_hist<N){
	
			//find the main E grid index
			find_E_grid_index(blks, NUM_THREADS, N, xs_length_numbers[1], d_xs_data_main_E_grid, d_E, d_index, d_done);

			// find what material we are in
			trace(2);
	
			// run macroscopic kernel to find interaction length and reaction isotope
			macroscopic( blks, NUM_THREADS, N, n_isotopes, MT_columns, d_space, d_isonum, d_index, d_matnum, d_xs_data_main_E_grid, d_rn_bank, d_E, d_xs_data_MT , d_number_density_matrix, d_done);
	
			// run optix to detect the nearest surface and move particle there
			trace(1);
	
			// run microscopic kernel to find reaction type
			microscopic(blks, NUM_THREADS, N, n_isotopes, MT_columns, d_isonum, d_index, d_xs_data_main_E_grid, d_rn_bank, d_E, d_xs_data_MT , d_xs_MT_numbers_total, d_xs_MT_numbers, d_xs_data_Q, d_rxn, d_Q, d_done);
	
			// concurrent calls to do escatter/iscatter/abs/fission, serial execution for now :(
			escatter( blks,  NUM_THREADS,   N, RNUM_PER_THREAD, d_isonum, d_index, d_rn_bank, d_E, d_space, d_rxn, d_awr_list, d_done, d_xs_data_scatter);
			iscatter( blks,  NUM_THREADS,   N, RNUM_PER_THREAD, d_isonum, d_index, d_rn_bank, d_E, d_space, d_rxn, d_awr_list, d_Q, d_done, d_xs_data_scatter);
			fission(  blks,  NUM_THREADS,   N, RNUM_PER_THREAD, d_rxn , d_yield , d_rn_bank, d_done);
			absorb(   blks,  NUM_THREADS,   N, d_rxn , d_done);
	
			// update RNGs
			update_RNG();

			// get how many histories are complete
			completed_hist = reduce_done();

			//std::cout << completed_hist << "/" << N << " histories complete\n";
		}

		//reduce yield
		keff = reduce_yield();

		std::cout << "DISCARDED cycle keff = " << keff << "\n";

		//reset cycle, adding new fission points to starting points
		current_fission_index = reset_cycle(current_fission_index);
		completed_hist = 0;

	}

		
}
unsigned whistory::reset_cycle(unsigned current_fission_index){

	unsigned valid_N;

	// do fissile query by setting rnx=16-18 as valid
	make_mask(blks, NUM_THREADS, N, d_mask, d_rxn, 18, 18);  // add ones for all fission numbers

	// copy fission points to fission points vector
	res = cudppCompact(compactplan, d_valid_result, (size_t*)d_valid_N , d_remap , d_mask , N);
	if (res != CUDPP_SUCCESS){fprintf(stderr, "Error in compacting\n");exit(-1);}  // compact
	copy_points(blks, NUM_THREADS, N, d_valid_N, current_fission_index, d_valid_result, d_fissile_points, d_space);   //copy in new values, keep track of index, copies positions and direction
	cudaMemcpy(d_space,d_fissile_points,N*sizeof(source_point),cudaMemcpyDeviceToDevice);

	// copy back and add, wrap to beginning 
	cudaMemcpy( &valid_N, d_valid_N, 1*sizeof(unsigned), cudaMemcpyDeviceToHost);
	current_fission_index += valid_N;
	if(current_fission_index>=N){current_fission_index = current_fission_index - N;}

	//set fission spectra
	sample_fission_spectra(blks, NUM_THREADS,N,d_rn_bank,d_E);

	//make directions isotropic
	sample_isotropic_directions(blks, NUM_THREADS, N , RNUM_PER_THREAD, d_space , d_rn_bank);

	// rest run arrays
	cudaMemcpy( d_Q,    		Q,			N*sizeof(float),		cudaMemcpyHostToDevice );
	cudaMemcpy( d_done,			done,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );
	cudaMemcpy( d_cellnum,		cellnum,	N*sizeof(unsigned),		cudaMemcpyHostToDevice );
	cudaMemcpy( d_matnum,		matnum,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );
	cudaMemcpy( d_isonum,		isonum,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );
	cudaMemcpy( d_yield,		yield,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );
	cudaMemcpy( d_rxn,			rxn,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );

	// return updated fission index
	return current_fission_index;

}
void whistory::reset_fixed(){

	// rest run arrays
	cudaMemcpy( d_space,		space,		N*sizeof(source_point),	cudaMemcpyHostToDevice );
	cudaMemcpy( d_Q,    		Q,			N*sizeof(float),		cudaMemcpyHostToDevice );
	cudaMemcpy( d_done,			done,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );
	cudaMemcpy( d_cellnum,		cellnum,	N*sizeof(unsigned),		cudaMemcpyHostToDevice );
	cudaMemcpy( d_matnum,		matnum,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );
	cudaMemcpy( d_isonum,		isonum,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );
	cudaMemcpy( d_yield,		yield,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );
	cudaMemcpy( d_rxn,			rxn,		N*sizeof(unsigned),		cudaMemcpyHostToDevice );

	//set fission spectra
	sample_fission_spectra(blks, NUM_THREADS,N,d_rn_bank,d_E);

	//make directions isotropic
	sample_isotropic_directions(blks, NUM_THREADS, N , RNUM_PER_THREAD, d_space , d_rn_bank);

}
void whistory::run(unsigned num_cycles){

	std::cout << "\e[1;32m" << "--- Running "<< num_cycles << " ACTIVE CYCLES --- " << "\e[m \n";

	float keff = 0.0;
	float keff_cycle = 0.0;
	float it = 0.0;
	unsigned completed_hist = 0;
	unsigned current_fission_index = 0;
	float runtime = get_time();

	//set mask to ones
	//cudaMemcpy(d_mask,ones,n_tally*sizeof(unsigned),cudaMemcpyHostToDevice);

	//set fission spectra
	sample_fission_spectra(blks, NUM_THREADS,N,d_rn_bank,d_E);

	//make directions isotropic
	sample_isotropic_directions(blks, NUM_THREADS, N , RNUM_PER_THREAD, d_space , d_rn_bank);

	for(int iteration = 0 ; iteration<num_cycles ; iteration++){

		while(completed_hist<N){
	
			//find the main E grid index
			find_E_grid_index(blks, NUM_THREADS, N, xs_length_numbers[1], d_xs_data_main_E_grid, d_E, d_index, d_done);

			// find what material we are in
			trace(2);
	
			// run macroscopic kernel to find interaction length and reaction isotope
			macroscopic( blks, NUM_THREADS, N, n_isotopes, MT_columns, d_space, d_isonum, d_index, d_matnum, d_xs_data_main_E_grid, d_rn_bank, d_E, d_xs_data_MT , d_number_density_matrix, d_done);
	
			// run tally kernel to compute spectra
			make_mask(blks, NUM_THREADS, N, d_mask, d_cellnum, tally_cell, tally_cell);
			tally_spec( blks,  NUM_THREADS,   N,  n_tally,  d_space, d_E, d_tally_score, d_tally_count, d_done, d_mask);
	
			// run optix to detect the nearest surface and move particle there
			trace(1);
	
			// run microscopic kernel to find reaction type
			microscopic(blks, NUM_THREADS, N, n_isotopes, MT_columns, d_isonum, d_index, d_xs_data_main_E_grid, d_rn_bank, d_E, d_xs_data_MT , d_xs_MT_numbers_total, d_xs_MT_numbers, d_xs_data_Q, d_rxn, d_Q, d_done);
	
			// concurrent calls to do escatter/iscatter/abs/fission, serial execution for now :(
			escatter( blks,  NUM_THREADS,   N, RNUM_PER_THREAD, d_isonum, d_index, d_rn_bank, d_E, d_space, d_rxn, d_awr_list, d_done, d_xs_data_scatter);
			iscatter( blks,  NUM_THREADS,   N, RNUM_PER_THREAD, d_isonum, d_index, d_rn_bank, d_E, d_space, d_rxn, d_awr_list, d_Q, d_done, d_xs_data_scatter);
			fission(  blks,  NUM_THREADS,   N, RNUM_PER_THREAD, d_rxn , d_yield , d_rn_bank, d_done);
			absorb(   blks,  NUM_THREADS,   N, d_rxn , d_done);
	
			// update RNGs
			update_RNG();

			// get how many histories are complete
			completed_hist = reduce_done();

			//std::cout << completed_hist << "/" << N << " histories complete\n";
			//if((N-completed_hist)<=150){print_histories( blks,  NUM_THREADS,  N, d_isonum, d_rxn, d_space, d_E, d_done);}
		}

		//reduce yield
		keff_cycle = reduce_yield();
		if (iteration == 0){
			keff  = keff_cycle;
		}
		else {
			it = (float) iteration;
			keff  = (it/(it+1)) * keff + (1/(it+1)) * keff_cycle;
		}

		std::cout << "Cumulative keff = "<< keff << ", ACTIVE cycle " << iteration << ", keff = " << keff_cycle << "\n";

		// reset cycle, adding new fission points to starting points
		//current_fission_index = reset_cycle(current_fission_index);
		reset_fixed();
		completed_hist = 0;

	}

	runtime = get_time() - runtime;
	if(runtime>60.0){
		std::cout << "RUNTIME = " << runtime/60.0 << " minutes.\n";
	}
	else{
		std::cout << "RUNTIME = " << runtime << " seconds.\n";
	}

}
void whistory::write_tally(unsigned tallynum, std::string filename){

	//tallynum is unused at this point

	// copy down from device
	cudaMemcpy( tally_score, d_tally_score , n_tally*sizeof(float),    cudaMemcpyDeviceToHost);
	cudaMemcpy( tally_count, d_tally_count , n_tally*sizeof(unsigned), cudaMemcpyDeviceToHost);

	// write tally values
	FILE* tfile = fopen(filename.c_str(),"w");
	for (int k=0;k<n_tally;k++){
		fprintf(tfile,"%10.8E %u\n",tally_score[k],tally_count[k]);
	}
	fclose(tfile);

	//write spacing
	float Emin = 1e-11;
	float Emax = 20.0;
	float edge = 0.0;
	float log_spacing 	= (log10f(Emax)-(-11.0))/(n_tally-2+1);//(log10f(Emax)-log10f(Emin))/(Ntally-2+1);
	float multiplier  	= powf(10,log_spacing);
	filename=filename+"bins";
	tfile = fopen(filename.c_str(),"w");
	edge=Emin;
	for (int k=0;k<n_tally;k++){
		fprintf(tfile,"%10.8E\n",edge);
		edge = edge*multiplier;
	}
	fclose(tfile);

}
float whistory::get_time(){

	return ((float)clock())/((float)CLOCKS_PER_SEC);

}
void whistory::set_tally_cell(unsigned cell){

	tally_cell = cell;

}































