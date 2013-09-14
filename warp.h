#include <vector> 
#include <iostream>

struct transform {
    int   cellnum;
    int   cellmat;
    float dx;
    float dy;
    float dz;
    float theta;
    float phi;
};

class box
{	
	public:
	 box();
	 box(int,int,float,float,float,float,float,float,float,float,float);
	~box();
	void add_transform();
	void add_transform(int,int,float,float,float,float,float);
	void print_transform();
	void print_transform(int);
	float       min[3];
	float       max[3];
	float       location[3];
	static int  num_boxes;
	int 		box_id;
	int         n_transforms;
	std::vector<transform>   transforms;
};

int box::num_boxes=0;

box::box(){
	//box default constructor
	min[0]=0;min[1]=0;min[2]=0;
	max[0]=0;max[1]=0;max[2]=0;
	location[0]=0;location[1]=0;location[2]=0;
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
	box_id=num_boxes;
	num_boxes++;
}
box::box(int cellnum ,int cellmat ,float xmin,float ymin,float zmin,float xmax,float ymax,float zmax,float x,float y,float z){
	//box valued constructor
	min[0]=xmin;min[1]=ymin;min[2]=zmin;
	max[0]=xmax;max[1]=ymax;max[2]=zmax;
	location[0]=x;location[1]=y;location[2]=z;
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
	box_id=num_boxes;
	num_boxes++;
}
box::~box(){
	//box destructor
	//delete min;
	//delete max;
	//delete location;
	//delete &box_id;
	//delete &n_transforms;
	//delete   &transforms;
}
void box::add_transform(){
	transform this_transform;
	this_transform.cellnum = 0;
	this_transform.cellmat = 0;
	this_transform.dx      = 0;
	this_transform.dy      = 0;
	this_transform.dz      = 0;
	this_transform.theta   = 0;
	this_transform.phi     = 0;
	transforms.push_back(this_transform);
	n_transforms++;
}
void box::add_transform(int cellnum , int cellmat , float dx , float dy , float dz , float theta , float phi ){
	transform this_transform;
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
void box::print_transform(){
	std::cout << "--- box id  = " << box_id << " ---" << "\n";
	std::cout << "   min,max  = (" << min[0] << min[1] << min[2] << "),(" << max[0] << max[1] << max[2] << ")" << "\n";
	std::cout << "   location = (" << location[0] << location[1] << location[2] << ")" << "\n";
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
void box::print_transform(int tnum){
	std::cout << "--- box id  = " << box_id << " ---" << "\n";
	std::cout << "   min,max  = (" << min[0] << min[1] << min[2] << "),(" << max[0] << max[1] << max[2] << ")" << "\n";
	std::cout << "   location = (" << location[0] << location[1] << location[2] << ")" << "\n";
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



class geometry {

	
};