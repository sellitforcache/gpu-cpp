#include <vector> 

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
	 box()
	~box()
	add_transform();
	add_transform( int  , int  , float  , float  , float  , float  , float  );
	float       min[3];
	float       max[3];
	float       location[3];
	int         n_transforms;
	std::vector<transform>   transforms;
};

box::box(){
	//box default constructor
	min={0,0,0};
	max={0,0,0};
	location={0,0,0};
	transforms[0].cellnum = 0;
	transforms[0].cellmat = 0;
	transforms[0].dx      = 0;
	transforms[0].dy      = 0;
	transforms[0].dz      = 0;
	transforms[0].theta   = 0;
	transforms[0].phi     = 0;
	transforms.push_end()
}
box::~box(){
	//box destructor
	delete min;
	delete max;
	delete location;
	delete n_transforms;
	delete   transforms;
}
box::add_transform(){
	transform this_transform;
	this_transform.cellnum = 0;
	this_transform.cellmat = 0;
	this_transform.dx      = 0;
	this_transform.dy      = 0;
	this_transform.dz      = 0;
	this_transform.theta   = 0;
	this_transform.phi     = 0;
	transforms.push_end(this_transform);
}
box::add_transform(int cellnum , int cellmat , float dx , float dy , float dz , float theta , float phi ){
	transform this_transform;
	this_transform.cellnum = cellnum;
	this_transform.cellmat = cellmat;
	this_transform.dx      = dx;
	this_transform.dy      = dy;
	this_transform.dz      = dz;
	this_transform.theta   = theta;
	this_transform.phi     = phi;
	transforms.push_end(this_transform);
}



class geometry {

	
}