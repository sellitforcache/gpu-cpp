/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//							Data stuff
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


struct transform {
    int   cellnum;
    int   cellmat;
    float dx;
    float dy;
    float dz;
    float theta;
    float phi;
};

// source point struct
struct source_point{ 
    float x; 
    float y; 
    float z; 
    float xhat; 
    float yhat;
    float zhat; 
    float samp_dist;
};

// 3-float for scattering calculations
struct f3{
    float x;
    float y;
    float z;
};

// intersection point struct
struct intersection_point {
    float   x;
    float   y;
    float   z;
    float   surf_dist;
    int     cell_first;
    int     cont;
    int     do_first_hit;
    int     hitbuff[10];
};
