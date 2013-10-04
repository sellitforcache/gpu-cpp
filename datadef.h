/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//							Data stuff
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


struct wtransform {
    unsigned    cellnum;
    unsigned    cellmat;
    float       dx;
    float       dy;
    float       dz;
    float       theta;
    float       phi;
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
    float       x;
    float       y;
    float       z;
    float       surf_dist;
    int         cell_first;
    int         cont;
    int         do_first_hit;
    unsigned    is_fissile;
    unsigned    matnum;
    int         hitbuff[10];
};

// intersection point struct
struct material_def {
    unsigned matnum;
    unsigned is_fissile;
    unsigned num_isotopes;
    float    density;
    unsigned * isotopes;
    float    * fractions;
};
