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
    float macro_t;
};

//hit buffer struct
struct hit_buffer {
    int     cell;
    int     mat;
    int     fiss;
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
    hit_buffer  hitbuff[10];
};

// intersection point struct
struct material_def {
    unsigned    id;
    unsigned    matnum;
    unsigned    is_fissile;
    unsigned    num_isotopes;
    float       density;
    unsigned *  isotopes;
    float    *  fractions;
};
