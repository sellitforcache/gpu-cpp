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

struct geom_data{
    float       min[3];
    float       max[3];
    int         cellnum;
    int         matnum;
    int         is_fissile;
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

//quaternary search node and host container
struct qnode{
    float  values[5];
    qnode* leaves[4];
};
struct qnode_host{
    qnode* cuda_pointer;
    qnode  node;
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
