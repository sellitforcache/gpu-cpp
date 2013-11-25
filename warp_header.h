// host calls
void print_banner();
// device calls
void set_positions_rand( unsigned, unsigned, unsigned, source_point * , float *  , float  * );
void copy_points(unsigned , unsigned , unsigned , unsigned*  , unsigned  , unsigned *  , source_point *  , source_point * );
void sample_fission_spectra(unsigned , unsigned , unsigned , unsigned* , unsigned* , unsigned* , unsigned* , float * , float *, source_point* , float** );
void sample_fixed_source( unsigned,unsigned,unsigned,unsigned*,float*,float*,source_point*);
void macroscopic( unsigned ,  unsigned, unsigned, unsigned , unsigned*, source_point * , unsigned* , unsigned * , unsigned * , float * , float * , float * , float *  , float* , unsigned*);
void microscopic( unsigned ,  unsigned , unsigned , unsigned , unsigned*, unsigned* , unsigned * , float * , float * , float * , float *  , unsigned * , unsigned * ,  float* , unsigned * , float*, unsigned* );
void tally_spec( unsigned ,  unsigned , unsigned , unsigned*, source_point * , float* , float * , unsigned * , unsigned*, unsigned*);
void escatter( unsigned , unsigned, unsigned , unsigned*, unsigned* , unsigned* , float* , float*, source_point* , unsigned*, float*, unsigned*, float**);
void iscatter( unsigned , unsigned , unsigned , unsigned*, unsigned* , unsigned * , float * , float *, source_point *  ,unsigned * , float* , float* , unsigned* , float**, float**);
void cscatter( unsigned , unsigned , unsigned , unsigned*, unsigned* , unsigned * , float * , float *, source_point *  ,unsigned * , float* , float* , unsigned* , float**, float**);
void fission( unsigned , unsigned, unsigned , unsigned*, unsigned* , unsigned* , unsigned*  , float * , unsigned* , float**);
void absorb( unsigned , unsigned , unsigned*, unsigned*  , unsigned* );
void find_E_grid_index( unsigned , unsigned , unsigned , unsigned*,float * , float* , unsigned *, unsigned* );
void find_E_grid_index_quad( unsigned, unsigned,  unsigned,  unsigned, unsigned*, qnode*, float*, unsigned*, unsigned*);
void make_mask( unsigned, unsigned, unsigned*, unsigned*, unsigned, unsigned);
void print_histories(unsigned, unsigned, unsigned, unsigned *, unsigned*, source_point*, float*, unsigned*);
void pop_secondaries( unsigned, unsigned, unsigned, unsigned* , unsigned* , unsigned* , unsigned* , unsigned*, unsigned*, source_point* , float* , float* , float** );
void pop_source( unsigned, unsigned, unsigned, unsigned* , unsigned* , unsigned* , unsigned* , unsigned*, unsigned*, source_point* , float* , float* , float*, source_point*, float** );
void flip_done(unsigned , unsigned , unsigned* );