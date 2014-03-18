//history struct
class whistory { 
	// geometry object
	wgeometry 			   problem_geom;
	// optix object 
//	optix_stuff 		   optix_obj;     now a global in the whistory object
	std::string 		   accel_type;
	// CUDPP
	CUDPPHandle            theCudpp;
	CUDPPHashTableConfig   hash_config;
	CUDPPConfiguration     compact_config;
	CUDPPConfiguration     scan_int_config;
	CUDPPConfiguration     redu_int_config;
	CUDPPConfiguration     redu_float_config;
	CUDPPHandle            mate_hash_table_handle;
	CUDPPHandle            fiss_hash_table_handle;
	CUDPPHandle            scanplan_int;
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
	unsigned 		N;
	unsigned 		Ndataset;
	unsigned  		RNUM_PER_THREAD;
	unsigned 		NUM_THREADS;
	unsigned 		blks;
	unsigned 		compute_device;
	cudaStream_t 	stream[5];
	// host data
	unsigned 		RUN_FLAG;
	unsigned 		qnodes_depth, qnodes_width;
    unsigned        outer_cell;
	unsigned 		n_materials;
	unsigned 		n_isotopes;
	unsigned 		n_tally;
	unsigned 		n_qnodes;
	unsigned 		n_skip;
	unsigned 		n_cycles;
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
    unsigned *		rn_bank;
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
    qnode * 		qnodes;
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
	unsigned *         d_rn_bank;
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
    float * 		d_fissile_energy;
    source_point * 	d_fissile_points;
    unsigned * 		d_mask;
    qnode *			d_qnodes;
    unsigned * 		d_completed;
 	unsigned *  	d_scanned;
 	unsigned * 		d_active;
 	unsigned * 		d_num_completed;
 	unsigned * 		d_num_active;
 	source_point *  d_bank_space;
 	float * 		d_bank_E;
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
    void init_RNG();
    void update_RNG();
    void init_CUDPP();
    void init_host();
    void copy_to_device();
    void load_cross_sections();
    void trace(unsigned);
    unsigned reduce_done();
    void reset_cycle(float);
    void reset_fixed();
    void converge(unsigned);
    void sample_fissile_points();
    float reduce_yield();
    void create_quad_tree();
    float get_time();
    void prep_secondaries();
    unsigned map_active();
    void  write_to_file(source_point*  , unsigned , std::string );
    void  write_to_file(unsigned*  , unsigned , std::string );
public:
     whistory(int,wgeometry);
    ~whistory();
    void print_xs_data();
    void print_pointers();
    void print_materials_table();
    void run();
    void write_xs_data(std::string);
    void write_tally(unsigned, std::string);
    void set_tally_cell(unsigned);
    void set_run_type(unsigned);
    void set_run_type(std::string);
    void set_run_param(unsigned,unsigned);
    void init();
    void device_report();
    void set_device(unsigned);
    void set_acceration(std::string);
};

