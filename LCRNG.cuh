inline __device__ float get_rand(unsigned* in)
{
/*
increments the random number with LCRNG 
adapated from openmc again
values from http://www.ams.org/journals/mcom/1999-68-225/S0025-5718-99-00996-5/S0025-5718-99-00996-5.pdf
since 32-bit math is being used, 30 bits are used here
*/
	unsigned a   		= 116646453;		 		// multiplier
	unsigned c   		= 1;						// constant add
	unsigned prn_mod    = 536870912;  				// 2^30
	unsigned prn_mask   = prn_mod - 1; 				// 2^30-1
	float prn_norm   	= 1.862645149230957e-09;	// 2^-30
	in[0] = (a * in[0] +  c) & prn_mask;   			// mod by truncation
	//printf("%10.8E\n",prn_norm*in[0]);
	return prn_norm*in[0];   						// return normalized float
}