inline __device__ float get_rand(float in)
{
/*
increments the random number with LCRNG
*/
	double m = 2147483647;
    double a = 214748362;
    double c = 2147483587;
    float out = fmod( a * m * in + c, m ) / m ;
    return out;
}