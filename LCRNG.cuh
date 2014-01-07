inline __device__ float get_rand(float in)
{
/*
increments the random number with LCRNG
*/
    int   m = 2147483647;
    float a = 214748362;
    int   c = 2147483587;
    rand = fmodf( (a * m * in) + c, m) / m;
}