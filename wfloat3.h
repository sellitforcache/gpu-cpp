//class definitions for device vector operations
class wfloat3{
public:
	float x,y,z;
	inline __device__ 			wfloat3();
	inline __device__ 			wfloat3(float);
	inline __device__ 			wfloat3(float,float,float);
	inline __device__ wfloat3 	operator+ (wfloat3); 
	inline __device__ wfloat3 	operator- (wfloat3);
	inline __device__ wfloat3 	operator* (wfloat3);
	inline __device__ wfloat3 	operator+ (float); 
	inline __device__ wfloat3 	operator- (float);
	inline __device__ wfloat3 	operator* (float);
	inline __device__ wfloat3 	operator/ (float);
	inline __device__ wfloat3 	cross(wfloat3);
	inline __device__ float   	dot(wfloat3);
	inline __device__ void  	rodrigues_rotation(wfloat3,float);
	inline __device__ float 	norm2();
};
__device__ wfloat3::wfloat3(){x=0;y=0;z=0;};
__device__ wfloat3::wfloat3(float a){x=a;y=a;z=a;};
__device__ wfloat3::wfloat3(float a,float b,float c){x=a;y=b;z=c;};
__device__ wfloat3 wfloat3::operator+ (wfloat3 arg){
	wfloat3 result(x+arg.x,y+arg.y,z+arg.z);
	return result;
}; 
__device__ wfloat3 wfloat3::operator- (wfloat3 arg){
	wfloat3 result(x-arg.x,y-arg.y,z-arg.z);
	return result;
};
__device__ wfloat3 wfloat3::operator* (wfloat3 arg){
	wfloat3 result(x*arg.x,y*arg.y,z*arg.z);
	return result;
};
__device__ wfloat3 wfloat3::operator+ (float arg){
	wfloat3 result(x+arg,y+arg,z+arg);
	return result;
}; 
__device__ wfloat3 wfloat3::operator- (float arg){
	wfloat3 result(x-arg,y-arg,z-arg);
	return result;
};
__device__ wfloat3 wfloat3::operator* (float arg){
	wfloat3 result(x*arg,y*arg,z*arg);
	return result;
};
__device__ wfloat3 wfloat3::operator/ (float arg){
	wfloat3 result(x/arg,y/arg,z/arg);
	return result;
};
__device__ wfloat3 wfloat3::cross(wfloat3 arg){
	wfloat3 result;
	result.x =  y*arg.z - arg.y*z ;
	result.y = -x*arg.z + arg.x*z ;
	result.z =  x*arg.y - arg.x*y ;
	return result;
};
__device__ float wfloat3::dot(wfloat3 arg){
	float result;
	result = x*arg.x + y*arg.y + z*arg.z;
	return result;
};
__device__ void wfloat3::rodrigues_rotation(wfloat3 k, float theta){
	*this = (*this)*cos(theta) + (k.cross(*this))*sin(theta) + k*(k.dot(*this))*(1.0-cos(theta));
};
__device__ float wfloat3::norm2(){
	float result = sqrtf(x*x+y*y+z*z);
	return result;
};