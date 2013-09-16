#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

using namespace optix;

rtDeclareVariable(float3, cyl_param, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void cyl_intersect(int)
{
  float3 boxmin = make_float3(-cyl_param.x,-cyl_param.x,cyl_param.y);
  float3 boxmax = make_float3(cyl_param.x,cyl_param.x,cyl_param.z);
  float3 t0 = (boxmin - ray.origin)/ray.direction;
  float3 t1 = (boxmax - ray.origin)/ray.direction;
  float3 near = fminf(t0, t1);
  float3 far = fmaxf(t0, t1);
  float tmin = fmaxf( near );
  float tmax = fminf( far );

  if(tmin <= tmax) {
    bool check_second = true;
    if( rtPotentialIntersection( tmin ) ) {
       if(rtReportIntersection(0))
         check_second = false;
    } 
    if(check_second) {
      if( rtPotentialIntersection( tmax ) ) {
        rtReportIntersection(0);
      }
    }
  }


}

RT_PROGRAM void cyl_bounds (int, float result[6])
{
  result[0] = -cyl_param.x;
  result[1] = -cyl_param.x;
  result[2] = cyl_param.y;
  result[3] = cyl_param.x;
  result[4] = cyl_param.x;
  result[5] = cyl_param.z;
}
