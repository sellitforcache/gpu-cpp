#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

using namespace optix;

rtDeclareVariable(float3, cyl_param, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void cyl_intersect(int)
{
  float a =        ( ray.direction.x * ray.direction.x ) + ( ray.direction.y * ray.direction.y );
  float b = 2.0 * (( ray.direction.x * ray.origin.x    ) + ( ray.direction.y * ray.origin.y    ));
  float c =        ( ray.origin.x * ray.origin.x       ) + ( ray.origin.y * ray.origin.y       ) - (cyl_param.x * cyl_param.x);

  float disc = (b*b)-(4*a*c);
  bool report = false;

  if (disc > 0.0f ){

    float sdisc = sqrt(disc);
    float t1 = (-b-sdisc)/(2*a);
    float t2 = (-b+sdisc)/(2*a);
    float z1 = ray.origin.z + t1 * ray.direction.z;
    float z2 = ray.origin.z + t2 * ray.direction.z;
    bool check_second = true;

    if ( z1 > cyl_param.z  && z2 < cyl_param.z ) {
      t1=(cyl_param.z-ray.origin.z)/ray.direction.z;
      report=true;
    }
    if ( z2 > cyl_param.z  && z1 < cyl_param.z ) {
      t2=(cyl_param.z-ray.origin.z)/ray.direction.z;
      report=true;
    }

    if ( z1 < cyl_param.y  && z2 > cyl_param.y ) {
      t1=(cyl_param.y-ray.origin.z)/ray.direction.z;
      report=true;
    }
    if ( z2 < cyl_param.y  && z1 > cyl_param.y ) {
      t2=(cyl_param.y-ray.origin.z)/ray.direction.z;
      report=true;
    }

    if ( z1 > cyl_param.z  && z2 < cyl_param.y ) {
      t1=(cyl_param.z-ray.origin.z)/ray.direction.z;
      t2=(cyl_param.y-ray.origin.z)/ray.direction.z;
      report=true;
    }
    if ( z2 > cyl_param.z  && z1 < cyl_param.y ) {
      t2=(cyl_param.z-ray.origin.z)/ray.direction.z;
      t1=(cyl_param.y-ray.origin.z)/ray.direction.z;
      report=true;
    }

    if (z1 > cyl_param.y && z1 < cyl_param.z && z2 > cyl_param.y && z2 < cyl_param.z ){
      report=true;
    }

    if (report){
    if (rtPotentialIntersection(t1) ) {
      if(rtReportIntersection(0)){
        check_second=false;
      }
    }
    if(check_second){
      if (rtPotentialIntersection(t2) ) {
        rtReportIntersection(0);
      }
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
