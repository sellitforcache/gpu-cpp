#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

using namespace optix;

rtDeclareVariable(float3, mins, , );
rtDeclareVariable(float3, maxs, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void intersect(int)
{
  float a =        ( ray.direction.x * ray.direction.x ) + ( ray.direction.y * ray.direction.y );
  float b = 2.0 * (( ray.direction.x * ray.origin.x    ) + ( ray.direction.y * ray.origin.y    ));
  float c =        ( ray.origin.x    * ray.origin.x    ) + ( ray.origin.y * ray.origin.y       ) - (maxs.y * maxs.y);

  float disc = (b*b)-(4*a*c);
  bool report = false;

  if (disc > 0.0f ){

    float sdisc = sqrt(disc);
    float t1 = (-b-sdisc)/(2*a);
    float t2 = (-b+sdisc)/(2*a);
    float z1 = ray.origin.z + t1 * ray.direction.z;
    float z2 = ray.origin.z + t2 * ray.direction.z;
    bool check_second = true;

    if ( z1 > maxs.x  && z2 < maxs.x ) {
      t1=(maxs.x-ray.origin.z)/ray.direction.z;
      report=true;
    }
    if ( z2 > maxs.x  && z1 < maxs.x ) {
      t2=(maxs.x-ray.origin.z)/ray.direction.z;
      report=true;
    }

    if ( z1 < mins.x  && z2 > mins.x ) {
      t1=(mins.x-ray.origin.z)/ray.direction.z;
      report=true;
    }
    if ( z2 < mins.x  && z1 > mins.x ) {
      t2=(mins.x-ray.origin.z)/ray.direction.z;
      report=true;
    }

    if ( z1 > maxs.x  && z2 < mins.x ) {
      t1=(maxs.x-ray.origin.z)/ray.direction.z;
      t2=(mins.x-ray.origin.z)/ray.direction.z;
      report=true;
    }
    if ( z2 > maxs.x  && z1 < mins.x ) {
      t2=(maxs.x-ray.origin.z)/ray.direction.z;
      t1=(mins.x-ray.origin.z)/ray.direction.z;
      report=true;
    }

    if (z1 > mins.x && z1 < maxs.x && z2 > mins.x && z2 < maxs.x ){
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

RT_PROGRAM void bounds (int, float result[6])
{
  result[0] = -maxs.y;
  result[1] = -maxs.y;
  result[2] = mins.x;
  result[3] = maxs.y;
  result[4] = maxs.y;
  result[5] = maxs.x;
}
