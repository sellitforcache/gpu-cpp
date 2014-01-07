#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include "datadef.h"

using namespace optix;

rtBuffer<geom_data,1>               dims;
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(unsigned,  cellnum,     attribute cell_num, );
rtDeclareVariable(unsigned,  cellmat,     attribute cell_mat, );
rtDeclareVariable(unsigned,  cellfissile, attribute cell_fis, );

RT_PROGRAM void intersect(int object_dex)
{
  float3 mins = make_float3(dims[object_dex].min[0],dims[object_dex].min[1],dims[object_dex].min[2]);
  float3 maxs = make_float3(dims[object_dex].max[0],dims[object_dex].max[1],dims[object_dex].max[2]);
  float3 t0 = (mins - ray.origin)/ray.direction;
  float3 t1 = (maxs - ray.origin)/ray.direction;
  float3 near = fminf(t0, t1);
  float3 far = fmaxf(t0, t1);
  float tmin = fmaxf( near );
  float tmax = fminf( far );

  if(tmin <= tmax) {
    bool check_second = true;
    if( rtPotentialIntersection( tmin ) ) {
        cellnum     = dims[object_dex].cellnum;
        cellmat     = dims[object_dex].matnum;
        cellfissile = dims[object_dex].is_fissile;
       if(rtReportIntersection(0))
         check_second = false;
    } 
    if(check_second) {
      if( rtPotentialIntersection( tmax ) ) {
        cellnum     = dims[object_dex].cellnum;
         cellmat     = dims[object_dex].matnum;
         cellfissile = dims[object_dex].is_fissile;
        rtReportIntersection(0);
      }
    }
  }
}

RT_PROGRAM void bounds (int object_dex, float result[6])
{
  float3 mins = make_float3(dims[object_dex].min[0],dims[object_dex].min[1],dims[object_dex].min[2]);
  float3 maxs = make_float3(dims[object_dex].max[0],dims[object_dex].max[1],dims[object_dex].max[2]);
  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->set(mins, maxs);
}
