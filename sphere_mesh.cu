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
  bool check_second = true;

  float  radius = dims[object_dex].max[0];
  float3 loc    = make_float3(dims[object_dex].loc[0],dims[object_dex].loc[1],dims[object_dex].loc[2]);
  float3 xformed_origin = ray.origin - loc;       //direction does not need to be transformed since only translations are supported now, not rotations
  
  //vector ops
  //float a = 1.0; //dot(ray.direction, ray.direction);  direction should always be normalized and mag=1 in 3d.
  float b = dot(xformed_origin, ray.direction);
  float disc = b*b - dot(xformed_origin, xformed_origin) + radius*radius;
  
  if(disc>0.0f){

    float t0 = -b-sqrtf(disc);
    float t1 = -b+sqrtf(disc);

      if( rtPotentialIntersection( t0 ) ) {
          cellnum     = dims[object_dex].cellnum;
          cellmat     = dims[object_dex].matnum;
          cellfissile = dims[object_dex].is_fissile;
         if(rtReportIntersection(0))
           check_second = false;
      } 
      if(check_second) {
        if( rtPotentialIntersection( t1 ) ) {
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
  float3 mins = make_float3(-dims[object_dex].max[0],-dims[object_dex].max[0],-dims[object_dex].max[0]);  //set all to the radius
  float3 maxs = make_float3( dims[object_dex].max[0], dims[object_dex].max[0], dims[object_dex].max[0]);
  float3 loc  = make_float3( dims[object_dex].loc[0], dims[object_dex].loc[1], dims[object_dex].loc[2]);

  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->set(mins+loc, maxs+loc);
}
