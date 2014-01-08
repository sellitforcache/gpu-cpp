#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include "datadef.h"

using namespace optix;

rtBuffer<geom_data,1>              dims;
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(unsigned,  cellnum,     attribute cell_num, );
rtDeclareVariable(unsigned,  cellmat,     attribute cell_mat, );
rtDeclareVariable(unsigned,  cellfissile, attribute cell_fis, );

RT_PROGRAM void intersect(int object_dex)
{
    float3 mins = make_float3(dims[object_dex].min[0],dims[object_dex].min[1],dims[object_dex].min[2]);
    float3 maxs = make_float3(dims[object_dex].max[0],dims[object_dex].max[1],dims[object_dex].max[2]);
    float3 loc  = make_float3(dims[object_dex].loc[0],dims[object_dex].loc[1],dims[object_dex].loc[2]);
    float3 xformed_origin = ray.origin - loc;

    float r    = maxs.y;
    float zmin = mins.x;
    float zmax = maxs.x;

    float3 bbmin = make_float3(-r,-r,zmin);
    float3 bbmax = make_float3( r, r,zmax);

    // get bounding box tmins and tmaxes
    float3 bbt0 = (bbmin - xformed_origin)/ray.direction;
    float3 bbt1 = (bbmax - xformed_origin)/ray.direction;
    float3 bbnear = fminf(bbt0, bbt1);
    float3 bbfar  = fmaxf(bbt0, bbt1);
    float bbtmin  = fmaxf( bbnear );
    float bbtmax  = fminf( bbfar );
    float t1,t2;

    float a =        ( ray.direction.x  * ray.direction.x  ) + ( ray.direction.y  * ray.direction.y  );
    float b = 2.0 * (( ray.direction.x  * xformed_origin.x ) + ( ray.direction.y  * xformed_origin.y ));
    float c =        ( xformed_origin.x * xformed_origin.x ) + ( xformed_origin.y * xformed_origin.y ) - (r * r);

    float disc = (b*b)-(4*a*c);
    bool report = false;
    bool check_second = true;

    if (disc > 0.0f){  //the line intersects the circle

        float sdisc = sqrt(disc);
        t1 = (-b-sdisc)/(2*a);
        t2 = (-b+sdisc)/(2*a);

        // bound the t's to within the bb
        t1 = fminf (  fmaxf( t1, bbtmin ) , bbtmax );
        t2 = fminf (  fmaxf( t2, bbtmin ) , bbtmax );

        report = true;
        
    }
    else if ( ray.direction.y+ray.direction.x < 1e-15 ){

        // the ray is completely perpendicular to the x-y plane
        if ( sqrtf(xformed_origin.x*xformed_origin.x + xformed_origin.y*xformed_origin.y) <= r ) {
            // inside the cap
            //rtPrintf("completely perpendicular and inside\n");
            t1 = bbtmin;
            t2 = bbtmax;
            report = true;
        }

    }

    if (report){
        if (rtPotentialIntersection(t1) ) {
            cellnum     = dims[object_dex].cellnum;
            cellmat     = dims[object_dex].matnum;
            cellfissile = dims[object_dex].is_fissile;
            if(rtReportIntersection(0)){
                check_second=false;
            }
        }
        if(check_second){
            if (rtPotentialIntersection(t2) ) {
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
    float3 loc  = make_float3(dims[object_dex].loc[0],dims[object_dex].loc[1],dims[object_dex].loc[2]);

    float r    = maxs.y;
    float zmin = mins.x;
    float zmax = maxs.x;

    result[0] = -r      + loc.x;
    result[1] = -r      + loc.y;
    result[2] = zmin    + loc.z;
    result[3] = r       + loc.x;
    result[4] = r       + loc.y;
    result[5] = zmax    + loc.z;
}
