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
    float r    = maxs.y;
    float zmin = mins.x;
    float zmax = maxs.x;

    float3 bbmin = make_float3(-r,-r,zmin);
    float3 bbmax = make_float3( r, r,zmax);

    // get bounding box tmins and tmaxes
    float3 bbt0 = (bbmin - ray.origin)/ray.direction;
    float3 bbt1 = (bbmax - ray.origin)/ray.direction;
    float3 bbnear = fminf(bbt0, bbt1);
    float3 bbfar  = fmaxf(bbt0, bbt1);
    float bbtmin  = fmaxf( bbnear );
    float bbtmax  = fminf( bbfar );
    float t1,t2;

    float a =        ( ray.direction.x * ray.direction.x ) + ( ray.direction.y * ray.direction.y );
    float b = 2.0 * (( ray.direction.x * ray.origin.x    ) + ( ray.direction.y * ray.origin.y    ));
    float c =        ( ray.origin.x    * ray.origin.x    ) + ( ray.origin.y    * ray.origin.y    ) - (r * r);

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
        if ( sqrtf(ray.origin.x*ray.origin.x + ray.origin.y*ray.origin.y) <= r ) {
            // inside the cap
            //rtPrintf("completely perpendicular and inside\n");
            t1 = bbtmin;
            t2 = bbtmax;
            report = true;
        }

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

RT_PROGRAM void bounds (int, float result[6])
{
    float r    = maxs.y;
    float zmin = mins.x;
    float zmax = maxs.x;

    result[0] = -r;
    result[1] = -r;
    result[2] = zmin;
    result[3] = r;
    result[4] = r;
    result[5] = zmax;
}
