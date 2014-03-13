#include "optix.h"
#include <optix_world.h>
#include "datadef.h"

using namespace optix;

rtBuffer<source_point,1>            positions_buffer;
rtBuffer<unsigned,1>                rxn_buffer;
rtBuffer<unsigned,1>                done_buffer;
rtBuffer<unsigned,1>                cellnum_buffer;
rtBuffer<unsigned,1>                matnum_buffer;
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(uint, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint, launch_dim,   rtLaunchDim, );
rtDeclareVariable(unsigned,  outer_cell, , );
rtDeclareVariable(unsigned,  trace_type, , );
rtDeclareVariable(unsigned,  boundary_condition, , );

RT_PROGRAM void camera()
{

	if(done_buffer[launch_index]){return;}

	// declare important stuff
	int                 cnt;
	float               epsilon=1e-4; 
	float               dist_to_surf = 0.0;
	float               x,y,z;
	unsigned            rxn, done, cellnum;
	float               samp_dist = positions_buffer[launch_index].samp_dist;
	
	intersection_point  payload;
	//rtPrintf("i=%u, (% 10.8E,% 10.8E,% 10.8E) (% 10.8E,% 10.8E,% 10.8E)\n",launch_index,positions_buffer[launch_index].x,positions_buffer[launch_index].y,positions_buffer[launch_index].z,positions_buffer[launch_index].xhat,positions_buffer[launch_index].yhat,positions_buffer[launch_index].zhat);
	
	// init payload flags
	payload.cont=1;
	payload.do_first_hit=1;
	
	// init ray
	float3 ray_direction  = make_float3(0,0,-1);   // just down, makes it much faster for 2d-like geoms?
	float3 ray_origin     = make_float3(positions_buffer[launch_index].x,    positions_buffer[launch_index].y,    positions_buffer[launch_index].z);
	optix::Ray ray        = optix::make_Ray( ray_origin, ray_direction, 0, epsilon, RT_DEFAULT_MAX );
	// init hitbuff
	for(cnt=0;cnt<10;cnt++){
		payload.hitbuff[cnt].cell = -1;
		payload.hitbuff[cnt].mat  = -1;
		payload.hitbuff[cnt].fiss = -1;
	}
	// first trace to find closest hit
	rtTrace(top_object, ray, payload);
	// check if bc
	if (payload.cell_first==outer_cell){
	 	payload.cont=0; 
	 	cellnum_buffer[launch_index]=outer_cell;
	}
	else{
		while(payload.cont){
			ray_origin = make_float3(payload.x,payload.y,payload.z);
			ray = optix::make_Ray( ray_origin, ray_direction, 0, epsilon, RT_DEFAULT_MAX );
			rtTrace(top_object, ray, payload);      
		}
		cellnum_buffer[launch_index] = payload.hitbuff[0].cell;
		// if number 3 requested, then write fissile flag to matnum instead of matnum
		//rtPrintf("cellnum,matnum,is_fissile = %d %d %d \n",payload.hitbuff[0].cell,payload.hitbuff[0].mat,payload.hitbuff[0].fiss);
	}
	if(trace_type == 2){
		matnum_buffer[launch_index]=payload.hitbuff[0].mat;
		//rtPrintf("matnum=%u\n",matnum_buffer[launch_index]);
	}
	else if(trace_type == 3){
		matnum_buffer[launch_index]=payload.hitbuff[0].fiss;
		rxn_buffer[launch_index]=18;
	}

}

RT_PROGRAM void exception()
{
	const unsigned int code = rtGetExceptionCode();
	rtPrintf( "Caught exception 0x%X at launch index (%d)\n", code, launch_index);
	rtPrintExceptionDetails();
}
