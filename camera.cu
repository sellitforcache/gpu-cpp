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
	intersection_point  payload;
	
	// init payload flags
	payload.cont=1;
	payload.do_first_hit=1;
	
	// init ray
	float3 ray_direction  = make_float3(positions_buffer[launch_index].xhat, positions_buffer[launch_index].yhat, positions_buffer[launch_index].zhat);
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
	positions_buffer[launch_index].surf_dist=payload.surf_dist;

	// check if bc, if first hit is BC, then its in the outer cell
	if (payload.cell_first==outer_cell){
	 	payload.cont=0; 
	 	cellnum_buffer[launch_index]=outer_cell;
	}
	else{ // else find the cell it's entering
		while(payload.cont){
			ray_origin = make_float3(payload.x,payload.y,payload.z);
			ray = optix::make_Ray( ray_origin, ray_direction, 0, epsilon, RT_DEFAULT_MAX );
			rtTrace(top_object, ray, payload);      
		}
		cellnum_buffer[launch_index] = payload.hitbuff[0].cell;
	}
	if(trace_type == 2){ //write material to buffer normally

		matnum_buffer[launch_index]=payload.hitbuff[0].mat;
	}
	else if(trace_type == 3){  //write fissile flag if fissile query
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
