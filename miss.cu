#include <optix_world.h>
#include "datadef.h"

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(intersection_point, payload, rtPayload, );
rtDeclareVariable(uint, launch_index, rtLaunchIndex, );
rtDeclareVariable(unsigned,  trace_type, , );
rtDeclareVariable(unsigned,  outer_cell, , );
rtBuffer<unsigned,1>          done_buffer;
rtBuffer<unsigned,1>          rxn_buffer;
rtBuffer<source_point,1>      positions_buffer;
rtBuffer<unsigned,1>      	  matnum_buffer;
rtBuffer<unsigned,1>      	  cellnum_buffer;

RT_PROGRAM void miss()
{
	rtPrintf("!!!MISS!!! rxn = %u outer_cell = %d launch_index = %d trace %d done %u (x,y,z-source)= % 10.9E % 10.9E % 10.9E (x,y,z,dist)= % 6.4E % 6.4E % 6.4E %6.4E \n",rxn_buffer[launch_index], outer_cell, launch_index, trace_type, done_buffer[launch_index],positions_buffer[launch_index].x,positions_buffer[launch_index].y,positions_buffer[launch_index].z, payload.x,payload.y,payload.z,payload.surf_dist);
	payload.cont 				=  0;
	done_buffer[launch_index]	=  1;
	rxn_buffer[launch_index]	=  99999; //miss code
	payload.cell_first 			=  4294967295;
	payload.do_first_hit 		=  0;
	payload.surf_dist 			= -1.0;
	payload.hitbuff[0].cell 	=  4294967295;
	payload.hitbuff[0].mat  	=  0;
	payload.hitbuff[0].fiss 	=  0;
}
