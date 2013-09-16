#include <optix_world.h>
#include "datadef.h"


rtDeclareVariable(intersection_point, payload, rtPayload, );
rtDeclareVariable(uint, launch_index, rtLaunchIndex, );
rtDeclareVariable(int,  trace_type, , );
//rtBuffer<source_point,1>          particles;
rtBuffer<unsigned,1>          done_buffer;
rtBuffer<unsigned,1>          rxn_buffer;

RT_PROGRAM void miss()
{
	rtPrintf("!!!MISS!!! launch_index = %d trace %d done %u (x,y,z,dist,samp)= % 10.8E % 10.8E % 10.8E %10.8E %10.8E \n", launch_index, trace_type, done_buffer[launch_index], payload.x,payload.y,payload.z,payload.surf_dist);//particles[launch_index].samp_dist,trace_type);
	payload.x = -1;
	payload.y = -1;
	payload.z = -1;
	payload.cont = 0;
	done_buffer[launch_index]=1;
	rxn_buffer[launch_index]=123; //miss code
}
