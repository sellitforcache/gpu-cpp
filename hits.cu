#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "datadef.h"

using namespace optix;

rtDeclareVariable(intersection_point, payload, rtPayload, ); 
rtDeclareVariable(float, int_dist, rtIntersectionDistance, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(uint, launch_index, rtLaunchIndex, );
rtDeclareVariable(unsigned,  cellnum, , );
rtDeclareVariable(unsigned,  outer_cell, , );

RT_PROGRAM void closest_hit()
{
	int j=0, k=0, go=1;

	// stop ray iterations if outer cell is hit
	if(cellnum==outer_cell){
		payload.cont=0;
	}

	// always update current position
	payload.x=int_dist*ray.direction.x+ray.origin.x;
	payload.y=int_dist*ray.direction.y+ray.origin.y;
	payload.z=int_dist*ray.direction.z+ray.origin.z;

	// write in first hit
	if(payload.do_first_hit==1){     // if this is the first hit and a transport trace, set first hit and current position
			payload.surf_dist    = int_dist;
			payload.hitbuff[0]   = cellnum;
			payload.cell_first   = cellnum;
			payload.do_first_hit = 0;
	}
	else if(payload.do_first_hit==0){		// scan the hitbuff
		//rtPrintf("in 0 case\n",payload.do_first_hit);
		while(go){
			if (payload.hitbuff[k]==cellnum){     // this cell has been intersected already, remove it from the list and shift all numbers down
				go=0;
				for(j=k;j<9;j++){
					payload.hitbuff[j]=payload.hitbuff[j+1];
				}
				payload.hitbuff[9]=-1;
			}
			else if (payload.hitbuff[k]==-1){      // not in list, append cellnum to the end of hitbuff
				go=0;
				payload.hitbuff[k]=cellnum; 
			}
			k++;
		}
	}
	else{  //something is weird
		rtPrintf("something is wrong!\n");
		payload.do_first_hit=0;
	}


}
