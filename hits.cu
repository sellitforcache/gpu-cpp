#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "datadef.h"

using namespace optix;

rtDeclareVariable(intersection_point, payload, rtPayload, ); 
rtDeclareVariable(float, int_dist, rtIntersectionDistance, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
//rtDeclareVariable(uint, launch_index, rtLaunchIndex, );
rtDeclareVariable(unsigned,  cellnum, , );
rtDeclareVariable(unsigned,  cellmat, , );
rtDeclareVariable(unsigned,  cellfissile, , );
rtDeclareVariable(unsigned,  outer_cell, , );

RT_PROGRAM void closest_hit()
{
	unsigned j = 0;
	unsigned cell_notfound = 1;
	unsigned end_notfound = 1;
	int index = -1;

	// stop ray iterations if outer cell is hit
	if(cellnum==outer_cell){
		payload.cont=0;
	}

	// write this cell's info into buffer element
	hit_buffer this_buff;
	this_buff.cell = cellnum;
	this_buff.mat  = cellmat;
	this_buff.fiss = cellfissile;

	// make end element
	hit_buffer end_buff;
	end_buff.cell = -1; 
	end_buff.mat  = -1; 
	end_buff.fiss = -1; 

	//rtPrintf("cellnum,matnum,isfiss %d %d %d\n",this_buff.cell,this_buff.mat,this_buff.fiss);

	// always update current position
	payload.x=int_dist*ray.direction.x+ray.origin.x;
	payload.y=int_dist*ray.direction.y+ray.origin.y;
	payload.z=int_dist*ray.direction.z+ray.origin.z;

	// write in first hit
	if(payload.do_first_hit==1){     // if this is the first hit and a transport trace, set first hit and current position
			payload.surf_dist       = int_dist; //rtPrintf("int_dist %6.4E\n",int_dist);
			payload.cell_first      = cellnum;
			payload.do_first_hit    = 0;
			payload.hitbuff[0]      = this_buff;
	}
	else if(payload.do_first_hit==0){		// this is a whereami trace, so find which cell we are in
		// scan the hitbuff to see if this cell has been intersected already
		for(j=0;j<10;j++){
			if(payload.hitbuff[j].cell==this_buff.cell & cell_notfound){
				cell_notfound = 0;
				end_notfound = 0;
				index=j;
				break;
			}
			if(payload.hitbuff[j].cell==-1 & cell_notfound & end_notfound){
				end_notfound = 0;
				index=j;
				break;
			}
		}
		if(end_notfound){
			rtPrintf("hit buffer overrun! index=%d ,last entry %d %d %d \n",index,payload.hitbuff[9].cell,payload.hitbuff[9].mat,payload.hitbuff[9].fiss);
			rtThrow(RT_EXCEPTION_USER + 0);  // throw user exception 0
		}
		else if(cell_notfound){  // append to end
			//rtPrintf("cell not found. append to index %d\n",index);
			payload.hitbuff[index]=this_buff;
		}
		else{  // shift all down append -1's to end
			//rtPrintf("found. shift from index %d\n",index);
			//memcpy(&payload.hitbuff[index],&payload.hitbuff[index+1],(9-index)*sizeof(hit_buffer));
			for(j=index;j<9;j++){
				payload.hitbuff[j]=payload.hitbuff[j+1];
			}
			payload.hitbuff[9]=end_buff;
		}

	}
	else{  //something is weird
		rtPrintf("something is wrong!\n");
		payload.do_first_hit=0;
	}


}
