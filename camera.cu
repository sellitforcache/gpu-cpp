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
  
  float3 ray_direction  = make_float3(positions_buffer[launch_index].xhat, positions_buffer[launch_index].yhat, positions_buffer[launch_index].zhat);
  float3 ray_origin     = make_float3(positions_buffer[launch_index].x,    positions_buffer[launch_index].y,    positions_buffer[launch_index].z);
  optix::Ray ray        = optix::make_Ray( ray_origin, ray_direction, 0, epsilon, RT_DEFAULT_MAX );

  //rtPrintf("i=%u, (% 10.8E,% 10.8E,% 10.8E) (% 10.8E,% 10.8E,% 10.8E)\n",launch_index,positions_buffer[launch_index].x,positions_buffer[launch_index].y,positions_buffer[launch_index].z,positions_buffer[launch_index].xhat,positions_buffer[launch_index].yhat,positions_buffer[launch_index].zhat);

  // init payload
  payload.cont=1;
  payload.do_first_hit=1;
  for(cnt=0;cnt<10;cnt++){
    payload.hitbuff[cnt].cell = -1;
    payload.hitbuff[cnt].mat  = -1;
    payload.hitbuff[cnt].fiss = -1;
  }

  // first trace to find closest hit
  rtTrace(top_object, ray, payload);


   if (trace_type==1){   // transport trace type
      dist_to_surf = payload.surf_dist;
      if ( (dist_to_surf - 1.75*epsilon) >= samp_dist ){  // interaction is closer
         x = positions_buffer[launch_index].x + samp_dist*positions_buffer[launch_index].xhat;
         y = positions_buffer[launch_index].y + samp_dist*positions_buffer[launch_index].yhat;
         z = positions_buffer[launch_index].z + samp_dist*positions_buffer[launch_index].zhat;
         rxn = rxn_buffer[launch_index];
         done = 0;
         cellnum = cellnum_buffer[launch_index];
      } 
      else{ // surface is closer 
         if (payload.cell_first==outer_cell){ // first check if BC
            if(boundary_condition == 0){
              rxn  = 888;  //  set leak code
              done = 1;   // set done flag
              // move out of geometry to "interaction point"
              cellnum = payload.cell_first;
              x = positions_buffer[launch_index].x + samp_dist*positions_buffer[launch_index].xhat;
              y = positions_buffer[launch_index].y + samp_dist*positions_buffer[launch_index].yhat;
              z = positions_buffer[launch_index].z + samp_dist*positions_buffer[launch_index].zhat;
            }
            else if(boundary_condition == 1){
              rtPrintf("CRAPPPPPPPPP\n");
            }
            else{
              rtPrintf("DOUBLE CRAPPPPPPPPP\n");
            }
         }
         else{
          rxn = 999; // set resample code
          done = 0;
          // move to surface
          cellnum = payload.cell_first;
          x = positions_buffer[launch_index].x + dist_to_surf*positions_buffer[launch_index].xhat;
          y = positions_buffer[launch_index].y + dist_to_surf*positions_buffer[launch_index].yhat;
          z = positions_buffer[launch_index].z + dist_to_surf*positions_buffer[launch_index].zhat;
         }
      }
   //write positions to buffers
   rtPrintf("launch_index = %d rxn %u done %u cellnum %u xyz_o % 10.8E % 10.8E % 10.8E xyz % 10.8E % 10.8E % 10.8E samp %10.8E surf %10.8E \n",launch_index,rxn,done,cellnum,positions_buffer[launch_index].x,positions_buffer[launch_index].y,positions_buffer[launch_index].z,x,y,z,samp_dist,dist_to_surf);
   rxn_buffer[launch_index]  = rxn;
   done_buffer[launch_index] = done;
   cellnum_buffer[launch_index] = cellnum;
   positions_buffer[launch_index].x = x;
   positions_buffer[launch_index].y = y;
   positions_buffer[launch_index].z = z;
   }


   else if(trace_type==2 | trace_type==3){   // where am I? trace
      // check if bc
      if (payload.cell_first==outer_cell){
            payload.cont=0; 
            cellnum_buffer[launch_index]==outer_cell;
      }
      while(payload.cont){
         ray_origin = make_float3(payload.x,payload.y,payload.z);
         ray = optix::make_Ray( ray_origin, ray_direction, 0, epsilon, RT_DEFAULT_MAX );
         rtTrace(top_object, ray, payload);      
      }
      cellnum_buffer[launch_index] = payload.hitbuff[0].cell;
      // if number 4 requested, then write fissile flag to matnum instead of matnum
      //rtPrintf("cellnum,matnum,is_fissile = %d %d %d \n",payload.hitbuff[0].cell,payload.hitbuff[0].mat,payload.hitbuff[0].fiss);
      if(trace_type == 2){
        matnum_buffer[launch_index]=payload.hitbuff[0].mat;
      }
      else if(trace_type == 3){
        matnum_buffer[launch_index]=payload.hitbuff[0].fiss;
      }

   }


}

RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf( "Caught exception 0x%X at launch index (%d)\n", code, launch_index);
  rtPrintExceptionDetails();
  //output_buffer[launch_index].x = -2;
  //output_buffer[launch_index].y = -2;
  //output_buffer[launch_index].z = -2;
}
