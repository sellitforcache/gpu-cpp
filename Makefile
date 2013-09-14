#
CC = /usr/bin/gcc
OPTIX=/Developer/OptiX
NVCC = /usr/local/cuda/bin/nvcc
ARCH = -arch sm_30
CFLAGS = -O3 -m64 -I$(OPTIX)/include -L$(OPTIX)/lib64
NVCCFLAGS = -m64 -I$(OPTIX)/include 
NVCCLIBS = -L$(OPTIX)/lib64 -loptix
CUDPP = -lcudpp_hash -lcudpp -lcurand
LIBS =
	
ptx_objects = 	camera.ptx \
				hits.ptx \
				miss.ptx \
				box.ptx \
				cylinder.ptx \
				hex.ptx

all:  	$(ptx_objects) \
		mt19937ar.o \
		mcgpu \

clean:
	rm -f *.ptx *.o mcgpu

camera.ptx:
	$(NVCC) $(ARCH) $(NVCCFLAGS) $(NVCCLIBS) -ptx camera.cu

hits.ptx:
	$(NVCC) $(ARCH) $(NVCCFLAGS) $(NVCCLIBS) -ptx hits.cu

miss.ptx:
	$(NVCC) $(ARCH) $(NVCCFLAGS) $(NVCCLIBS) -ptx miss.cu

box.ptx:
	$(NVCC) $(ARCH) $(NVCCFLAGS) $(NVCCLIBS) -ptx box.cu

cylinder.ptx:
	$(NVCC) $(ARCH) $(NVCCFLAGS) $(NVCCLIBS) -ptx cylinder.cu

hex.ptx:
	$(NVCC) $(ARCH) $(NVCCFLAGS) $(NVCCLIBS) -ptx hex.cu

mt19937ar.o:
	$(CC) -c -O mt19937ar.cpp

mcgpu: 	$(ptx_objects) mt19937ar.o
	$(NVCC) -O mt19937ar.o $(ARCH) $(NVCCFLAGS) $(LIBS) $(NVCCLIBS) $(CUDPP) -o $@ mcgpu.cu

debug: 	$(ptx_objects) mt19937ar.o
	$(NVCC) -Xcompiler -rdynamic -lineinfo -O mt19937ar.o $(ARCH) $(NVCCFLAGS) $(LIBS) $(NVCCLIBS) $(CUDPP) -g -G -o mcgpu mcgpu.cu
