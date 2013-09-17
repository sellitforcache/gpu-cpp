#
CC = /usr/bin/gcc
CXX = /usr/bin/g++
OPTIX=/Developer/OptiX
NVCC = /usr/local/cuda/bin/nvcc
ARCH = -arch sm_30
CFLAGS = -O3 -m64 -I$(OPTIX)/include -L$(OPTIX)/lib64
NVCCFLAGS = -m64 -I$(OPTIX)/include 
NVCCLIBS = -L$(OPTIX)/lib64 -loptix
CUDPP = -lcudpp_hash -lcudpp -lcurand
LIBS =


COBJS =	mt19937ar.o \
		print_banner.o \
		main.o

ptx_objects = 	camera.ptx \
				hits.ptx \
				miss.ptx \
				box.ptx \
				cylinder.ptx \
				hex.ptx\

all:  	$(ptx_objects) \
		$(COBJS) \
		gpu \

clean:
	rm -f *.ptx *.o gpu

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
	$(CXX) -c -O mt19937ar.cpp

main.o:
	$(NVCC) $(CFLAGS) -c -O main.cpp

print_banner.o:
	$(NVCC) $(CFLAGS) -c -O print_banner.cpp

gpu: $(ptx_objects) $(COBJS)
	 $(NVCC) $(CFLAGS) $(NVCCLIBS) $(COBJS) -o $@ 

debug: 	$(ptx_objects) mt19937ar.o
	$(NVCC) -Xcompiler -rdynamic -lineinfo -O mt19937ar.o $(ARCH) $(NVCCFLAGS) $(LIBS) $(NVCCLIBS) $(CUDPP) -g -G -o mcgpu mcgpu.cu
