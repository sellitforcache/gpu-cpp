#
CC = /usr/bin/gcc
CXX = /usr/bin/g++
CYTHON = /usr/local/bin/cython
OPTIX = /Developer/OptiX
NVCC = /usr/local/cuda/bin/nvcc
ARCH = -arch sm_30
C_FLAGS = -O3 -m64
NVCC_FLAGS = -m64 
CURAND_LIBS = -lcurand
OPTIX_FLAGS = -I$(OPTIX)/include -L$(OPTIX)/lib64 
OPTIX_LIBS = -loptix 
CUDPP_PATH = /usr/local/cudpp-2.0/
CUDPP_FLAGS = -I/$(CUDPP_PATH)/include -L/$(CUDPP_PATH)/lib
CUDPP_LIBS = -lcudpp_hash -lcudpp
PYTHON_FLAGS = -I/System/Library/Frameworks/Python.framework/Headers/
PYTHON_LIBS = -lpython2.7
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
	rm -f *.ptx *.o unionize.c gpu

camera.ptx:
	$(NVCC) $(ARCH) $(NVCCC_FLAGS) $(OPTIX_FLAGS) $(OPTIX_LIBS) -ptx camera.cu

hits.ptx:
	$(NVCC) $(ARCH) $(NVCCC_FLAGS) $(OPTIX_FLAGS) $(OPTIX_LIBS) -ptx hits.cu

miss.ptx:
	$(NVCC) $(ARCH) $(NVCCC_FLAGS) $(OPTIX_FLAGS) $(OPTIX_LIBS) -ptx miss.cu

box.ptx:
	$(NVCC) $(ARCH) $(NVCCC_FLAGS) $(OPTIX_FLAGS) $(OPTIX_LIBS) -ptx box.cu

cylinder.ptx:
	$(NVCC) $(ARCH) $(NVCCC_FLAGS) $(OPTIX_FLAGS) $(OPTIX_LIBS) -ptx cylinder.cu

hex.ptx:
	$(NVCC) $(ARCH) $(NVCCC_FLAGS) $(OPTIX_FLAGS) $(OPTIX_LIBS) -ptx hex.cu

mt19937ar.o:
	$(CXX) $(C_FLAGS) -c -O mt19937ar.cpp

main.o:
	$(NVCC) $(NVCCC_FLAGS) $(NVCC_FLAGS) $(OPTIX_FLAGS) $(CUDPP_FLAGS) $(PYTHON_FLAGS) -c -O main.cpp

print_banner.o:
	$(NVCC) $(NVCC_FLAGS) -c -O print_banner.cpp

gpu: $(ptx_objects) $(COBJS)
	 $(NVCC) $(NVCC_FLAGS) $(OPTIX_FLAGS) $(CUDPP_FLAGS) $(CURAND_LIBS) $(OPTIX_LIBS) $(CUDPP_LIBS) $(PYTHON_LIBS) $(COBJS) -o $@ 

debug: 	$(ptx_objects) $(COBJS)
	$(NVCC) $(NVCC_FLAGS) $(OPTIX_FLAGS) $(CUDPP_FLAGS) $(CURAND_LIBS) $(OPTIX_LIBS) $(CUDPP_LIBS) $(PYTHON_LIBS) $(COBJS) -g -G -o $@
