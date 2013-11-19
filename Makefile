#
CC =  /usr/local/Cellar/gcc46/4.6.4/bin/gcc-4.6
CXX = /usr/local/Cellar/gcc46/4.6.4/bin/g++-4.6
OPTIX = /Developer/OptiX
NVCC = nvcc -ccbin=/usr/local/Cellar/gcc46/4.6.4/bin
ARCH = -arch sm_30
C_FLAGS = -O3 -m64
NVCC_FLAGS = -m64  -use_fast_math
CURAND_LIBS = -lcurand
OPTIX_FLAGS = -I$(OPTIX)/include -L$(OPTIX)/lib64 
OPTIX_LIBS = -loptix 
CUDPP_PATH = /usr/local/cudpp-2.1/
CUDPP_FLAGS = -I/$(CUDPP_PATH)/include -L/$(CUDPP_PATH)/lib
CUDPP_LIBS = -lcudpp_hash -lcudpp
PYTHON_FLAGS = -I/System/Library/Frameworks/Python.framework/Headers
PYTHON_LIBS = -lpython2.7
PNG_FLAGS = -L/
PNG_LIBS = -lpng15
LIBS =


COBJS =	mt19937ar.o \
		print_banner.o \
		set_positions_rand.o \
		copy_points.o \
		macroscopic.o \
		microscopic.o \
		find_E_grid_index.o \
		find_E_grid_index_quad.o \
		sample_fission_spectra.o \
		sample_isotropic_directions.o \
		tally_spec.o \
		escatter.o \
		iscatter.o \
		cscatter.o \
		fission.o \
		absorb.o \
		make_mask.o \
		print_histories.o \
		pop_secondaries.o \
		flip_done.o \
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
	rm -f *.ptx *.o unionize.c gpu debug

camera.ptx:
	$(NVCC) $(ARCH) $(NVCC_FLAGS) $(OPTIX_FLAGS) $(OPTIX_LIBS) -ptx camera.cu

hits.ptx:
	$(NVCC) $(ARCH) $(NVCC_FLAGS) $(OPTIX_FLAGS) $(OPTIX_LIBS) -ptx hits.cu

miss.ptx:
	$(NVCC) $(ARCH) $(NVCC_FLAGS) $(OPTIX_FLAGS) $(OPTIX_LIBS) -ptx miss.cu

box.ptx:
	$(NVCC) $(ARCH) $(NVCC_FLAGS) $(OPTIX_FLAGS) $(OPTIX_LIBS) -ptx box.cu

cylinder.ptx:
	$(NVCC) $(ARCH) $(NVCC_FLAGS) $(OPTIX_FLAGS) $(OPTIX_LIBS) -ptx cylinder.cu

hex.ptx:
	$(NVCC) $(ARCH) $(NVCC_FLAGS) $(OPTIX_FLAGS) $(OPTIX_LIBS) -ptx hex.cu

mt19937ar.o:
	$(CXX) $(C_FLAGS) -c -O mt19937ar.cpp

main.o:
	$(NVCC) $(NVCC_FLAGS) $(OPTIX_FLAGS) $(CUDPP_FLAGS) $(PYTHON_FLAGS) -c -O main.cpp

print_banner.o:
	$(NVCC) $(NVCC_FLAGS) -c -O print_banner.cpp

set_positions_rand.o:
	$(NVCC) $(ARCH) $(NVCC_FLAGS) -c set_positions_rand.cu

find_E_grid_index.o:
	$(NVCC) $(ARCH) $(NVCC_FLAGS) -c find_E_grid_index.cu

find_E_grid_index_quad.o:
	$(NVCC) $(ARCH) $(NVCC_FLAGS) -c find_E_grid_index_quad.cu

sample_fission_spectra.o:
	$(NVCC) $(ARCH) $(NVCC_FLAGS) -c sample_fission_spectra.cu

sample_isotropic_directions.o:
	$(NVCC) $(ARCH) $(NVCC_FLAGS) -c sample_isotropic_directions.cu

macroscopic.o:
	$(NVCC) $(ARCH) $(NVCC_FLAGS) -c macroscopic.cu

microscopic.o:
	$(NVCC) $(ARCH) $(NVCC_FLAGS) -c microscopic.cu

copy_points.o:
	$(NVCC) $(ARCH) $(NVCC_FLAGS) -c copy_points.cu

tally_spec.o:
	$(NVCC) $(ARCH) $(NVCC_FLAGS) -c tally_spec.cu

escatter.o: 
	$(NVCC) $(ARCH) $(NVCC_FLAGS) -c escatter.cu

iscatter.o: 
	$(NVCC) $(ARCH) $(NVCC_FLAGS) -c iscatter.cu

cscatter.o: 
	$(NVCC) $(ARCH) $(NVCC_FLAGS) -c cscatter.cu

fission.o:
	$(NVCC) $(ARCH) $(NVCC_FLAGS) -c fission.cu

absorb.o:
	$(NVCC) $(ARCH) $(NVCC_FLAGS) -c absorb.cu

make_mask.o:
	$(NVCC) $(ARCH) $(NVCC_FLAGS) -c make_mask.cu

print_histories.o:
	$(NVCC) $(ARCH) $(NVCC_FLAGS) -c print_histories.cu

pop_secondaries.o:
	$(NVCC) $(ARCH) $(NVCC_FLAGS) -c pop_secondaries.cu

flip_done.o:
	$(NVCC) $(ARCH) $(NVCC_FLAGS) -c flip_done.cu

gpu: $(ptx_objects) $(COBJS)
	 $(NVCC) $(NVCC_FLAGS) $(OPTIX_FLAGS) $(CUDPP_FLAGS) $(PNG_FLAGS) $(CURAND_LIBS) $(OPTIX_LIBS) $(CUDPP_LIBS) $(PYTHON_LIBS) $(PNG_LIBS) $(COBJS) -o $@ 

debug: 	$(ptx_objects) $(COBJS)
	 $(NVCC) $(NVCC_FLAGS) $(OPTIX_FLAGS) $(CUDPP_FLAGS) $(PNG_FLAGS) $(CURAND_LIBS) $(OPTIX_LIBS) $(CUDPP_LIBS) $(PYTHON_LIBS) $(PNG_LIBS) $(COBJS) -g -G -o $@ 
