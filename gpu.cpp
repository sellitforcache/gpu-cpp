//library and SDK includes
#include <optix_world.h>
#include <optix_cuda_interop.h>
#include <cuda.h>
#include <curand.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <regex.h>
#include <assert.h>
#include <string.h>
#include <sys/errno.h>
#include <unistd.h>
#include <cudpp_hash.h>
#include "mt19937ar.h"

// all mt classes
#include "warp.h"

// preprocessor constants and flags
#define NUM_THREADS 256
#define RNUM_PER_THREAD 8//12
#define ISOTOPES_MAX 5
#define print_errors 0


