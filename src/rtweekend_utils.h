#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <iostream>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)

// Constants

#define PI 3.1415927f

// Utility Functions

inline void check_cuda(cudaError_t result, const char* const func, const char* const file, const int line)
{
	if (result)
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " \"" << cudaGetErrorString(result) << "\" at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

__host__ __device__ inline float degrees_to_radians(float degrees) {
	return degrees * PI / 180.0f;
}

__device__ inline float random_range(float min, float max, curandState* local_rand_state)
{
	return min + (max - min) * curand_uniform(local_rand_state);
}

__device__ inline int random_int(int min, int max, curandState* local_rand_state)
{
	return int(random_range(min, max + 1, local_rand_state));
}

inline float random_float() {
	// Returns a random real in [0,1).
	return std::rand() / (RAND_MAX + 1.0f);
}

inline float random_float(float min, float max) {
	// Returns a random real in [min,max).
	return min + (max - min) * random_float();
}

inline int random_int(int min, int max)
{
	return int(random_float(min, max + 1));
}

#include "vec3.h"
#include "interval.h"
#include "ray.h"
#include "color.h"
