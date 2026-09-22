#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <iostream>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)

// Constants
#define PI 3.1415927f

// Utility Functions

void check_cuda(cudaError_t result, const char* const func, const char* const file, const int line)
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

__device__ inline float degrees_to_radians(float degrees) {
	return degrees * PI / 180.0f;
}

#include "vec3.h"
#include "interval.h"
#include "ray.h"
#include "color.h"
