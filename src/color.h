#pragma once

#include "rtweekend_utils.h"

using color = vec3;

__host__ __device__ inline float linear_to_gamma(float linear_component)
{
	if (linear_component > 0.0f)
		return sqrt(linear_component);

	return 0.0f;
}

void write_color(unsigned char* out, vec3* fb, int x, int y, int stride, int channels = 3)
{
	int index = (y * stride + x);

	float r = fb[index].r();
	float g = fb[index].g();
	float b = fb[index].b();

	// Translate the [0,1] component values to the byte range [0,255].
	static const interval intensity(0.000f, 0.999f);
	int rbyte = int(256 * intensity.clamp(r));
	int gbyte = int(256 * intensity.clamp(g));
	int bbyte = int(256 * intensity.clamp(b));

	// Write out the pixel color components.
	out[index * channels + 0] = rbyte;
	out[index * channels + 1] = gbyte;
	out[index * channels + 2] = bbyte;
}
