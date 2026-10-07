#pragma once

#include "aabb.h"

struct hit_record
{
	point3 p;
	vec3 normal;
	int mat_idx;
	float t;
	float u = 0.0f;
	float v = 0.0f;
	bool front_face;

	__device__ void set_face_normal(const ray& r, const vec3& outward_normal)
	{
		front_face = dot(r.direction(), outward_normal) < 0.0f;
		normal = front_face ? outward_normal : -outward_normal;
	}
};


