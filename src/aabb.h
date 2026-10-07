#pragma once

#include "rtweekend_utils.h"

#include "interval.h"

class aabb
{
public:
	__host__ __device__ aabb() {}

	__host__ __device__ aabb(const interval& x, const interval& y, const interval& z) : x(x), y(y), z(z)
	{
		pad_to_minimums();
	}

	__host__ __device__ aabb(const point3& a, const point3& b)
	{
		x = (a[0] <= b[0]) ? interval(a[0], b[0]) : interval(b[0], a[0]);
		y = (a[1] <= b[1]) ? interval(a[1], b[1]) : interval(b[1], a[1]);
		z = (a[2] <= b[2]) ? interval(a[2], b[2]) : interval(b[2], a[2]);

		pad_to_minimums();
	}

	__host__ __device__ aabb(const aabb& box0, const aabb& box1)
	{
		x = interval(box0.x, box1.x);
		y = interval(box0.y, box1.y);
		z = interval(box0.z, box1.z);
	}

	__host__ __device__ const interval& axis_interval(int n) const
	{
		if (n == 1) return y;
		if (n == 2) return z;
		return x;
	}

	__device__ bool hit(const point3& orig, const vec3& inv_dir, interval ray_t) const
	{
		for (int axis = 0; axis < 3; axis++)
		{
			const interval& ax = axis_interval(axis);
			//float localMax = ax.size();
			//float localRayOrig = orig[axis] - ax.min;

			float t0 = (ax.min - orig[axis]) * inv_dir[axis];
			float t1 = (ax.max - orig[axis]) * inv_dir[axis];

			//float t0 = (/*0.0f*/ - localRayOrig) * inv_dir[axis];
			//float t1 = (localMax - localRayOrig) * inv_dir[axis];

			if (t0 > t1)
			{
				float tmp = t0;
				t0 = t1;
				t1 = tmp;
			}

			t1 *= 1.0000008f;

			ray_t.min = fmaxf(t0, ray_t.min);
			ray_t.max = fminf(t1, ray_t.max);

			if (ray_t.max < ray_t.min)
				return false;
		}

		return true;
	}

	__host__ __device__ int longest_axis() const
	{
		// Returns the index of the longest axis of the bounding box.
		if (x.size() > y.size())
			return x.size() > z.size() ? 0 : 2;
		else
			return y.size() > z.size() ? 1 : 2;
	}

	__host__ __device__ static const aabb empty() {
		return aabb(interval::empty(), interval::empty(), interval::empty());
	}

	__host__ __device__ static const aabb universe() {
		return aabb(interval::universe(), interval::universe(), interval::universe());
	}

	__host__ __device__ aabb& operator+=(const vec3& offset)
	{
		x += offset.x();
		y += offset.y();
		z += offset.z();

		return *this;
	}

private:
	__host__ __device__ void pad_to_minimums()
	{
		// Adjust the AABB so that no side is narrower than some delta, padding if necessary.
		float delta = 0.0001f;
		if (x.size() < delta) x = x.expand(delta);
		if (y.size() < delta) y = y.expand(delta);
		if (z.size() < delta) z = z.expand(delta);
	}

public:
	interval x{}, y{}, z{};
};

__host__ __device__ inline aabb operator+(const aabb& bbox, const vec3& offset)
{
	return aabb(bbox.x + offset.x(), bbox.y + offset.y(), bbox.z + offset.z());
}

__host__ __device__ inline aabb operator+(const vec3& offset, const aabb& bbox)
{
	return bbox + offset;
}
