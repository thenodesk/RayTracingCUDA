#pragma once

#include "hittable.h"

class quad
{
public:
	quad(const point3& Q, const vec3& u, const vec3& v, bool child = false)
		: Q(Q), u(u), v(v), child(child)
	{
		vec3 n = cross(u, v);
		normal = unit_vector(n);
		D = dot(normal, Q);
		w = n / n.length_squared();

		set_bounding_box();
	}

	void set_bounding_box()
	{
		// Compute the bounding box of all four vertices.
		aabb bbox_diagonal1 = aabb(Q, Q + u + v);
		aabb bbox_diagonal2 = aabb(Q + u, Q + v);
		bbox = aabb(bbox_diagonal1, bbox_diagonal2);
	}

	void translate(const vec3& offset)
	{
		translation += offset;
		update_bbox();
	}

	void rotate_y(float angle)
	{
		if (angle == 0) return;

		sin_theta_y = std::sin(degrees_to_radians(angle));
		cos_theta_y = std::cos(degrees_to_radians(angle));

		update_bbox();
	}

	__host__ __device__ aabb bounding_box() const { return bbox; }

	__host__ __device__ bool is_child() const { return child; }

	__device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int mat_id, bool child_access = false) const
	{
		if (child && !child_access) return false;

		// Apply translation
		ray offset_r(r.origin() - translation, r.direction(), r.time());

		// Apply rotation
		point3 origin = world_to_object_space(offset_r.origin());
		vec3 direction = world_to_object_space(offset_r.direction());
		ray rotated_r(origin, direction, offset_r.time());

		float denom = dot(normal, rotated_r.direction());

		// No hit if the ray is parallel to the plane.
		if (fabs(denom) < 1e-8f) return false;

		// Return false if the hit point parameter t is outside the ray interval.
		float t = (D - dot(normal, rotated_r.origin())) / denom;
		if (!ray_t.contains(t)) return false;

		// Determine if the hit point lies within the planar shape using its plane coordinates.
		point3 intersection = rotated_r.at(t);
		vec3 planar_hitpoint_vector = intersection - Q;
		float alpha = dot(w, cross(planar_hitpoint_vector, v));
		float beta = dot(w, cross(u, planar_hitpoint_vector));

		if (!is_interior(alpha, beta, rec)) return false;

		// Ray hits the 2D shape; set the rest of the hit record and return true.
		rec.t = t;
		rec.p = object_to_world_space(intersection) + translation;
		rec.mat_idx = mat_id;
		rec.set_face_normal(rotated_r, normal);
		rec.normal = object_to_world_space(rec.normal);

		return true;
	}

	__device__ bool is_interior(float a, float b, hit_record& rec) const
	{
		interval unit_interval = interval(0.0f, 1.0f);
		// Given the hit point in plane coordinates, return false if it is outside the
		// primitive, otherwise set the hit record UV coordinates and return true.
		if (!unit_interval.contains(a) || !unit_interval.contains(b)) return false;

		rec.u = a;
		rec.v = b;

		return true;
	}

private:
	void update_bbox()
	{
		point3 mn(INFINITY, INFINITY, INFINITY);
		point3 mx(-INFINITY, -INFINITY, -INFINITY);

		for (int i = 0; i < 2; i++)
			for (int j = 0; j < 2; j++)
				for (int k = 0; k < 2; k++)
				{
					vec3 c(i ? bbox.x.max : bbox.x.min,
						j ? bbox.y.max : bbox.y.min,
						k ? bbox.z.max : bbox.z.min);
					c = object_to_world_space(c) + translation;

					for (int a = 0; a < 3; a++)
					{
						mn[a] = std::fmin(mn[a], c[a]);
						mx[a] = std::fmax(mx[a], c[a]);
					}
				}

		bbox = aabb(mn, mx);
	}

	__host__ __device__ inline vec3 world_to_object_space(const vec3& v) const
	{
		if (cos_theta_y == 1.0f && sin_theta_y == 0.0f) return v;

		return vec3(
			(cos_theta_y * v.x()) - (sin_theta_y * v.z()),
			v.y(),
			(sin_theta_y * v.x()) + (cos_theta_y * v.z())
		);
	}

	__host__ __device__ inline vec3 object_to_world_space(const vec3& v) const
	{
		if (cos_theta_y == 1.0f && sin_theta_y == 0.0f) return v;

		return vec3(
			(cos_theta_y * v.x()) + (sin_theta_y * v.z()),
			v.y(),
			(-sin_theta_y * v.x()) + (cos_theta_y * v.z())
		);
	}

private:
	point3 Q;
	vec3 u, v;
	vec3 w;
	aabb bbox;
	vec3 normal;
	vec3 translation{};
	float cos_theta_y = 1.0f, sin_theta_y = 0.0f;
	float D;

	bool child = false;
};

class box
{
public:
	box(const int(&sides_idx)[6], bool constant_medium = false, float density = 0.0f)
		: neg_inv_density(density > 0.0f ? -1.0f / density : 0.0f), constant_medium(constant_medium)
	{
		for (int i = 0; i < 6; i++)
			indexes[i] = sides_idx[i];
	}

	void set_bounding_box(aabb box_bbox)
	{
		bbox = box_bbox;
	}

	__host__ __device__ aabb bounding_box() const { return bbox; }

	template <class Obj>
	__device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int mat_id, const Obj* objects, curandState* local_rand_state) const
	{
		if (constant_medium)
			return hit_volume(r, ray_t, rec, mat_id, objects, local_rand_state);

		return hit_quads(r, ray_t, rec, mat_id, objects);
	}

	template <class Obj>
	__device__ bool hit_volume(const ray& r, interval ray_t, hit_record& rec, int mat_id, const Obj* objects, curandState* local_rand_state) const
	{
		hit_record rec1, rec2;

		if (!hit_quads(r, interval::universe(), rec1, mat_id, objects))
			return false;

		if (!hit_quads(r, interval(rec1.t + 0.0001f, INFINITY), rec2, mat_id, objects))
			return false;

		if (rec1.t < ray_t.min) rec1.t = ray_t.min;
		if (rec2.t > ray_t.max) rec2.t = ray_t.max;

		if (rec1.t >= rec2.t)
			return false;

		if (rec1.t < 0.0f)

			rec1.t = 0.0f;

		float ray_length = r.direction().length();
		float distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
		float hit_distance = neg_inv_density * log(curand_uniform(local_rand_state));

		if (hit_distance > distance_inside_boundary)
			return false;

		rec.t = rec1.t + hit_distance / ray_length;
		rec.p = r.at(rec.t);
		rec.normal = vec3(1.0f, 0.0f, 0.0f); // arbitrary
		rec.front_face = true;				 // also arbitrary
		rec.mat_idx = mat_id;

		return true;
	}

	template <class Obj>
	__device__ bool hit_quads(const ray& r, interval ray_t, hit_record& rec, int mat_id, const Obj* objects) const
	{
		hit_record temp_rec;
		bool hit_anything = false;
		float closest_so_far = ray_t.max;
		for (int i = 0; i < 6; i++)
		{
			if (objects[indexes[i]].quad_obj.hit(r, interval(ray_t.min, closest_so_far), temp_rec, mat_id, true))
			{
				hit_anything = true;
				closest_so_far = temp_rec.t;
				rec = temp_rec;
			}
		}

		return hit_anything;
	}

public:
	int indexes[6];
	aabb bbox;

	float neg_inv_density;
	bool constant_medium;
};
