#pragma once

#include "hittable.h"

class sphere
{
public:
    // Stationary Sphere
    __host__ __device__ sphere(const point3& center, float radius, bool constant_medium = false, float density = 0.0f)
        : center(center, vec3(0.0f, 0.0f, 0.0f)), radius(fmax(0.0f, radius)), neg_inv_density(density > 0.0f ? -1.0f / density : 0.0f), constant_medium(constant_medium)
    {
        vec3 rvec = vec3(radius, radius, radius);
        bbox = aabb(center - rvec, center + rvec);
    }

    // Moving Sphere
    __host__ __device__ sphere(const point3& center1, const point3& center2, float radius, bool constant_medium = false, float density = 0.0f)
        : center(center1, center2 - center1), radius(fmax(0.0f, radius)), neg_inv_density(density > 0.0f ? -1.0f / density : 0.0f), constant_medium(constant_medium)
    {
        vec3 rvec = vec3(radius, radius, radius);
        aabb box1(center.at(0) - rvec, center.at(0) + rvec);
        aabb box2(center.at(1) - rvec, center.at(1) + rvec);
        bbox = aabb(box1, box2);
    }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int mat_id, curandState* local_rand_state) const
    {
        if (constant_medium)
            return hit_volume(r, ray_t, rec, mat_id, local_rand_state);

        return hit_solid(r, ray_t, rec, mat_id);
    }

    __device__ bool hit_solid(const ray& r, interval ray_t, hit_record& rec, int mat_id) const
    {
        // Apply translation
        ray offset_r(r.origin() - translation, r.direction(), r.time());

        // Apply rotation
        point3 origin = world_to_object_space(offset_r.origin());
        vec3 direction = world_to_object_space(offset_r.direction());
        ray rotated_r(origin, direction, offset_r.time());

        point3 current_center = center.at(rotated_r.time());
        vec3 oc = current_center - rotated_r.origin();
        float a = rotated_r.direction().length_squared();
        float h = dot(rotated_r.direction(), oc);
        float c = oc.length_squared() - radius * radius;

        float discriminant = h * h - a * c;
        if (discriminant < 0.0f)
            return false;

        float sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        float root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root))
                return false;
        }

        rec.t = root;
        rec.p = object_to_world_space(rotated_r.at(rec.t)) + translation;
        vec3 outward_normal = (rotated_r.at(root) - current_center) / radius;
        rec.set_face_normal(rotated_r, outward_normal);
        rec.normal = object_to_world_space(rec.normal);
        get_sphere_uv(outward_normal, rec.u, rec.v);
        rec.mat_idx = mat_id;

        return true;
    }

    __device__ bool hit_volume(const ray& r, interval ray_t, hit_record& rec, int mat_id, curandState* local_rand_state) const
    {
        hit_record rec1, rec2;

        if (!hit_solid(r, interval::universe(), rec1, mat_id))
            return false;

        if (!hit_solid(r, interval(rec1.t + 0.0001f, INFINITY), rec2, mat_id))
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

    void translate(const vec3& offset)
    {
        translation += offset;
        update_bbox();
    }

    void rotate_y(float angle)
    {
        if (angle == 0.0f) return;
        sin_theta_y = std::sin(degrees_to_radians(angle));
        cos_theta_y = std::cos(degrees_to_radians(angle));
        update_bbox();
    }

    __host__ __device__ aabb bounding_box() const { return bbox; }

private:
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

    void update_bbox()
    {
        vec3 rvec(radius, radius, radius);
        point3 c0 = object_to_world_space(center.at(0.0f)) + translation;
        point3 c1 = object_to_world_space(center.at(1.0f)) + translation;   // = c0 se estacionária
        bbox = aabb(aabb(c0 - rvec, c0 + rvec), aabb(c1 - rvec, c1 + rvec));
    }

    __device__ static void get_sphere_uv(const point3& p, float& u, float& v)
    {
        // p: a given point on the sphere of radius one, centered at the origin.
        // u: returned value [0,1] of angle around the Y axis from X=-1.
        // v: returned value [0,1] of angle from Y=-1 to Y=+1.
        //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
        //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
        //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

        float theta = acos(-p.y());
        float phi = atan2(-p.z(), p.x()) + PI;

        u = phi / (2.0f * PI);
        v = theta / PI;
    }

private:
    ray center;
    float radius;
    vec3 translation{};
    float cos_theta_y = 1.0f, sin_theta_y = 0.0f;
    aabb bbox;

    float neg_inv_density;
    bool constant_medium;
};
