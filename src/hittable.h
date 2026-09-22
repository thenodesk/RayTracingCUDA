#pragma once

#include "ray.h"

class material;

struct hit_record
{
    point3 p;
    vec3 normal;
    material* mat;
    float t;
    bool front_face;

    __device__ void set_face_normal(const ray& r, const vec3& outward_normal)
    {
        front_face = dot(r.direction(), outward_normal) < 0.0f;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable
{
public:
    __device__ virtual ~hittable() = default;

    __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;
};
