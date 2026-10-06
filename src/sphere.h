#pragma once

#include "hittable.h"
#include "material.h"
#include "vec3.h"

class sphere
{
public:
    __host__ __device__ sphere(const point3& center, float radius)
        : center(center), radius(fmax(0.0f, radius)) {}

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int mat_id) const
    {
        vec3 oc = center - r.origin();
        float a = r.direction().length_squared();
        float h = dot(r.direction(), oc);
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
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat_idx = mat_id;

        return true;
    }

private:
    point3 center;
    float radius;
};
