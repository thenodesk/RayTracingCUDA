#pragma once

#include "hittable.h"
#include "material.h"
#include "vec3.h"

class sphere : public hittable
{
public:
    __device__ sphere(const point3& center, float radius, material* mat)
        : center(center), radius(fmax(0.0f, radius)), mat_ptr(mat) {}

    __device__ ~sphere()
    {
        delete mat_ptr;
        mat_ptr = nullptr;
    }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override
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
        rec.mat = mat_ptr;

        return true;
    }

private:
    point3 center;
    float radius;
    material* mat_ptr;
};
