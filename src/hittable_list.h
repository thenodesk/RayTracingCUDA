#pragma once

#include "hittable.h"

class hittable_list : public hittable
{
public:
    __device__ hittable_list() {}
    __device__ hittable_list(hittable** list, int n) { objects = list; list_size = n; }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override
    {
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = ray_t.max;

        for (int i = 0; i < list_size; i++)
        {
            if (objects[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec))
            {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        
        return hit_anything;
    }

public:
    hittable** objects;
    int list_size;
};
