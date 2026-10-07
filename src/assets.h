#pragma once

#include "rtweekend_utils.h"
#include "sphere.h"

class camera;

struct MaterialData;
struct TextureData;

enum HittableType { HIT_SPHERE, HIT_QUAD };

struct HittableObject {
    HittableType type;
    int material_id;

    union {
        sphere sphere_obj;
    };

    HittableObject(const sphere& obj, int mat_id) : type(HIT_SPHERE), material_id(mat_id), sphere_obj(obj) {}

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec)
    {
        switch (type)
        {
            case HIT_SPHERE:
                return sphere_obj.hit(r, ray_t, rec, material_id);
        }
        return false;
    }
};

struct HittableList {
    HittableObject* objects;
    int count;

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const
    {
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = ray_t.max;

        for (int i = 0; i < count; i++) {
            if (objects[i].hit(r, interval(ray_t.min, closest_so_far), temp_rec))
            {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }
};

struct DeviceScene
{
    camera* camera;

    HittableList world;
    MaterialData* materials;
};