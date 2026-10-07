#pragma once

#include "rtweekend_utils.h"
#include "aabb.h"
#include "sphere.h"
#include "quad.h"

class camera;
class solid_color;
class checker_texture;
class image_texture;
class noise_texture;
class rtw_image;

struct MaterialData;
struct TextureData;

enum HittableType { HIT_SPHERE, HIT_QUAD };

struct HittableObject {
    HittableType type;
    int material_id;

    union {
        sphere sphere_obj;
        quad quad_obj;
    };

    HittableObject(const sphere& obj, int mat_id) : type(HIT_SPHERE), material_id(mat_id), sphere_obj(obj) {}
    HittableObject(const quad& obj, int mat_id) : type(HIT_QUAD), material_id(mat_id), quad_obj(obj) {}

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* local_rand_state)
    {
        switch (type)
        {
            case HIT_SPHERE:
                return sphere_obj.hit(r, ray_t, rec, material_id, local_rand_state);
            case HIT_QUAD:
                return quad_obj.hit(r, ray_t, rec, material_id);
        }
        return false;
    }

    __host__ __device__ aabb bounding_box() const
    {
        switch (type) {
        case HIT_SPHERE:
            return sphere_obj.bounding_box();
        case HIT_QUAD:
            return quad_obj.bounding_box();
        }
        return aabb();
    }

    __host__ __device__ bool is_child() const
    {
        return type == HIT_QUAD && quad_obj.is_child();
    }

    void translate(const vec3& v)
    {
        switch (type) {
        case HIT_SPHERE:
            return sphere_obj.translate(v);
        case HIT_QUAD:
            return quad_obj.translate(v);
        }
    }

    void rotate_y(float angle)
    {
        switch (type) {
        case HIT_SPHERE:
            return sphere_obj.rotate_y(angle);
        case HIT_QUAD:
            return quad_obj.rotate_y(angle);
        }
    }
};

enum InstanceType { INST_BOX };

struct HittableInstance {
    InstanceType type;
    int material_id;

    union {
        box box_inst;
    };

    HittableInstance(const box& inst, int mat_id) : type(INST_BOX), material_id(mat_id), box_inst(inst) {}

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, HittableObject* objects, curandState* local_rand_state)
    {
        switch (type)
        {
        case INST_BOX:
            return box_inst.hit(r, ray_t, rec, material_id, objects, local_rand_state);
        }
        return false;
    }

    __host__ __device__ aabb bounding_box() const
    {
        switch (type) {
        case INST_BOX:
            return box_inst.bounding_box();
        }
        return aabb();
    }
};

enum PrimType { PRIM_SPHERE, PRIM_QUAD, PRIM_BOX };

struct Primitive {
    PrimType type;
    int index;
    aabb bbox;
    aabb bounding_box() const { return bbox; }
};

struct BVHNode {
    aabb bbox;
    int left;   // Left child index (or -1)
    int right;  // Right child index (or -1)

    int object_index = -1;
    PrimType type;
    int axis = 0;

};

struct HittableList {
    HittableObject* objects;
    HittableInstance* instances;
    BVHNode* bvh_nodes;
    int bvh_count;

    // O método hit da lista que percorre o array contíguo sequencialmente
    //__device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const {
    //    hit_record temp_rec;
    //    bool hit_anything = false;
    //    float closest_so_far = ray_t.max;

    //    // Como 'objects' é contíguo, o cache da GPU adora esse loop linear!
    //    for (int i = 0; i < count; i++) {
    //        if (objects[i].hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
    //            hit_anything = true;
    //            closest_so_far = temp_rec.t;
    //            rec = temp_rec;
    //        }
    //    }
    //    return hit_anything;
    //}

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* local_rand_state) const
    {
        if (bvh_count == 0) return false;

        const point3& orig = r.origin();
        const vec3& dir = r.direction();
        const vec3 inv_dir(1.0f / dir.x(), 1.0f / dir.y(), 1.0f / dir.z());

        int stack[64];
        int stack_ptr = 0;

        stack[stack_ptr++] = 0;

        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = ray_t.max;

        while (stack_ptr > 0)
        {
            int node_idx = stack[--stack_ptr];
            const BVHNode& node = bvh_nodes[node_idx];

            if (!node.bbox.hit(orig, inv_dir, interval(ray_t.min, closest_so_far)))
                continue;

            if (node.object_index != -1)
            {
                interval range(ray_t.min, closest_so_far);
                bool intersected = false;

                switch (node.type) {
                case PRIM_SPHERE:
                case PRIM_QUAD:
                    intersected = objects[node.object_index].hit(r, range, temp_rec, local_rand_state); break;
                case PRIM_BOX:
                    intersected = instances[node.object_index].hit(r, range, temp_rec, objects, local_rand_state); break;
                }

                if (intersected) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }
            else
            {
                if (stack_ptr < 62)
                {
                    if (dir[node.axis] > 0.0f)
                    {
                        stack[stack_ptr++] = node.right;
                        stack[stack_ptr++] = node.left;
                    }
                    else
                    {
                        stack[stack_ptr++] = node.left;
                        stack[stack_ptr++] = node.right;
                    }
                }
            }
        }

        return hit_anything;
    }
};

struct DeviceScene
{
    unsigned char** images_addr;
    int images_count;

    struct
    {
        solid_color* solid;
        checker_texture* checker;
        image_texture* image;
        noise_texture* noise;
    } texture_types;

    camera* camera;

    HittableList world;
    MaterialData* materials;
    TextureData* textures;
    rtw_image* images;
};