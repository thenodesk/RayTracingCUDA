#pragma once

#include "assets.h"
#include "onb.h"

class sphere_pdf
{
public:
    __device__ sphere_pdf() {}

    __device__ float value(const vec3& direction, curandState* local_rand_state) const
    {
        return 1.0f / (4.0f * PI);
    }

    __device__ vec3 generate(curandState* local_rand_state) const
    {
        return random_unit_vector(local_rand_state);
    }
};

class cosine_pdf
{
public:
    __device__ cosine_pdf(const vec3& w) : uvw(w) {}

    __device__ float value(const vec3& direction, curandState* local_rand_state) const
    {
        float cosine_theta = dot(unit_vector(direction), uvw.w());
        return fmaxf(0.0f, cosine_theta / PI);
    }

    __device__ vec3 generate(curandState* local_rand_state) const
    {
        return uvw.transform(random_cosine_direction(local_rand_state));
    }

private:
    onb uvw;
};

class hittable_pdf
{
public:
    __device__ hittable_pdf(HittableObject* objects, int* indexes, int objects_size, const point3& origin)
        : objects(objects), indexes(indexes), size(objects_size), origin(origin) { }

    __device__ float value(const vec3& direction, curandState* local_rand_state) const
    {
        float weight = 1.0f / size;
        float sum = 0.0f;

        for (int i = 0; i < size; i++)
            sum += weight * objects[indexes[i]].pdf_value(origin, direction, local_rand_state);

        return sum;
    }

    __device__ vec3 generate(curandState* local_rand_state) const
    {
        int random_idx = random_int(0, size - 1, local_rand_state);
        return objects[indexes[random_idx]].random(origin, local_rand_state);
    }

private:
    HittableObject* objects;
    int* indexes;
    int size;
    point3 origin;
};

template<class T, class U>
class mixture_pdf
{
public:
    __device__ mixture_pdf(T p0, U p1) : p0(p0), p1(p1) {}

    __device__ float value(const vec3& direction, curandState* local_rand_state) const
    {
        return 0.5 * p0.value(direction, local_rand_state) + 0.5 * p1.value(direction, local_rand_state);
    }

    __device__ vec3 generate(curandState* local_rand_state) const
    {
        if (curand_uniform(local_rand_state) < 0.5)
            return p0.generate(local_rand_state);
        else
            return p1.generate(local_rand_state);
    }

private:
    T p0;
    U p1;
};
