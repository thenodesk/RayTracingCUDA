#pragma once

#include "hittable.h"

class lambertian
{
public:
    lambertian(const color& albedo) : albedo(albedo) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state) const
    {
        vec3 scatter_direction = rec.normal + random_unit_vector(local_rand_state);

        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

private:
    color albedo;
};

class metal
{
public:
    metal(const color& albedo, float fuzz) : albedo(albedo), fuzz(fuzz < 1.0f ? fuzz : 1.0f) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state) const
    {
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        reflected = unit_vector(reflected) + (fuzz * random_unit_vector(local_rand_state));
        scattered = ray(rec.p, reflected);
        attenuation = albedo;
        return dot(scattered.direction(), rec.normal) > 0.0f;
    }

private:
    color albedo;
    float fuzz;
};

class dielectric
{
public:
    dielectric(float refraction_index) : refraction_index(refraction_index) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state) const
    {
        attenuation = color(1.0f, 1.0f, 1.0f);
        float ri = rec.front_face ? (1.0f / refraction_index) : refraction_index;

        vec3 unit_direction = unit_vector(r_in.direction());
        float cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0f);
        float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0f;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, ri) > curand_uniform(local_rand_state))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, ri);

        scattered = ray(rec.p, direction);
        return true;
    }

private:
    __device__ static float reflectance(float cosine, float refraction_index)
    {
        float r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
        r0 = r0 * r0;
        float m = 1.0f - cosine;
        return r0 + (1.0f - r0) * (m * m * m * m * m);
    }

private:
    float refraction_index;
};

enum MaterialType { MAT_LAMBERTIAN, MAT_METAL, MAT_DIELECTRIC };

struct MaterialData {
    MaterialType type;

    union {
        lambertian mat_diffuse;
        metal mat_metal;
        dielectric mat_glass;
    };

    MaterialData(const lambertian& mat) : type(MAT_LAMBERTIAN), mat_diffuse(mat) {}
    MaterialData(const metal& mat) : type(MAT_METAL), mat_metal(mat) {}
    MaterialData(const dielectric& mat) : type(MAT_DIELECTRIC), mat_glass(mat) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state) const
    {
        switch (type) {
        case MAT_LAMBERTIAN:
            return mat_diffuse.scatter(r_in, rec, attenuation, scattered, local_rand_state);
        case MAT_METAL:
            return mat_metal.scatter(r_in, rec, attenuation, scattered, local_rand_state);
        case MAT_DIELECTRIC:
            return mat_glass.scatter(r_in, rec, attenuation, scattered, local_rand_state);
        }
        return false;
    }
};
