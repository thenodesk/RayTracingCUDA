#pragma once

#include "hittable.h"
#include "pdf.h"
#include "texture.h"
#include "assets.h"

class lambertian
{
public:
    __host__ __device__ lambertian() {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, DeviceScene& d_scene, int tex_id, curandState* local_rand_state) const
    {
        attenuation = d_scene.textures[tex_id].value(rec.u, rec.v, rec.p, &d_scene);
        return true;
    }

    __device__ float scattering_pdf(const ray& r_in, const hit_record& rec, ray& scattered) const
    {
        float cos_theta = dot(rec.normal, unit_vector(scattered.direction()));
        return cos_theta < 0.0f ? 0.0f : cos_theta / PI;
    }
};

class metal
{
public:
    __host__ __device__ metal(float fuzz) : fuzz(fuzz < 1.0f ? fuzz : 1.0f) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, DeviceScene& d_scene, int tex_id, curandState* local_rand_state) const
    {
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        reflected = unit_vector(reflected) + (fuzz * random_unit_vector(local_rand_state));
        scattered = ray(rec.p, reflected, r_in.time());
        attenuation = d_scene.textures[tex_id].value(rec.u, rec.v, rec.p, &d_scene);
        return dot(scattered.direction(), rec.normal) > 0.0f;
    }

private:
    float fuzz;
};

class dielectric
{
public:
    __host__ __device__ dielectric(float refraction_index) : refraction_index(refraction_index) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, DeviceScene& d_scene, int tex_id, curandState* local_rand_state) const
    {
        attenuation = tex_id == -1 ? color(1.0f, 1.0f, 1.0f) : d_scene.textures[tex_id].value(rec.u, rec.v, rec.p, &d_scene);
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

        scattered = ray(rec.p, direction, r_in.time());
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

class diffuse_light
{
public:
    diffuse_light() {}

    __device__ color emitted(const ray& r_in, const hit_record& rec, float u, float v, const point3& p, DeviceScene& d_scene, int tex_id) const
    {
        if (!rec.front_face)
            return color(0.0f, 0.0f, 0.0f);

        return d_scene.textures[tex_id].value(u, v, p, &d_scene);
    }
};

class isotropic
{
public:
    isotropic() {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, DeviceScene& d_scene, int tex_id, curandState* local_rand_state) const
    {
        attenuation = d_scene.textures[tex_id].value(rec.u, rec.v, rec.p, &d_scene);
        return true;
    }

    __device__ float scattering_pdf(const ray& r_in, const hit_record& rec, ray& scattered) const
    {
        return 1.0f / (4.0f * PI);
    }
};

enum MaterialType { MAT_LAMBERTIAN, MAT_METAL, MAT_DIELECTRIC, MAT_DIFFUSE_LIGHT, MAT_ISOTROPIC };

struct MaterialData {
    MaterialType type;
    int texture_id;

    union {
        lambertian mat_diffuse;
        metal mat_metal;
        dielectric mat_glass;
        diffuse_light mat_diffuse_light;
        isotropic mat_isotropic;
    };

    MaterialData(const lambertian& mat, int tex_id) : type(MAT_LAMBERTIAN), mat_diffuse(mat), texture_id(tex_id) {}
    MaterialData(const metal& mat, int tex_id) : type(MAT_METAL), mat_metal(mat), texture_id(tex_id) {}
    MaterialData(const dielectric& mat, int tex_id = -1) : type(MAT_DIELECTRIC), mat_glass(mat), texture_id(tex_id) {}
    MaterialData(const diffuse_light& mat, int tex_id) : type(MAT_DIFFUSE_LIGHT), mat_diffuse_light(mat), texture_id(tex_id) {}
    MaterialData(const isotropic& mat, int tex_id) : type(MAT_ISOTROPIC), mat_isotropic(mat), texture_id(tex_id) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, DeviceScene& d_scene, curandState* local_rand_state) const
    {
        switch (type) {
        case MAT_LAMBERTIAN:
            return mat_diffuse.scatter(r_in, rec, attenuation, scattered, d_scene, texture_id, local_rand_state);
        case MAT_METAL:
            return mat_metal.scatter(r_in, rec, attenuation, scattered, d_scene, texture_id, local_rand_state);
        case MAT_DIELECTRIC:
            return mat_glass.scatter(r_in, rec, attenuation, scattered, d_scene, texture_id, local_rand_state);
        case MAT_ISOTROPIC:
            return mat_isotropic.scatter(r_in, rec, attenuation, scattered, d_scene, texture_id, local_rand_state);
        }
        return false;
    }

    __device__ color emitted(const ray& r_in, const hit_record& rec, float u, float v, const point3& p, DeviceScene& d_scene) const
    {
        switch (type) {
        case MAT_DIFFUSE_LIGHT:
            return mat_diffuse_light.emitted(r_in, rec, u, v, p, d_scene, texture_id);
        }
        return color(0.0f, 0.0f, 0.0f);
    }

    __device__ float scattering_pdf(const ray& r_in, const hit_record& rec, ray& scattered) const
    {
        switch (type) {
        case MAT_LAMBERTIAN:
            return mat_diffuse.scattering_pdf(r_in, rec, scattered);
        case MAT_ISOTROPIC:
            return mat_isotropic.scattering_pdf(r_in, rec, scattered);
        }
        return 1.0f;
    }

    template <class T>
    __device__ inline void apply_material_to_pdf(T pdf, const ray& r_in, const hit_record& rec_in, ray& scattered, float& pdf_value, curandState* local_rand_state) const
    {
        switch (type)
        {
        case MAT_LAMBERTIAN:
        {
            mixture_pdf<T, cosine_pdf> mixed_pdf(pdf, cosine_pdf(rec_in.normal));
            scattered = ray(rec_in.p, mixed_pdf.generate(local_rand_state), r_in.time());
            pdf_value = mixed_pdf.value(scattered.direction(), local_rand_state);
            break;
        }
        case MAT_ISOTROPIC:
        {
            mixture_pdf<T, sphere_pdf> mixed_pdf(pdf, sphere_pdf());
            scattered = ray(rec_in.p, mixed_pdf.generate(local_rand_state), r_in.time());
            pdf_value = mixed_pdf.value(scattered.direction(), local_rand_state);
            break;
        }
        default:
            pdf_value = 1.0f;
        }
    }
};
