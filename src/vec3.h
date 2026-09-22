#pragma once

#include "rtweekend_utils.h"

class vec3
{
public:
    __host__ __device__ vec3() : e{ 0.0f, 0.0f, 0.0f } {}
    __host__ __device__ vec3(float e0, float e1, float e2) : e{ e0, e1, e2 } {}

    __host__ __device__ float x() const { return e[0]; }
    __host__ __device__ float y() const { return e[1]; }
    __host__ __device__ float z() const { return e[2]; }
    __host__ __device__ float r() const { return e[0]; }
    __host__ __device__ float g() const { return e[1]; }
    __host__ __device__ float b() const { return e[2]; }

    __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ float operator[](int i) const { return e[i]; }
    __host__ __device__ float& operator[](int i) { return e[i]; }

    __host__ __device__ vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ vec3& operator*=(const vec3& v) {
        e[0] *= v.e[0];
        e[1] *= v.e[1];
        e[2] *= v.e[2];
        return *this;
    }

    __host__ __device__ vec3& operator*=(float t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ vec3& operator/=(float t) {
        return *this *= 1.0f / t;
    }

    __host__ __device__ float length() const {
        return sqrt(length_squared());
    }

    __host__ __device__ float length_squared() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }
    
    __host__ __device__ bool near_zero() const {
        float s = 1e-8f;
        return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
    }

    __device__ static vec3 random(curandState* local_rand_state)
    {
        return vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state));
    }

    __device__ static vec3 random(float min, float max, curandState* local_rand_state)
    {
        return vec3(min + curand_uniform(local_rand_state) * (max - min), 
                    min + curand_uniform(local_rand_state) * (max - min),
                    min + curand_uniform(local_rand_state) * (max - min));
    }

public:
    float e[3];
};

using point3 = vec3;


// Vector Utility Functions

inline std::ostream& operator<<(std::ostream& out, const vec3& v)
{
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v)
{
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v)
{
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v)
{
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v)
{
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, float t)
{
    return t * v;
}

__host__ __device__ inline vec3 operator/(const vec3& v, float t)
{
    return (1 / t) * v;
}

__host__ __device__ inline float dot(const vec3& u, const vec3& v)
{
    return u.e[0] * v.e[0]
        + u.e[1] * v.e[1]
        + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v)
{
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
        u.e[2] * v.e[0] - u.e[0] * v.e[2],
        u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(const vec3& v)
{
    return v / v.length();
}

__device__ inline vec3 random_in_unit_disk(curandState* local_rand_state) {
    while (true) {
        vec3 p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0.0f) - vec3(1.0f, 1.0f, 0.0f);
        if (p.length_squared() < 1.0f)
            return p;
    }
}

__device__ vec3 random_unit_vector(curandState* local_rand_state)
{
    while (true) {
        vec3 p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state)) - vec3(1.0f, 1.0f, 1.0f);
        float lensq = p.length_squared();
        if (lensq > 1e-42f && lensq <= 1)
            return p / sqrt(lensq);
    }
}

__device__ vec3 random_on_hemisphere(const vec3& normal, curandState* local_rand_state)
{
    vec3 on_unit_sphere = random_unit_vector(local_rand_state);
    if (dot(on_unit_sphere, normal) > 0.0f) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

__device__ inline vec3 reflect(const vec3& v, const vec3& n)
{
    return v - 2 * dot(v, n) * n;
}

__device__ inline vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) 
{
    float cos_theta = fmin(dot(-uv, n), 1.0f);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}
