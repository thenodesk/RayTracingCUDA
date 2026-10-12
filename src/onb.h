#pragma once

// Orthonormal basis
class onb
{
public:
    __device__ onb(const vec3& n)
    {
        axis[2] = unit_vector(n);
        vec3 a = (fabs(axis[2].x()) > 0.9f) ? vec3(0.0f, 1.0f, 0.0f) : vec3(1.0f, 0.0f, 0.0f);
        axis[1] = unit_vector(cross(axis[2], a));
        axis[0] = cross(axis[2], axis[1]);
    }

    __device__ const vec3& u() const { return axis[0]; }
    __device__ const vec3& v() const { return axis[1]; }
    __device__ const vec3& w() const { return axis[2]; }

    __device__ vec3 transform(const vec3& v) const
    {
        // Transform from basis coordinates to local space.
        return (v[0] * axis[0]) + (v[1] * axis[1]) + (v[2] * axis[2]);
    }

private:
    vec3 axis[3];
};
