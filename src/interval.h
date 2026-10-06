#pragma once

class interval
{
public:
    __host__ __device__ interval() : min(+INFINITY), max(-INFINITY) {} // Default interval is empty

    __host__ __device__ interval(float min, float max) : min(min), max(max) {}

    __host__ __device__ float size() const
    {
        return max - min;
    }

    __host__ __device__ bool contains(float x) const
    {
        return min <= x && x <= max;
    }

    __host__ __device__ bool surrounds(float x) const
    {
        return min < x && x < max;
    }

    __host__ __device__ float clamp(float x) const
    {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    __host__ __device__ static const interval empty() {
        return interval(+INFINITY, -INFINITY);
    }

    __host__ __device__ static const interval universe() {
        return interval(-INFINITY, +INFINITY);
    }

public:
    float min, max;
};

