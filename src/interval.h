#pragma once

class interval
{
public:
    __host__ __device__ interval() : min(+INFINITY), max(-INFINITY) {} // Default interval is empty

    __host__ __device__ interval(float min, float max) : min(min), max(max) {}

    __host__ __device__ interval(const interval& a, const interval& b)
    {
        min = a.min <= b.min ? a.min : b.min;
        max = a.max >= b.max ? a.max : b.max;
    }

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

    __host__ __device__ interval expand(float delta) const
    {
        float padding = delta / 2.0f;
        return interval(min - padding, max + padding);
    }

    __host__ __device__ static const interval empty() {
        return interval(+INFINITY, -INFINITY);
    }

    __host__ __device__ static const interval universe() {
        return interval(-INFINITY, +INFINITY);
    }

    __host__ __device__ interval& operator+=(float displacement)
    {
        min += displacement;
        max += displacement;

        return *this;
    }

public:
    float min, max;
};

__host__ __device__ inline interval operator+(const interval& ival, float displacement)
{
    return interval(ival.min + displacement, ival.max + displacement);
}

__host__ __device__ inline interval operator+(float displacement, const interval& ival)
{
    return ival + displacement;
}
