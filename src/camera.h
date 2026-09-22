#pragma once

#include "hittable.h"
#include "material.h"

struct camera_props
{
    float aspect_ratio = 16.0f / 9.0f;
    int img_width = 1280;
    int channels = 3;
    int depth = 10;
    float vfov = 60.0f;

    point3 lookfrom = point3(0.0f, 0.0f, 0.0f);
    point3 lookat   = point3(0.0f, 0.0f, -1.0f);
    vec3   vup      = vec3(0.0f, 1.0f, 0.0f);

    float defocus_angle = 0.0f; // Variation angle of rays through each pixel
    float focus_dist = 1.0f;    // Distance from camera lookfrom point to plane of perfect focus
};

class camera
{
public:
    __device__ void initialize()
    {
        img_height = int(props.img_width / props.aspect_ratio);
        img_height = (img_height < 1) ? 1 : img_height;

        center = props.lookfrom;

        // Determine viewport dimensions.
        float theta = degrees_to_radians(props.vfov);
        float h = tan(theta / 2.0f);
        float viewport_height = 2.0f * h * props.focus_dist;
        float viewport_width = viewport_height * (float(props.img_width) / img_height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(props.lookfrom - props.lookat);
        u = unit_vector(cross(props.vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        vec3 viewport_u = viewport_width * u;
        vec3 viewport_v = viewport_height * -v;

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / props.img_width;
        pixel_delta_v = viewport_v / img_height;

        // Calculate the location of the upper left pixel.
        vec3 viewport_upper_left = center - (props.focus_dist * w) - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

        // Calculate the camera defocus disk basis vectors.
        float defocus_radius = props.focus_dist * tan(degrees_to_radians(props.defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

    __device__ ray get_ray(float u, float v, curandState* local_rand_state)
    {
        vec3 pixel_sample = pixel00_loc + (u * pixel_delta_u) + (v * pixel_delta_v);
        
        point3 ray_origin = (props.defocus_angle <= 0) ? center : defocus_disk_sample(local_rand_state);
        vec3 ray_direction = pixel_sample - ray_origin;

        return ray(ray_origin, ray_direction);
    }

    __device__ point3 defocus_disk_sample(curandState* local_rand_state) const {
        // Returns a random point in the camera defocus disk.
        vec3 p = random_in_unit_disk(local_rand_state);
        return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
    }

    __device__ color ray_color(const ray& r, hittable** world, curandState* local_rand_state) const
    {
        ray cur_ray = r;
        color cur_attenuation = color(1.0f, 1.0f, 1.0f);

        ray scattered;
        color attenuation;

        for (int i = 0; i < props.depth; i++)
        {
            hit_record rec;
            if ((*world)->hit(cur_ray, interval(0.001f, INFINITY), rec))
            {
                if (rec.mat->scatter(cur_ray, rec, attenuation, scattered, local_rand_state))
                {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else
                {
                    return color(0.0f, 0.0f, 0.0f);
                }
            }
            else
            {
                vec3 unit_direction = unit_vector(cur_ray.direction());
                float t = 0.5f * (unit_direction.y() + 1.0f);
                color c = (1.0f - t) * color(1.0f, 1.0f, 1.0f) + t * color(0.5f, 0.7f, 1.0f);
                return cur_attenuation * c;
            }

        }

        return color(0.0f, 0.0f, 0.0f);
    }

public:
    camera_props props;

private:
    int    img_height;     // Rendered image height
    point3 center;         // Camera center
    point3 pixel00_loc;    // Location of pixel 0, 0
    vec3   pixel_delta_u;  // Offset to pixel to the right
    vec3   pixel_delta_v;  // Offset to pixel below
    vec3   v, u, w;        // Camera frame basis vectors
    vec3   defocus_disk_u; // Defocus disk horizontal radius
    vec3   defocus_disk_v; // Defocus disk vertical radius
};