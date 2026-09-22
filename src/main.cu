#include "rtweekend_utils.h"

#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "material.h"
#include "camera.h"

#include "external/stb_image_write.h"

#include <time.h>

__global__ void rand_init(curandState* rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curand_init(1996, 0, 0, rand_state);
	}
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i >= max_x) || (j >= max_y)) return;

	int pixel_index = j * max_x + i;

	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1996, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* fb, int max_x, int max_y, int num_samples, camera** cam, hittable** world, curandState* rand_state)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= max_x || j >= max_y) return;

	int pixel_index = (j * max_x + i);

	curandState local_rand_state = rand_state[pixel_index];
	vec3 col(0.0f, 0.0f, 0.0f);
	for (int s = 0; s < num_samples; s++)
	{
		float u = float(i + (curand_uniform(&local_rand_state) - 0.5f));
		float v = float(j + (curand_uniform(&local_rand_state) - 0.5f));

		ray r = (*cam)->get_ray(u, v, &local_rand_state);
		col += (*cam)->ray_color(r, world, &local_rand_state);

	}
	rand_state[pixel_index] = local_rand_state;

	col /= float(num_samples);

	// Apply linear to gamma
	col[0] = sqrt(col[0]);
	col[1] = sqrt(col[1]);
	col[2] = sqrt(col[2]);

	fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hittable** d_list, hittable** d_world, camera** d_camera, camera_props cam_props, curandState* rand_state)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		curandState local_rand_state = *rand_state;
		int i = 0;

		d_list[i++] = new sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new lambertian(color(0.5f, 0.5f, 0.5f)));

		for (int a = -11; a < 11; a++)
		{
			for (int b = -11; b < 11; b++)
			{
				float choose_mat = RND;
				point3 center(a + 0.9f * RND, 0.2f, b + 0.9f * RND);

				if ((center - point3(4.0f, 0.2f, 0.0f)).length() > 0.9f)
				{
					if (choose_mat < 0.8f)
					{
						// diffuse
						color albedo = color::random(&local_rand_state) * color::random(&local_rand_state);
						d_list[i++] = new sphere(center, 0.2f, new lambertian(albedo));
					}
					else if (choose_mat < 0.95f)
					{
						// metal
						color albedo = color::random(0.5f, 1.0f, &local_rand_state);
						float fuzz = RND * 0.5f;
						d_list[i++] = new sphere(center, 0.2f, new metal(albedo, fuzz));
					}
					else
					{
						// glass
						d_list[i++] = new sphere(center, 0.2f, new dielectric(1.5f));
					}
				}
			}
		}

		d_list[i++] = new sphere(vec3(0.0f, 1.0f, 0.0f), 1.0f, new dielectric(1.5f));
		d_list[i++] = new sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f, new lambertian(color(0.4f, 0.2f, 0.1f)));
		d_list[i++] = new sphere(vec3(4.0f, 1.0f, 0.0f), 1.0f, new metal(color(0.7f, 0.6f, 0.5f), 0.0f));

		*rand_state = local_rand_state;

		*d_world = new hittable_list(d_list, i);

		(*d_camera) = new camera();
		(*d_camera)->props = cam_props;
		(*d_camera)->initialize();
	}
}

__global__ void free_world(hittable** d_list, hittable** d_world, camera** d_camera)
{
	int num_hittables = 22 * 22 + 1 + 3;
	for (int i = num_hittables - 1; i >= 0; i--)
	{
		delete d_list[i];
	}

	delete* d_world;
	delete* d_camera;
}

int main()
{
	//---- Image ----//
	float aspect_ratio = 16.0f / 9.0f;
	int img_width = 1280;
	int samples_per_pixel = 100;
	int depth = 10;
	int channels = 3;
	float vfov = 20.0f;
	
	int img_height = int(img_width / aspect_ratio);
	img_height = (img_height < 1) ? 1 : img_height;

	//---- Frame Buffer ----//
	int num_pixels = img_width * img_height;
	size_t fb_size = num_pixels * sizeof(vec3);

	// Allocate FB
	vec3* fb;
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

	// allocate random states
	curandState* d_rand_state;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
	curandState* d_rand_state2;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));

	rand_init<<<1, 1>>>(d_rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// World of hittables and camera
	hittable** d_list;
	int num_hittables = 22 * 22 + 1 + 3;
	checkCudaErrors(cudaMalloc((void**)&d_list, num_hittables * sizeof(hittable*)));
	hittable** d_world;
	checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));

	camera_props cam_props;
	cam_props.aspect_ratio = aspect_ratio;
	cam_props.img_width = img_width;
	cam_props.channels = channels;
	cam_props.depth = depth;
	cam_props.vfov = vfov;
	cam_props.lookfrom = point3(13, 2, 3);
	cam_props.lookat = point3(0, 0, 0);
	cam_props.vup = vec3(0, 1, 0);
	cam_props.defocus_angle = 0.6f;
	cam_props.focus_dist = 10.0f;

	camera** d_camera;
	checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));

	create_world<<<1, 1>>>(d_list, d_world, d_camera, cam_props, d_rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	//---- Render ----//
	clock_t start, stop;
	start = clock();

	int tx = 16;
	int ty = 16;
	dim3 blocks(img_width / tx + 1, img_height / ty + 1);
	dim3 threads(tx, ty);

	render_init<<<blocks, threads>>>(img_width, img_height, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	render<<<blocks, threads>>>(fb, img_width, img_height, samples_per_pixel, d_camera, d_world, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cout << "\nRender time: " << timer_seconds << " seconds.\n";

	// Output FB as Image
	unsigned char* host_fb = new unsigned char[num_pixels * channels];

	for (int j = 0; j < img_height; j++)
	{
		for (int i = 0; i < img_width; i++)
		{
			write_color(host_fb, fb, i, j, img_width, channels);
		}
	}

	// CUDA clean up
	free_world<<<1, 1>>>(d_list, d_world, d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(fb));

	//---- Save Image ----//
	stbi_write_png("image.png", img_width, img_height, channels, host_fb, img_width * channels);

	delete[] host_fb;

	system("pause");

}