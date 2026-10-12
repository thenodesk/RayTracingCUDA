#include "rtweekend_utils.h"

#include "hittable.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"
#include "bvh.h"
#include "texture.h"
#include "scenes.h"

#include "external/stb_image_write.h"

#include <time.h>
#include <vector>

// Samples per kernel launch. Small launches avoids Windows driver timeout (TDR).
static const int SAMPLES_PER_LAUNCH = 20;

__global__ void render_init(int max_x, int max_y, curandState* rand_state)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i >= max_x) || (j >= max_y)) return;

	int pixel_index = j * max_x + i;

	curand_init(1996 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* fb, int max_x, int max_y, int sqrt_spp, float recip_sqrt_spp, int n_begin, int n_end, DeviceScene d_scene, curandState* rand_state)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= max_x || j >= max_y) return;

	int pixel_index = (j * max_x + i);

	curandState local_rand_state = rand_state[pixel_index];
	vec3 col(0.0f, 0.0f, 0.0f);
	for (int n = n_begin; n < n_end; n++)
	{
		int s_i = n % sqrt_spp;
		int s_j = n / sqrt_spp;

		float u = i + ((s_i + curand_uniform(&local_rand_state)) * recip_sqrt_spp - 0.5f);
		float v = j + ((s_j + curand_uniform(&local_rand_state)) * recip_sqrt_spp - 0.5f);

		ray r = d_scene.camera->get_ray(u, v, &local_rand_state);
		color c = d_scene.camera->ray_color(r, d_scene, &local_rand_state);

		if (isfinite(c.x()) && isfinite(c.y()) && isfinite(c.z()))
			col += c;
	}
	rand_state[pixel_index] = local_rand_state;

	fb[pixel_index] += col;
}

__global__ void finalize_image(vec3* fb, int max_x, int max_y, float recip_total_samples, DeviceScene d_scene)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= max_x || j >= max_y) return;

	int pixel_index = (j * max_x + i);

	vec3 col = fb[pixel_index] * recip_total_samples;

	// Apply linear to gamma
	col[0] = sqrt(col[0]);
	col[1] = sqrt(col[1]);
	col[2] = sqrt(col[2]);

	fb[pixel_index] = col;
}

void create_world_cpu(DeviceScene& d_scene, camera_props& cam_props)
{
	HostScene h_scene;

	create_scene1(h_scene, cam_props);

	d_scene.images_count = h_scene.images.size();
	if (d_scene.images_count > 0)
	{
		d_scene.images_addr = (unsigned char**)malloc(h_scene.images.size() * sizeof(unsigned char*));
		memcpy(d_scene.images_addr, h_scene.images.data(), h_scene.images.size() * sizeof(unsigned char*));
	}

	init_scene_gpu(d_scene, h_scene, cam_props);
}

void free_world(vec3* fb, curandState* d_rand_state, DeviceScene& d_scene)
{
	for (int i = 0; i < d_scene.images_count; i++)
		checkCudaErrors(cudaFree(d_scene.images_addr[i]));

	free(d_scene.images_addr);

	checkCudaErrors(cudaFree(d_scene.texture_types.solid));
	checkCudaErrors(cudaFree(d_scene.texture_types.checker));
	checkCudaErrors(cudaFree(d_scene.texture_types.image));
	checkCudaErrors(cudaFree(d_scene.texture_types.noise));
	checkCudaErrors(cudaFree(d_scene.world.lights));
	checkCudaErrors(cudaFree(d_scene.world.bvh_nodes));
	checkCudaErrors(cudaFree(d_scene.world.instances));
	checkCudaErrors(cudaFree(d_scene.world.objects));
	checkCudaErrors(cudaFree(d_scene.materials));
	checkCudaErrors(cudaFree(d_scene.textures));
	checkCudaErrors(cudaFree(d_scene.camera));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(fb));
}

int main()
{
	// World of hittables and camera
	DeviceScene d_scene{};
	camera_props cam_props;

	std::cout << "Building scene...\n";
	create_world_cpu(d_scene, cam_props);

	//---- Frame Buffer ----//
	int num_pixels = cam_props.img_width * cam_props.img_height;
	size_t fb_size = num_pixels * sizeof(vec3);

	// Allocate FB
	vec3* fb;
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));
	checkCudaErrors(cudaMemset(fb, 0, fb_size));

	// allocate random states
	curandState* d_rand_state;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));

	//---- Render ----//
	clock_t start, stop;
	start = clock();

	int tx = 16;
	int ty = 16;
	dim3 blocks((cam_props.img_width + tx - 1) / tx, (cam_props.img_height + ty - 1) / ty);
	dim3 threads(tx, ty);

	render_init<<<blocks, threads>>>(cam_props.img_width, cam_props.img_height, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	std::cout << "Rendering scene... (Dimensions: " << cam_props.img_width << "x" << cam_props.img_height 
		      << " | Samples per pixel: " << cam_props.samples_per_pixel << " | Max. ray bounces: " << cam_props.depth << ")\n";

	int sqrt_spp = int(std::sqrtf(cam_props.samples_per_pixel));
	int total_samples = sqrt_spp * sqrt_spp;
	float recip_sqrt_spp = 1.0f / sqrt_spp;
	float recip_total_samples = 1.0f / total_samples;
	for (int done = 0; done < total_samples; done += SAMPLES_PER_LAUNCH)
	{
		int n_end = std::min(done + SAMPLES_PER_LAUNCH, total_samples);

		render<<<blocks, threads>>>(fb, cam_props.img_width, cam_props.img_height, sqrt_spp, recip_sqrt_spp, done, n_end, d_scene, d_rand_state);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}

	finalize_image<<<blocks, threads>>>(fb, cam_props.img_width, cam_props.img_height, recip_total_samples, d_scene);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cout << "\nRender time: " << timer_seconds << " seconds.\n\n";

	// Output FB as Image
	unsigned char* host_fb = new unsigned char[num_pixels * cam_props.channels];

	for (int j = 0; j < cam_props.img_height; j++)
	{
		for (int i = 0; i < cam_props.img_width; i++)
		{
			write_color(host_fb, fb, i, j, cam_props.img_width, cam_props.channels);
		}
	}

	// CUDA clean up
	free_world(fb, d_rand_state, d_scene);

	//---- Save Image ----//
	stbi_write_png("image.png", cam_props.img_width, cam_props.img_height, cam_props.channels, host_fb, cam_props.img_width * cam_props.channels);

	delete[] host_fb;

	system("pause");

}