#pragma once

#include <vector>

struct HostScene
{
	std::vector<MaterialData> materials;
	std::vector<HittableObject> objects;
};

//-------- SCENES --------//

void create_scene(HostScene& h_scene, camera_props& cam_props)
{
	cam_props.aspect_ratio = 16.0f / 9.0f;
	cam_props.img_width = 1920;
	cam_props.samples_per_pixel = 500;
	cam_props.depth = 20;
	cam_props.vfov = 20.0f;
	cam_props.lookfrom = point3(13, 2, 3);
	cam_props.lookat = point3(0, 0, 0);
	cam_props.vup = vec3(0, 1, 0);
	cam_props.defocus_angle = 0.6f;
	cam_props.focus_dist = 10.0f;

	h_scene.materials.emplace_back(lambertian(color(0.5, 0.5, 0.5)));
	h_scene.objects.emplace_back(sphere(point3(0, -1000, 0), 1000), h_scene.materials.size() - 1);

	int glass_mat_idx = h_scene.materials.size();
	h_scene.materials.emplace_back(dielectric(1.5f));

	for (int a = -11; a < 11; a++)
	{
		for (int b = -11; b < 11; b++)
		{
			float choose_mat = random_float();
			point3 center(a + 0.9f * random_float(), 0.2f, b + 0.9f * random_float());

			if ((center - point3(4.0f, 0.2f, 0.0f)).length() > 0.9f)
			{
				if (choose_mat < 0.8f)
				{
					// diffuse
					color albedo = color::random() * color::random();
					vec3 center2 = center + vec3(0.0f, random_float() * 0.5f, 0.0f);

					h_scene.materials.emplace_back(lambertian(albedo));
					h_scene.objects.emplace_back(sphere(center, 0.2f), h_scene.materials.size() - 1);
				}
				else if (choose_mat < 0.95f)
				{
					// metal
					color albedo = color::random(0.5f, 1.0f);
					float fuzz = random_float() * 0.5f;

					h_scene.materials.emplace_back(metal(albedo, fuzz));
					h_scene.objects.emplace_back(sphere(center, 0.2f), h_scene.materials.size() - 1);
				}
				else
				{
					// glass
					h_scene.objects.emplace_back(sphere(center, 0.2f), glass_mat_idx);
				}
			}
		}
	}

	h_scene.objects.emplace_back(sphere(vec3(0.0f, 1.0f, 0.0f), 1.0f), glass_mat_idx);

	h_scene.materials.emplace_back(lambertian(color(0.4f, 0.2f, 0.1f)));
	h_scene.objects.emplace_back(sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f), h_scene.materials.size() - 1);

	h_scene.materials.emplace_back(metal(color(0.7f, 0.6f, 0.5f), 0.0f));
	h_scene.objects.emplace_back(sphere(vec3(4.0f, 1.0f, 0.0f), 1.0f), h_scene.materials.size() - 1);
}

//-------- END SCENES --------//

template <class T>
T* upload_to_gpu(const std::vector<T>& h_asset)
{
	if (h_asset.empty()) return nullptr;

	T* d_asset;
	checkCudaErrors(cudaMalloc((void**)&d_asset, h_asset.size() * sizeof(T)));
	checkCudaErrors(cudaMemcpy(d_asset, h_asset.data(), h_asset.size() * sizeof(T), cudaMemcpyHostToDevice));

	return d_asset;
}

void init_scene_gpu(DeviceScene& d_scene, HostScene& h_scene, camera_props& cam_props)
{

	camera h_camera;
	h_camera.props = cam_props;
	h_camera.initialize();
	cam_props.img_height = h_camera.get_image_height();

	d_scene.world.count = h_scene.objects.size();

	d_scene.world.objects = upload_to_gpu(h_scene.objects);
	d_scene.materials = upload_to_gpu(h_scene.materials);

	checkCudaErrors(cudaMalloc((void**)&d_scene.camera, sizeof(camera)));
	checkCudaErrors(cudaMemcpy(d_scene.camera, &h_camera, sizeof(camera), cudaMemcpyHostToDevice));
}
