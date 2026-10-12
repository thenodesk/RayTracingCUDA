#pragma once

struct HostScene
{
	std::vector<TextureData> textures;
	std::vector<MaterialData> materials;
	std::vector<HittableObject> objects;
	std::vector<HittableInstance> instances;
	std::vector<unsigned char*> images;
	std::vector<int> lights;

	std::vector<solid_color> solid;
	std::vector<checker_texture> checker;
	std::vector<image_texture> image;
	std::vector<noise_texture> noise;
};

void add_quad_box(HostScene& h_scene, const point3& a, const point3& b, int material_idx,
	const vec3& translation = vec3(0,0,0), float rotation_y = 0.0f, bool constant_medium = false, float density = 0.0f)
{
	point3 min = point3(std::fmin(a.x(), b.x()), std::fmin(a.y(), b.y()), std::fmin(a.z(), b.z()));
	point3 max = point3(std::fmax(a.x(), b.x()), std::fmax(a.y(), b.y()), std::fmax(a.z(), b.z()));

	vec3 dx = vec3(max.x() - min.x(), 0, 0);
	vec3 dy = vec3(0, max.y() - min.y(), 0);
	vec3 dz = vec3(0, 0, max.z() - min.z());

	int sides[6];
	aabb box_bbox;

	sides[0] = h_scene.objects.size();
	h_scene.objects.emplace_back(quad(point3(min.x(), min.y(), max.z()), dx, dy, false, true), material_idx); // front
	h_scene.objects.back().rotate_y(rotation_y);
	h_scene.objects.back().translate(translation);
	box_bbox = h_scene.objects.back().bounding_box();

	sides[1] = h_scene.objects.size();
	h_scene.objects.emplace_back(quad(point3(max.x(), min.y(), max.z()), -dz, dy, false, true), material_idx); // right
	h_scene.objects.back().rotate_y(rotation_y);
	h_scene.objects.back().translate(translation);
	box_bbox = aabb(box_bbox, h_scene.objects.back().bounding_box());

	sides[2] = h_scene.objects.size();
	h_scene.objects.emplace_back(quad(point3(max.x(), min.y(), min.z()), -dx, dy, false, true), material_idx); // back
	h_scene.objects.back().rotate_y(rotation_y);
	h_scene.objects.back().translate(translation);
	box_bbox = aabb(box_bbox, h_scene.objects.back().bounding_box());

	sides[3] = h_scene.objects.size();
	h_scene.objects.emplace_back(quad(point3(min.x(), min.y(), min.z()), dz, dy, false, true), material_idx); // left
	h_scene.objects.back().rotate_y(rotation_y);
	h_scene.objects.back().translate(translation);
	box_bbox = aabb(box_bbox, h_scene.objects.back().bounding_box());

	sides[4] = h_scene.objects.size();
	h_scene.objects.emplace_back(quad(point3(min.x(), max.y(), max.z()), dx, -dz, false, true), material_idx); // top
	h_scene.objects.back().rotate_y(rotation_y);
	h_scene.objects.back().translate(translation);
	box_bbox = aabb(box_bbox, h_scene.objects.back().bounding_box());

	sides[5] = h_scene.objects.size();
	h_scene.objects.emplace_back(quad(point3(min.x(), min.y(), min.z()), dx, dz, false, true), material_idx); // bottom
	h_scene.objects.back().rotate_y(rotation_y);
	h_scene.objects.back().translate(translation);
	box_bbox = aabb(box_bbox, h_scene.objects.back().bounding_box());

	box new_box(sides, constant_medium, density);
	new_box.set_bounding_box(box_bbox);
	h_scene.instances.emplace_back(new_box, material_idx);
}

std::vector<Primitive> build_primitives(const HostScene& h_scene)
{
	std::vector<Primitive> prims;
	prims.reserve(h_scene.objects.size() + h_scene.instances.size());

	for (size_t i = 0; i < h_scene.objects.size(); i++)
	{
		const HittableObject& obj = h_scene.objects[i];
		if (obj.is_child()) continue;

		PrimType t = (obj.type == HIT_SPHERE) ? PRIM_SPHERE : PRIM_QUAD;
		prims.push_back(Primitive{ t, int(i), obj.bounding_box() });
	}

	for (size_t i = 0; i < h_scene.instances.size(); i++)
		prims.push_back(Primitive{ PRIM_BOX, int(i), h_scene.instances[i].bounding_box() });

	return prims;
}

//-------- SCENES --------//

// Cornell Box
void create_scene1(HostScene& h_scene, camera_props& cam_props)
{
	cam_props.aspect_ratio = 1.0f;
	cam_props.img_width = 1200;
	cam_props.samples_per_pixel = 1000;
	cam_props.depth = 10;
	cam_props.background = color(0, 0, 0);
	cam_props.vfov = 40.0f;
	cam_props.lookfrom = point3(278, 278, -800);
	cam_props.lookat = point3(278, 278, 0);
	cam_props.vup = vec3(0, 1, 0);
	cam_props.defocus_angle = 0.0f;
	//cam_props.focus_dist = 10.0f;

	h_scene.solid.emplace_back(solid_color(.65f, .05f, .05f));
	h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
	int red_mat = h_scene.materials.size();
	h_scene.materials.emplace_back(lambertian(), h_scene.textures.size() - 1);

	h_scene.solid.emplace_back(solid_color(.73f, .73f, .73f));
	h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
	int white_mat = h_scene.materials.size();
	h_scene.materials.emplace_back(lambertian(), h_scene.textures.size() - 1);

	h_scene.solid.emplace_back(solid_color(.12f, .45f, .15f));
	h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
	int green_mat = h_scene.materials.size();
	h_scene.materials.emplace_back(lambertian(), h_scene.textures.size() - 1);

	h_scene.solid.emplace_back(solid_color(15, 15, 15));
	h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
	int light_mat = h_scene.materials.size();
	h_scene.materials.emplace_back(diffuse_light(), h_scene.textures.size() - 1);

	int glass_mat = h_scene.materials.size();
	h_scene.materials.emplace_back(dielectric(1.5f));

	h_scene.objects.emplace_back(quad(point3(555, 0, 0), vec3(0, 0, 555), vec3(0, 555, 0)), green_mat);
	h_scene.objects.emplace_back(quad(point3(0, 0, 555), vec3(0, 0, -555), vec3(0, 555, 0)), red_mat);
	h_scene.objects.emplace_back(quad(point3(0, 555, 0), vec3(555, 0, 0), vec3(0, 0, 555)), white_mat);
	h_scene.objects.emplace_back(quad(point3(0, 0, 555), vec3(555, 0, 0), vec3(0, 0, -555)), white_mat);
	h_scene.objects.emplace_back(quad(point3(555, 0, 555), vec3(-555, 0, 0), vec3(0, 555, 0)), white_mat);

	h_scene.objects.emplace_back(quad(point3(343, 554, 332), vec3(-130, 0, 0), vec3(0, 0, -105)), light_mat);
	h_scene.lights.emplace_back(h_scene.objects.size() - 1);

	add_quad_box(h_scene, point3(0, 0, 0), point3(165, 330, 165), white_mat, vec3(265, 0, 295), 15);

	h_scene.objects.emplace_back(sphere(point3(190, 90, 190), 90), glass_mat);
	h_scene.lights.emplace_back(h_scene.objects.size() - 1);
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
	std::vector<Primitive> primitives = build_primitives(h_scene);

	std::vector<BVHNode> h_bvh_nodes;
	build_bvh_recursive(primitives, 0, primitives.size(), h_bvh_nodes);

	d_scene.world.bvh_count = int(h_bvh_nodes.size());
	d_scene.world.lights_count = h_scene.lights.size();

	camera h_camera;
	h_camera.props = cam_props;
	h_camera.initialize();
	cam_props.img_height = h_camera.get_image_height();

	d_scene.world.objects = upload_to_gpu(h_scene.objects);
	d_scene.world.instances = upload_to_gpu(h_scene.instances);
	d_scene.world.bvh_nodes = upload_to_gpu(h_bvh_nodes);
	d_scene.world.lights = upload_to_gpu(h_scene.lights);
	d_scene.materials = upload_to_gpu(h_scene.materials);
	d_scene.textures = upload_to_gpu(h_scene.textures);
	d_scene.texture_types.solid = upload_to_gpu(h_scene.solid);
	d_scene.texture_types.checker = upload_to_gpu(h_scene.checker);
	d_scene.texture_types.image = upload_to_gpu(h_scene.image);
	d_scene.texture_types.noise = upload_to_gpu(h_scene.noise);

	checkCudaErrors(cudaMalloc((void**)&d_scene.camera, sizeof(camera)));
	checkCudaErrors(cudaMemcpy(d_scene.camera, &h_camera, sizeof(camera), cudaMemcpyHostToDevice));
}
