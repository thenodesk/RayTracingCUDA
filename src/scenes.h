#pragma once

struct HostScene
{
	std::vector<TextureData> textures;
	std::vector<MaterialData> materials;
	std::vector<HittableObject> objects;
	std::vector<HittableInstance> instances;
	std::vector<unsigned char*> images;

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
	h_scene.objects.emplace_back(quad(point3(min.x(), min.y(), max.z()), dx, dy, true), material_idx); // front
	h_scene.objects.back().rotate_y(rotation_y);
	h_scene.objects.back().translate(translation);
	box_bbox = h_scene.objects.back().bounding_box();

	sides[1] = h_scene.objects.size();
	h_scene.objects.emplace_back(quad(point3(max.x(), min.y(), max.z()), -dz, dy, true), material_idx); // right
	h_scene.objects.back().rotate_y(rotation_y);
	h_scene.objects.back().translate(translation);
	box_bbox = aabb(box_bbox, h_scene.objects.back().bounding_box());

	sides[2] = h_scene.objects.size();
	h_scene.objects.emplace_back(quad(point3(max.x(), min.y(), min.z()), -dx, dy, true), material_idx); // back
	h_scene.objects.back().rotate_y(rotation_y);
	h_scene.objects.back().translate(translation);
	box_bbox = aabb(box_bbox, h_scene.objects.back().bounding_box());

	sides[3] = h_scene.objects.size();
	h_scene.objects.emplace_back(quad(point3(min.x(), min.y(), min.z()), dz, dy, true), material_idx); // left
	h_scene.objects.back().rotate_y(rotation_y);
	h_scene.objects.back().translate(translation);
	box_bbox = aabb(box_bbox, h_scene.objects.back().bounding_box());

	sides[4] = h_scene.objects.size();
	h_scene.objects.emplace_back(quad(point3(min.x(), max.y(), max.z()), dx, -dz, true), material_idx); // top
	h_scene.objects.back().rotate_y(rotation_y);
	h_scene.objects.back().translate(translation);
	box_bbox = aabb(box_bbox, h_scene.objects.back().bounding_box());

	sides[5] = h_scene.objects.size();
	h_scene.objects.emplace_back(quad(point3(min.x(), min.y(), min.z()), dx, dz, true), material_idx); // bottom
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

// Bouncing Spheres
void create_scene1(HostScene& h_scene, camera_props& cam_props)
{
	cam_props.aspect_ratio = 16.0f / 9.0f;
	cam_props.img_width = 1920;
	cam_props.samples_per_pixel = 200;
	cam_props.depth = 10;
	cam_props.background = color(0.70f, 0.80f, 1.00f);
	cam_props.vfov = 20.0f;
	cam_props.lookfrom = point3(13, 2, 3);
	cam_props.lookat = point3(0, 0, 0);
	cam_props.vup = vec3(0, 1, 0);
	cam_props.defocus_angle = 0.0f;

	h_scene.checker.emplace_back(checker_texture(0.32f, color(.2f, .3f, .1f), color(.9f, .9f, .9f)));
	h_scene.textures.emplace_back(TEX_CHECKER, h_scene.checker.size() - 1);
	h_scene.materials.emplace_back(lambertian(), h_scene.textures.size() - 1);
	h_scene.objects.emplace_back(sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f), h_scene.materials.size() - 1);

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

					h_scene.solid.emplace_back(solid_color(albedo));
					h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
					h_scene.materials.emplace_back(lambertian(), h_scene.textures.size() - 1);
					h_scene.objects.emplace_back(sphere(center, center2, 0.2f), h_scene.materials.size() - 1);
				}
				else if (choose_mat < 0.95f)
				{
					// metal
					color albedo = color::random(0.5f, 1.0f);
					float fuzz = random_float() * 0.5f;

					h_scene.solid.emplace_back(solid_color(albedo));
					h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
					h_scene.materials.emplace_back(metal(fuzz), h_scene.textures.size() - 1);
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

	h_scene.solid.emplace_back(solid_color(0.4f, 0.2f, 0.1f));
	h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
	h_scene.materials.emplace_back(lambertian(), h_scene.textures.size() - 1);
	h_scene.objects.emplace_back(sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f), h_scene.materials.size() - 1);

	h_scene.solid.emplace_back(solid_color(0.7f, 0.6f, 0.5f));
	h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
	h_scene.materials.emplace_back(metal(0.0f), h_scene.textures.size() - 1);
	h_scene.objects.emplace_back(sphere(vec3(4.0f, 1.0f, 0.0f), 1.0f), h_scene.materials.size() - 1);
}

// Checkered Spheres
void create_scene2(HostScene& h_scene, camera_props& cam_props)
{
	cam_props.aspect_ratio = 16.0f / 9.0f;
	cam_props.img_width = 1920;
	cam_props.samples_per_pixel = 200;
	cam_props.depth = 10;
	cam_props.background = color(0.70f, 0.80f, 1.00f);
	cam_props.vfov = 20.0f;
	cam_props.lookfrom = point3(13, 2, 3);
	cam_props.lookat = point3(0, 0, 0);
	cam_props.vup = vec3(0, 1, 0);
	cam_props.defocus_angle = 0.0f;

	h_scene.checker.emplace_back(checker_texture(0.32f, color(.2f, .3f, .1f), color(.9f, .9f, .9f)));
	h_scene.textures.emplace_back(TEX_CHECKER, h_scene.checker.size() - 1);
	h_scene.materials.emplace_back(lambertian(), h_scene.textures.size() - 1);
	h_scene.objects.emplace_back(sphere(vec3(0.0f, -10.0f, 0.0f), 10.0f), h_scene.materials.size() - 1);
	h_scene.objects.emplace_back(sphere(vec3(0.0f, 10.0f, 0.0f), 10.0f), h_scene.materials.size() - 1);
}

// Earth
void create_scene3(HostScene& h_scene, camera_props& cam_props)
{
	cam_props.aspect_ratio = 16.0f / 9.0f;
	cam_props.img_width = 1920;
	cam_props.samples_per_pixel = 200;
	cam_props.depth = 10;
	cam_props.background = color(0.70f, 0.80f, 1.00f);
	cam_props.vfov = 20.0f;
	cam_props.lookfrom = point3(0, 0, 12);
	cam_props.lookat = point3(0, 0, 0);
	cam_props.vup = vec3(0, 1, 0);
	cam_props.defocus_angle = 0.0f;

	unsigned char* img_data = nullptr;

	h_scene.image.emplace_back(image_texture("earthmap.jpg", img_data));
	h_scene.images.push_back(img_data);
	h_scene.textures.emplace_back(TEX_IMAGE, h_scene.image.size() - 1);
	h_scene.materials.emplace_back(lambertian(), h_scene.textures.size() - 1);
	h_scene.objects.emplace_back(sphere(vec3(0.0f, 0.0f, 0.0f), 2.0f), h_scene.materials.size() - 1);
}

// Perlin Spheres
void create_scene4(HostScene& h_scene, camera_props& cam_props)
{
	cam_props.aspect_ratio = 16.0f / 9.0f;
	cam_props.img_width = 1920;
	cam_props.samples_per_pixel = 200;
	cam_props.depth = 10;
	cam_props.background = color(0.70f, 0.80f, 1.00f);
	cam_props.vfov = 20.0f;
	cam_props.lookfrom = point3(13, 2, 3);
	cam_props.lookat = point3(0, 0, 0);
	cam_props.vup = vec3(0, 1, 0);
	cam_props.defocus_angle = 0.0f;

	h_scene.noise.emplace_back(noise_texture(4));
	h_scene.textures.emplace_back(TEX_NOISE, h_scene.noise.size() - 1);
	h_scene.materials.emplace_back(lambertian(), h_scene.textures.size() - 1);
	h_scene.objects.emplace_back(sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f), h_scene.materials.size() - 1);
	h_scene.objects.emplace_back(sphere(vec3(0.0f, 2.0f, 0.0f), 2.0f), h_scene.materials.size() - 1);
}

// Quads
void create_scene5(HostScene& h_scene, camera_props& cam_props)
{
	cam_props.aspect_ratio = 1.0f;
	cam_props.img_width = 1200;
	cam_props.samples_per_pixel = 200;
	cam_props.depth = 10;
	cam_props.background = color(0.70f, 0.80f, 1.00f);
	cam_props.vfov = 80.0f;
	cam_props.lookfrom = point3(0, 0, 9);
	cam_props.lookat = point3(0, 0, 0);
	cam_props.vup = vec3(0, 1, 0);
	cam_props.defocus_angle = 0.0f;

	h_scene.solid.emplace_back(solid_color(1.0f, 0.2f, 0.2f));
	h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
	h_scene.materials.emplace_back(lambertian(), h_scene.textures.size() - 1);
	h_scene.objects.emplace_back(quad(point3(-3, -2, 5), vec3(0, 0, -4), vec3(0, 4, 0)), h_scene.materials.size() - 1);

	h_scene.solid.emplace_back(solid_color(0.2f, 1.0f, 0.2f));
	h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
	h_scene.materials.emplace_back(lambertian(), h_scene.textures.size() - 1);
	h_scene.objects.emplace_back(quad(point3(-2, -2, 0), vec3(4, 0, 0), vec3(0, 4, 0)), h_scene.materials.size() - 1);

	h_scene.solid.emplace_back(solid_color(0.2f, 0.2f, 1.0f));
	h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
	h_scene.materials.emplace_back(lambertian(), h_scene.textures.size() - 1);
	h_scene.objects.emplace_back(quad(point3(3, -2, 1), vec3(0, 0, 4), vec3(0, 4, 0)), h_scene.materials.size() - 1);

	h_scene.solid.emplace_back(solid_color(1.0f, 0.5f, 0.0f));
	h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
	h_scene.materials.emplace_back(lambertian(), h_scene.textures.size() - 1);
	h_scene.objects.emplace_back(quad(point3(-2, 3, 1), vec3(4, 0, 0), vec3(0, 0, 4)), h_scene.materials.size() - 1);

	h_scene.solid.emplace_back(solid_color(0.2f, 0.8f, 0.8f));
	h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
	h_scene.materials.emplace_back(lambertian(), h_scene.textures.size() - 1);
	h_scene.objects.emplace_back(quad(point3(-2, -3, 5), vec3(4, 0, 0), vec3(0, 0, -4)), h_scene.materials.size() - 1);
}

// Simple Light
void create_scene6(HostScene& h_scene, camera_props& cam_props)
{
	cam_props.aspect_ratio = 16.0f / 9.0f;
	cam_props.img_width = 1920;
	cam_props.samples_per_pixel = 200;
	cam_props.depth = 10;
	cam_props.background = color(0, 0, 0);
	cam_props.vfov = 20.0f;
	cam_props.lookfrom = point3(26, 3, 6);
	cam_props.lookat = point3(0, 2, 0);
	cam_props.vup = vec3(0, 1, 0);
	cam_props.defocus_angle = 0.0f;

	h_scene.noise.emplace_back(noise_texture(4));
	h_scene.textures.emplace_back(TEX_NOISE, h_scene.noise.size() - 1);
	h_scene.materials.emplace_back(lambertian(), h_scene.textures.size() - 1);
	h_scene.objects.emplace_back(sphere(point3(0, -1000, 0), 1000), h_scene.materials.size() - 1);
	h_scene.objects.emplace_back(sphere(point3(0, 2, 0), 2), h_scene.materials.size() - 1);

	h_scene.solid.emplace_back(solid_color(4, 4, 4));
	h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
	h_scene.materials.emplace_back(diffuse_light(), h_scene.textures.size() - 1);
	h_scene.objects.emplace_back(sphere(point3(0, 7, 0), 2), h_scene.materials.size() - 1);
	h_scene.objects.emplace_back(quad(point3(3, 1, -2), vec3(2, 0, 0), vec3(0, 2, 0)), h_scene.materials.size() - 1);
}

// Cornell Box
void create_scene7(HostScene& h_scene, camera_props& cam_props)
{
	cam_props.aspect_ratio = 1.0f;
	cam_props.img_width = 1200;
	cam_props.samples_per_pixel = 200;
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

	h_scene.objects.emplace_back(quad(point3(555, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555)), green_mat);
	h_scene.objects.emplace_back(quad(point3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555)), red_mat);
	h_scene.objects.emplace_back(quad(point3(343, 554, 332), vec3(-130, 0, 0), vec3(0, 0, -105)), light_mat);
	h_scene.objects.emplace_back(quad(point3(0, 0, 0), vec3(555, 0, 0), vec3(0, 0, 555)), white_mat);
	h_scene.objects.emplace_back(quad(point3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555)), white_mat);
	h_scene.objects.emplace_back(quad(point3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0)), white_mat);

	add_quad_box(h_scene, point3(0, 0, 0), point3(165, 330, 165), white_mat, vec3(265, 0, 295), 15);
	add_quad_box(h_scene, point3(0, 0, 0), point3(165, 165, 165), white_mat, vec3(130, 0, 65), -18);
}

// Cornell Smoke
void create_scene8(HostScene& h_scene, camera_props& cam_props)
{
	cam_props.aspect_ratio = 1.0f;
	cam_props.img_width = 1200;
	cam_props.samples_per_pixel = 200;
	cam_props.depth = 10;
	cam_props.background = color(0, 0, 0);
	cam_props.vfov = 40.0f;
	cam_props.lookfrom = point3(278, 278, -800);
	cam_props.lookat = point3(278, 278, 0);
	cam_props.vup = vec3(0, 1, 0);
	cam_props.defocus_angle = 0.0f;

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

	h_scene.solid.emplace_back(solid_color(7, 7, 7));
	h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
	int light_mat = h_scene.materials.size();
	h_scene.materials.emplace_back(diffuse_light(), h_scene.textures.size() - 1);

	h_scene.solid.emplace_back(solid_color(1, 1, 1));
	h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
	int white_fog_mat = h_scene.materials.size();
	h_scene.materials.emplace_back(isotropic(), h_scene.textures.size() - 1);

	h_scene.solid.emplace_back(solid_color(0, 0, 0));
	h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
	int black_fog_mat = h_scene.materials.size();
	h_scene.materials.emplace_back(isotropic(), h_scene.textures.size() - 1);

	h_scene.objects.emplace_back(quad(point3(555, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555)), green_mat);
	h_scene.objects.emplace_back(quad(point3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555)), red_mat);
	h_scene.objects.emplace_back(quad(point3(113, 554, 127), vec3(330, 0, 0), vec3(0, 0, 305)), light_mat);
	h_scene.objects.emplace_back(quad(point3(0, 0, 0), vec3(555, 0, 0), vec3(0, 0, 555)), white_mat);
	h_scene.objects.emplace_back(quad(point3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555)), white_mat);
	h_scene.objects.emplace_back(quad(point3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0)), white_mat);

	add_quad_box(h_scene, point3(0, 0, 0), point3(165, 330, 165), black_fog_mat, vec3(265, 0, 295), 15, true, 0.01f);
	add_quad_box(h_scene, point3(0, 0, 0), point3(165, 165, 165), white_fog_mat, vec3(130, 0, 65), -18, true, 0.01f);
}

// Final Scene
void create_scene9(HostScene& h_scene, camera_props& cam_props)
{
	cam_props.aspect_ratio = 1.0f;
	cam_props.img_width = 1200;
	cam_props.samples_per_pixel = 1000;
	cam_props.depth = 20;
	cam_props.background = color(0, 0, 0);
	cam_props.vfov = 40.0f;
	cam_props.lookfrom = point3(478, 278, -600);
	cam_props.lookat = point3(278, 278, 0);
	cam_props.vup = vec3(0, 1, 0);
	cam_props.defocus_angle = 0.0f;

	h_scene.solid.emplace_back(solid_color(.48f, .83f, .53f));
	h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
	int ground_mat = h_scene.materials.size();
	h_scene.materials.emplace_back(lambertian(), h_scene.textures.size() - 1);

	int boxes_per_side = 20;
	for (int i = 0; i < boxes_per_side; i++) {
		for (int j = 0; j < boxes_per_side; j++)
		{
			float w = 100.0f;
			float x0 = -1000.0f + i * w;
			float z0 = -1000.0f + j * w;
			float y0 = 0.0f;
			float x1 = x0 + w;
			float y1 = random_float(1, 101);
			float z1 = z0 + w;

			add_quad_box(h_scene, point3(x0, y0, z0), point3(x1, y1, z1), ground_mat);
		}
	}

	h_scene.solid.emplace_back(solid_color(7, 7, 7));
	h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
	int light_mat = h_scene.materials.size();
	h_scene.materials.emplace_back(diffuse_light(), h_scene.textures.size() - 1);
	h_scene.objects.emplace_back(quad(point3(123, 554, 147), vec3(300, 0, 0), vec3(0, 0, 265)), light_mat);

	h_scene.solid.emplace_back(solid_color(0.7, 0.3, 0.1));
	h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
	int sphere_mat = h_scene.materials.size();
	h_scene.materials.emplace_back(lambertian(), h_scene.textures.size() - 1);
	point3 center1 = point3(400, 400, 200);
	vec3 center2 = center1 + vec3(30, 0, 0);
	h_scene.objects.emplace_back(sphere(center1, center2, 50), sphere_mat);

	h_scene.solid.emplace_back(solid_color(1, 1, 1));
	h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
	int glass_sphere_mat = h_scene.materials.size();
	h_scene.materials.emplace_back(dielectric(1.5f), h_scene.textures.size() - 1);
	h_scene.objects.emplace_back(sphere(point3(260, 150, 45), 50), glass_sphere_mat);

	h_scene.solid.emplace_back(solid_color(0.8, 0.8, 0.9));
	h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
	int metal_sphere_mat = h_scene.materials.size();
	h_scene.materials.emplace_back(metal(1.0f), h_scene.textures.size() - 1);
	h_scene.objects.emplace_back(sphere(point3(0, 150, 145), 50), metal_sphere_mat);

	h_scene.solid.emplace_back(solid_color(0.2, 0.4, 0.9));
	h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
	int volume_sphere_mat = h_scene.materials.size();
	h_scene.materials.emplace_back(isotropic(), h_scene.textures.size() - 1);
	h_scene.objects.emplace_back(sphere(point3(360, 150, 145), 70, true, 0.2f), volume_sphere_mat);
	h_scene.objects.emplace_back(sphere(point3(360, 150, 145), 70), glass_sphere_mat);

	h_scene.solid.emplace_back(solid_color(1, 1, 1));
	h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
	int volume_sphere2_mat = h_scene.materials.size();
	h_scene.materials.emplace_back(isotropic(), h_scene.textures.size() - 1);
	h_scene.objects.emplace_back(sphere(point3(0, 0, 0), 5000, true, 0.0001f), volume_sphere2_mat);
	h_scene.objects.emplace_back(sphere(point3(0, 0, 0), 5000), glass_sphere_mat);


	unsigned char* img_data = nullptr;
	h_scene.image.emplace_back(image_texture("earthmap.jpg", img_data));
	h_scene.images.push_back(img_data);
	h_scene.textures.emplace_back(TEX_IMAGE, h_scene.image.size() - 1);
	h_scene.materials.emplace_back(lambertian(), h_scene.textures.size() - 1);
	h_scene.objects.emplace_back(sphere(point3(400, 200, 400), 100), h_scene.materials.size() - 1);

	h_scene.noise.emplace_back(noise_texture(0.2f));
	h_scene.textures.emplace_back(TEX_NOISE, h_scene.noise.size() - 1);
	h_scene.materials.emplace_back(lambertian(), h_scene.textures.size() - 1);
	h_scene.objects.emplace_back(sphere(point3(220, 280, 300), 80), h_scene.materials.size() - 1);

	h_scene.solid.emplace_back(solid_color(.73f, .73f, .73f));
	h_scene.textures.emplace_back(TEX_SOLID, h_scene.solid.size() - 1);
	int white_mat = h_scene.materials.size();
	h_scene.materials.emplace_back(lambertian(), h_scene.textures.size() - 1);
	int ns = 1000;
	for (int j = 0; j < ns; j++)
	{
		h_scene.objects.emplace_back(sphere(point3::random(0, 165), 10), white_mat);
		h_scene.objects.back().rotate_y(15);
		h_scene.objects.back().translate(vec3(-100, 270, 395));
	}
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

	camera h_camera;
	h_camera.props = cam_props;
	h_camera.initialize();
	cam_props.img_height = h_camera.get_image_height();

	d_scene.world.objects = upload_to_gpu(h_scene.objects);
	d_scene.world.instances = upload_to_gpu(h_scene.instances);
	d_scene.world.bvh_nodes = upload_to_gpu(h_bvh_nodes);
	d_scene.materials = upload_to_gpu(h_scene.materials);
	d_scene.textures = upload_to_gpu(h_scene.textures);
	d_scene.texture_types.solid = upload_to_gpu(h_scene.solid);
	d_scene.texture_types.checker = upload_to_gpu(h_scene.checker);
	d_scene.texture_types.image = upload_to_gpu(h_scene.image);
	d_scene.texture_types.noise = upload_to_gpu(h_scene.noise);

	checkCudaErrors(cudaMalloc((void**)&d_scene.camera, sizeof(camera)));
	checkCudaErrors(cudaMemcpy(d_scene.camera, &h_camera, sizeof(camera), cudaMemcpyHostToDevice));
}
