#pragma once

#include "rtweekend_utils.h"

#include "perlin.h"
#include "rtw_image.h"
#include "assets.h"

class solid_color
{
public:
	__host__ __device__ solid_color(const color& albedo) : albedo(albedo) {}
	__host__ __device__ solid_color(float red, float green, float blue) : solid_color(color(red, green, blue)) {}

	__device__ color value(float u, float v, const point3& p) const
	{
		return albedo;
	}

private:
	color albedo;
};

class checker_texture
{
public:
	__host__ __device__ checker_texture(float scale, const solid_color& even, const solid_color& odd)
		: inv_scale(1.0f / scale), even(even), odd(odd) {}

	__host__ __device__ checker_texture(float scale, const color& c1, const color& c2)
		: checker_texture(scale, solid_color(c1), solid_color(c2)) {}

	__device__ color value(float u, float v, const point3& p) const
	{
		int xInteger = int(floor(inv_scale * p.x()));
		int yInteger = int(floor(inv_scale * p.y()));
		int zInteger = int(floor(inv_scale * p.z()));

		bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;

		return isEven ? even.value(u, v, p) : odd.value(u, v, p);
	}

private:
	float inv_scale;
	solid_color even;
	solid_color odd;
};

class image_texture
{
public:
	__host__ image_texture(const char* filename, unsigned char*& img_data) : image(filename, img_data) {}

	__device__ color value(float u, float v, const point3& p) const
	{
		// If we have no texture data, then return solid cyan as a debugging aid.
		if (image.height() <= 0) return color(0.0f, 1.0f, 1.0f);

		// Clamp input texture coordinates to [0,1] x [1,0]
		u = interval(0.0f, 1.0f).clamp(u);
		v = 1.0f - interval(0.0f, 1.0f).clamp(v);  // Flip V to image coordinates

		int i = int(u * image.width());
		int j = int(v * image.height());
		const unsigned char* pixel = image.pixel_data(i, j);

		float color_scale = 1.0f / 255.0f;
		return color(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
	}

	void free_image()
	{
		image.free_image();
	}

private:
	rtw_image image;
};

class noise_texture
{
public:
	__host__ noise_texture(float scale) : scale(scale) {}

	__device__ color value(float u, float v, const point3& p) const
	{
		return color(.5f, .5f, .5f) * (1.0f + sin(scale * p.z() + 10 * noise.turb(p, 7)));
	}

private:
	perlin noise;
	float scale;
};

enum TextureType { TEX_SOLID, TEX_CHECKER, TEX_IMAGE, TEX_NOISE };

struct TextureData {
	TextureType type;

	int texture_type_id;

	TextureData(TextureType type, int tex_type_id) : type(type), texture_type_id(tex_type_id) {}

	~TextureData() {}

	__device__ color value(float u, float v, const point3& p, DeviceScene* d_scene) const
	{
		switch (type) {
		case TEX_SOLID:
			return d_scene->texture_types.solid[texture_type_id].value(u, v, p);
		case TEX_CHECKER:
			return d_scene->texture_types.checker[texture_type_id].value(u, v, p);
		case TEX_IMAGE:
			return d_scene->texture_types.image[texture_type_id].value(u, v, p);
		case TEX_NOISE:
			return d_scene->texture_types.noise[texture_type_id].value(u, v, p);
		}
		return color(0.0f, 0.0f, 0.0f);
	}
};
