#pragma once

// Disable strict warnings for this header from the Microsoft Visual C++ compiler.
#ifdef _MSC_VER
	#pragma warning (push, 0)
#endif

#define STBI_FREE
#include "external/stb_image.h"

#include <cstdlib>
#include <iostream>


class rtw_image
{
public:
	rtw_image() {}

	rtw_image(const char* image_filename, unsigned char*& img_data)
	{
		// Loads image data from the specified file. If the RTW_IMAGES environment variable is
		// defined, looks only in that directory for the image file. If the image was not found,
		// searches for the specified image file first from the current directory, then in the
		// images/ subdirectory, then the _parent's_ images/ subdirectory, and then _that_
		// parent, on so on, for six levels up. If the image was not loaded successfully,
		// width() and height() will return 0.

		std::string filename = std::string(image_filename);
		auto imagedir = getenv("RTW_IMAGES");

		// Hunt for the image file in some likely locations.
		if (imagedir && load(std::string(imagedir) + "/" + image_filename, img_data)) return;
		if (load(filename, img_data)) return;
		if (load("images/" + filename, img_data)) return;
		if (load("../images/" + filename, img_data)) return;
		if (load("../../images/" + filename, img_data)) return;
		if (load("../../../images/" + filename, img_data)) return;
		if (load("../../../../images/" + filename, img_data)) return;
		if (load("../../../../../images/" + filename, img_data)) return;
		if (load("../../../../../../images/" + filename, img_data)) return;

		std::cerr << "ERROR: Could not load image file '" << image_filename << "'.\n";
	}

	bool load(const std::string& filename, unsigned char*& img_data)
	{
		// Loads the linear (gamma=1) image data from the given file name. Returns true if the
		// load succeeded. The resulting data buffer contains the three [0.0, 1.0]
		// floating-point values for the first pixel (red, then green, then blue). Pixels are
		// contiguous, going left to right for the width of the image, followed by the next row
		// below, for the full height of the image.

		int n = bytes_per_pixel; // Dummy out parameter: original components per pixel
		float* h_fdata = stbi_loadf(filename.c_str(), &image_width, &image_height, &n, bytes_per_pixel);
		if (h_fdata == nullptr) return false;

		bytes_per_scanline = image_width * bytes_per_pixel;

		unsigned char* h_bdata = convert_to_bytes(h_fdata);

		int total_bytes = image_width * image_height * bytes_per_pixel;
		checkCudaErrors(cudaMalloc((void**)&bdata, total_bytes));

		checkCudaErrors(cudaMemcpy(bdata, h_bdata, total_bytes, cudaMemcpyHostToDevice));

		img_data = bdata;

		stbi_image_free(h_fdata);
		delete[] h_bdata;

		return true;
	}

	void free_image()
	{
		checkCudaErrors(cudaFree(bdata));
	}

	__host__ __device__ int width() const { return (bdata == nullptr) ? 0 : image_width; }
	__host__ __device__ int height() const { return (bdata == nullptr) ? 0 : image_height; }

	__device__ const unsigned char* pixel_data(int x, int y) const
	{
		// Return the address of the three RGB bytes of the pixel at x,y. If there is no image
		// data, returns magenta.
		static unsigned char magenta[] = { 255, 0, 255 };
		if (bdata == nullptr) return magenta;

		x = clamp(x, 0, image_width);
		y = clamp(y, 0, image_height);

		return bdata + y * bytes_per_scanline + x * bytes_per_pixel;
	}

	unsigned char* data() const { return bdata; }

private:
	__host__ __device__ static int clamp(int x, int low, int high)
	{
		// Return the vlaue clamped to the range [low, high).
		if (x < low) return low;
		if (x < high) return x;
		return high - 1;
	}

	static unsigned char float_to_byte(float value)
	{
		if (value <= 0.0f)
			return 0;
		if (value >= 1.0f)
			return 255;
		return static_cast<unsigned char>(256.0f * value);
	}

	unsigned char* convert_to_bytes(float* fdata)
	{
		// Convert the linear floating point pixel data to bytes, storing the resulting byte
		// data in the `bdata` member.

		int total_bytes = image_width * image_height * bytes_per_pixel;
		unsigned char* bdata = new unsigned char[total_bytes];

		// Iterate through all pixel components, converting from [0.0, 1.0] float values to
		// unsigned [0, 255] byte values.

		auto* bptr = bdata;
		auto* fptr = fdata;
		for (int i = 0; i < total_bytes; i++, fptr++, bptr++)
			*bptr = float_to_byte(*fptr);

		return bdata;
	}

private:
	const int	   bytes_per_pixel = 3;
	//float*		   fdata = nullptr;			// Linear floating point pixel data
	unsigned char* bdata = nullptr;			// Linear 8-bit pixel data
	int			   image_width = 0;			// Loaded image width
	int			   image_height = 0;		// Loaded image height
	int			   bytes_per_scanline = 0;
};

#ifdef _MSC_VER
#pragma warning (pop)
#endif
