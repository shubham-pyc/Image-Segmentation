#include <stdint.h>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// using namespace cv;
using namespace std;

#define CHANNEL_NUM 3
struct Image
{
	uint8_t *image;
	int height;
	int width;
	int channels;
};

Image imread()
{
	int height, width;
	int bpp;
	uint8_t *rgb_image = stbi_load("./include/image.jpg", &width, &height, &bpp, 3);
	// stbi_image_free(rgb_image);
	Image img = {.image = rgb_image, .height = height, .width = width, .channels = CHANNEL_NUM};
	// cout << "Checking" << img.width;
	return img;
}

void imwrite(Image img)
{
	int width = img.width;
	int height = img.height;
	int channels = img.channels;

	uint8_t *rgb_image;
	rgb_image = img.image;

	// Write your code to populate rgb_image here

	stbi_write_png("image.png", width, height, CHANNEL_NUM, rgb_image, width * CHANNEL_NUM);

	return;
}