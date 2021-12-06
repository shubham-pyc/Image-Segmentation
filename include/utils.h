#include <stdint.h>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image.h"
#include "../lib/stb_image_write.h"

// using namespace cv;
using namespace std;

#define CHANNEL_NUM 1
struct Image
{
	uint8_t *image;

	uint8_t *r_channel;
	uint8_t *g_channel;
	uint8_t *b_channel;

	int height;
	int width;
	int channels;
};

Image imread()
{
	int height, width;
	int bpp;
	uint8_t *rgb_image = stbi_load("./include/Lena.jpg", &width, &height, &bpp, CHANNEL_NUM);
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

int *convert_to_array(vector<Point> points, int size)
{
	int *point_array = new int[size]();
	for (int i = 0; i < size; i++)
	{
		point_array[i] = points[i].x;
	}

	return point_array;
}