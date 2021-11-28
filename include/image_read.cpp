#include <stdint.h>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// using namespace cv;
using namespace std;

#define CHANNEL_NUM 3

int main()
{
	int width, height, bpp;

	uint8_t *rgb_image = stbi_load("image.jpg", &width, &height, &bpp, 3);

	// stbi_image_free(rgb_image);

	uint8_t *rgb_image_temp = new uint8_t[width * height * CHANNEL_NUM];

	for (int i = 0; i < width * height * CHANNEL_NUM; i++)
	{

		// cout << i << endl;
		rgb_image_temp[i] = rgb_image[i];
	}

	// cout << "Checking height: " << height << endl;
	// cout << "Checking width: " << width << endl;

	// cout << "Checking import";
	stbi_write_png("image_temp.jpg", width, height, CHANNEL_NUM, rgb_image_temp, width * CHANNEL_NUM);

	return 0;
}
