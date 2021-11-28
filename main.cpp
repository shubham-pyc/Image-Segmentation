#include "./include/utils.h"
#include <iostream>
#include <vector>
#include "./include/kmeans.h"
// #include "./include/image.h"

using namespace std;

int main()
{
    Image img = imread();
    vector<Point> points;
    vector<size_t> assigments;

    for (int i = 0; i < img.height * img.width * img.channels; i++)
    {
        Point p = {.x = img.image[i]};
        points.push_back(p);
    }

    vector<Point> test = k_means(points, 4, 10, assigments);
    uint8_t *newIm = new uint8_t[img.height * img.width * img.channels];

    for (int i = 0; i < img.height * img.width * img.channels; i++)
    {
        newIm[i] = test[assigments[i]].x;
    }
    img.image = newIm;
    imwrite(img);
}