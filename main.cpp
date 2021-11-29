#include <iostream>
#include <vector>
#include <random>
#include "./include/point.h"
#include "./include/helpers.h"
#include "./include/utils.h"
#include "./include/kmeans.h"
// #include "./include/image.h"

using namespace std;
vector<Point> get_image_vector(Image img)
{
    vector<Point> points;
    for (int i = 0; i < img.height * img.width * img.channels; i++)
    {
        Point p = {.x = img.image[i]};
        points.push_back(p);
    }
    return points;
}

int main()
{

    srand(100);

    Image img = imread();
    vector<Point> points = get_image_vector(img);
    vector<size_t> assigments;
    int k = 3;

    int *means_ = get_initial_means(k, points);

    // vector<Point> test = k_means(points, means_, k, 15, assigments);
    vector<Point> test = k_means_shared(points, means_, k, 15, assigments);
    uint8_t *newIm = new uint8_t[img.height * img.width * img.channels];

    for (int i = 0; i < img.height * img.width * img.channels; i++)
    {
        newIm[i] = test[assigments[i]].x;
    }
    img.image = newIm;
    imwrite(img);
}