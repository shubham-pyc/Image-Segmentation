#include <random>
#include <vector>
#include <math.h>
#include <bits/stdc++.h>
#include <algorithm>

using frame = std::vector<Point>;
using namespace std;

int square(int value)
{
    /*
        Function to square a number
        param: 
            int value
        return:
            int
    */
    return value * value;
}

int squared_euclidean_distance(Point first, Point second)
{

    /*
        Function to calculate the euclidean distance between 2 points
        Param:
            Point first second
        Returns:
            int
    */
    return square(first.x - second.x);
}

int squared_euclidean_distance_int(int first, int second)
{
    return square(first - second);
}

int get_random_number(float min, float max)
{
    /*
        Function to generate a random number
        Param:
            int min,max
        Returns:
            int
    */
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = max - min;
    int r = random * diff;
    return min + r;
}

int *get_initial_means(int k, frame data)
{
    /*
        Function to calculate initial means
    */
    int *means = new int[k];
    for (int i = 0; i < k; i++)
    {
        means[i] = data[get_random_number(0, data.size())].x;
    }
    return means;
}

double equation(int Xn, int Xi, float r)
{
    /*
        Implementation fo the equation given in paper for subtractive clustering
        Param:
            int, xi, xn, and radius
        Returns
            double
    */
    return exp(((-4 * Xn) - (Xi * Xi)) / (r * r));
}

// Defination in kmeans.cu file as this is computationaly expensive O(N^2)
vector<double> get_potentials(frame data);

int *subtractive_clustering(int k, frame data)
{
    /*
        Implementation of subtractive cluster algothm
        Param:
            k: num of means
            data: pixel data of image
    */
    int *means = new int[k];
    vector<double> potentials;
    const float hyper_penalty_radius = 0.1;
    potentials = get_potentials(data);

    int maxElementIndex = std::max_element(potentials.begin(), potentials.end()) - potentials.begin();
    double p1 = potentials[maxElementIndex];
    int x1 = data[maxElementIndex].x;

    means[0] = maxElementIndex;

    for (int Xn = 0; Xn < data.size(); Xn++)
    {
        potentials[Xn] = potentials[Xn] - p1 * equation(x1, data[Xn].x, hyper_penalty_radius);
    }
    potentials[maxElementIndex] = -999999;

    for (int itr = 0; itr < k - 1; itr++)
    {
        maxElementIndex = std::max_element(potentials.begin(), potentials.end()) - potentials.begin();
        means[itr + 1] = maxElementIndex;
        potentials[maxElementIndex] = -999999;
    }

    return means;
}