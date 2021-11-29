#include <random>
#include <vector>

using frame = std::vector<Point>;

int square(int value)
{
    return value * value;
}

int squared_l2_distance(Point first, Point second)
{
    return square(first.x - second.x);
}

int get_random_number(float min, float max)
{
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = max - min;
    int r = random * diff;
    return min + r;
}


int *get_initial_means(int k, frame data)
{
    int *means = new int[k];
    for (int i = 0; i < k; i++)
    {
        means[i] = data[get_random_number(0, data.size())].x;
    }
    return means;
}
