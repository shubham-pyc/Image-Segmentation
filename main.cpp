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

int main(int argc, char *argv[])
{
    int is_mpi_program = false;
    int my_rank, total_processes;

    Image img;

    int k = 2;
    if (argc > 1)
    {
        is_mpi_program = true;
    }

    // srand(100);
    if (is_mpi_program)
    {
        int my_rank, total_processes;
        MPI_Init(NULL, NULL);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &total_processes);
        vector<Point> points;
        vector<size_t> assigments;
        vector<size_t> assigments_dist;

        int *means_;
        if (my_rank == 0)
        {
            img = imread();
            points = get_image_vector(img);
            means_ = subtractive_clustering(k, points);
        }
        vector<Point> test = k_means_distributed(points, means_, k, 5, assigments_dist);
        if (my_rank == 0)
        {
            vector<Point> test1 = k_means(points, means_, k, 5, assigments);
        }

        MPI_Finalize();
    }
    else
    {
        img = imread();
        vector<Point> points = get_image_vector(img);
        vector<size_t> assigments;
        int *means_ = subtractive_clustering(k, points);
        // int *means_ = get_initial_means(k, points);
        vector<Point> test1 = k_means(points, means_, k, 15, assigments);
        vector<Point> test = k_means_cuda(points, means_, k, 15, assigments);
        // vector<Point> test = k_means_shared(points, means_, k, 15, assigments);

        uint8_t *newIm = new uint8_t[img.height * img.width * img.channels];

        for (int i = 0; i < img.height * img.width * img.channels; i++)
        {
            newIm[i] = test[assigments[i]].x;
            // newIm[i] = img.image[i];
        }
        img.image = newIm;
        imwrite(img);
    }
}