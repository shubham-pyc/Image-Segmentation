#include <iostream>
#include <vector>
#include <random>
#include "./include/point.h"
#include "./include/helpers.h"
#include "./include/utils.h"
#include "./include/kmeans.h"
#include <chrono>
// #include "./include/image.h"

using namespace std;
using namespace std::chrono;

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
    int *means_;

    Image img;

    string imp_type = argv[1];

    vector<Point> points;
    vector<Point> final_means;

    vector<size_t> assigments;
    int k = 2;
    if (imp_type == "mpi")
    {
        is_mpi_program = true;
        MPI_Init(NULL, NULL);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &total_processes);
    }
    else
    {
        my_rank = 0;
    }

    if (my_rank == 0)
    {
        img = imread();
        points = get_image_vector(img);
        means_ = subtractive_clustering(k, points);
    }

    vector<Point> (*k_means_imp)(const DataFrame &, int *, size_t, size_t, vector<size_t> &);

    if (imp_type == "cuda")
    {

        k_means_imp = &k_means_cuda;
    }
    else if (imp_type == "omp")
    {

        k_means_imp = &k_means_shared;
    }
    else if (imp_type == "mpi")
    {
        k_means_imp = &k_means_distributed;
    }
    else
    {
        k_means_imp = &k_means;
    }

    final_means = k_means_imp(points, means_, k, 15, assigments);

    if (is_mpi_program)
        MPI_Finalize();

    if (my_rank == 0)
    {

        uint8_t *newIm = new uint8_t[img.height * img.width * img.channels];

        for (int i = 0; i < img.height * img.width * img.channels; i++)
        {
            newIm[i] = final_means[assigments[i]].x;
        }

        img.image = newIm;
        imwrite(img);
    }
}