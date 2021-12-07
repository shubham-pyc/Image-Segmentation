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
    string imp_type = "cuda";

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
        vector<Point> (*k_means_imp)(const DataFrame &, int *, size_t, size_t, vector<size_t> &);

        if (imp_type == "cuda")
        {

            k_means_imp = &k_means_cuda;
        }
        else if (imp_type == "omp")
        {

            k_means_imp = &k_means_shared;
        }
        else
        {
            k_means_imp = &k_means;
        }

        vector<Point> final_means = k_means_imp(points, means_, k, 15, assigments);

        uint8_t *newIm = new uint8_t[img.height * img.width * img.channels];

        for (int i = 0; i < img.height * img.width * img.channels; i++)
        {
            newIm[i] = final_means[assigments[i]].x;
            // newIm[i] = img.image[i];
        }
        img.image = newIm;
        imwrite(img);
    }
}