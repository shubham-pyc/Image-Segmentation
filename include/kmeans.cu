#include <iostream>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "point.h"

using namespace std;
using DataFrame = vector<Point>;

__device__ int squared_euclidean_distance(int first, int second)
{
    return first * first - second * second;
}

__global__ void assign_clusters_to_points(const thrust::device_ptr<int> d_points,
                                          const thrust::device_ptr<int> means,
                                          thrust::device_ptr<int> sums,
                                          thrust::device_ptr<int> counts,
                                          int k,
                                          int size

)
{

    __shared__ int shared_means[k];
    const int id = threadIdx.x;
    const int index = blockIdx.x * blockDim.x + id;
    if (id < k)
        shared_means[id] = means[id];

    __syncthreads();

    if (index >= size)
        return;

    int point = d_points[index];
    int least_distance = INT_MAX;
    int best_cluster = 0;

    for (int cluster = 0; cluster < k; cluster++)
    {
        int centroid = shared_means[cluster];
        int dist = squared_euclidean_distance(point, centroid);
        if (dist < least_distance)
        {
            least_distance = dist;
            best_cluster = cluster;
        }
    }

    printf("Checking the size of data %d and second elements %d\n", size, x);
    // printf("This is kernel\n");
}

DataFrame k_means_cuda(const DataFrame &data, int *initial_means, size_t k,
                       size_t number_of_iterations, vector<size_t> &assign)
{
    int data_size = data.size();
    thrust::host_vector<int> h_points;

    thrust::host_vector<int> h_means;
    thrust::device_vector<int> d_new_sums;
    thrust::device_vector<int> d_counts;

    for (int i = 0; i < data.size(); i++)
    {
        h_points.push_back(data[i].x);
    }
    for (int i = 0; i < k; i++)
    {
        h_means.push_back(initial_means[k]);
    }

    thrust::device_vector<int> d_points = h_points;
    thrust::device_vector<int> d_means = h_means;

    cout << "Checking second elements: " << h_points[2] << endl;

    assign_clusters_to_points<<<1, 1>>>(
        d_points.data(),
        d_means.data(),
        d_new_sums.data(),
        d_counts.data(),
        k,
        data_size);

    cudaDeviceSynchronize();
    // cout  << "Chec"
    return data;
}
