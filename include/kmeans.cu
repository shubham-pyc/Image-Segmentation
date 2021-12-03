#include <iostream>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "point.h"
#include <math.h>

using namespace std;
using DataFrame = vector<Point>;
int MAX_THREADS = 1024;

__device__ int squared_euclidean_distance(int first, int second)
{
    return (first - second) * (first - second);
}

__global__ void assign_clusters_to_points(const thrust::device_ptr<int> d_points,
                                          thrust::device_ptr<int> d_assignments,
                                          const thrust::device_ptr<int> means,
                                          thrust::device_ptr<int> sums,
                                          thrust::device_ptr<int> counts,
                                          int k,
                                          int size,
                                          bool copy_to_assign

)
{

    extern __shared__ int shared_means[];
    const int id = threadIdx.x;
    const int index = blockIdx.x * blockDim.x + id;

    if (index >= size)
        return;

    if (index == 0)
    {
        for (int i = 0; i < k; i++)
        {
            int something = sums[i];
            int something1 = counts[i];
            int mean = means[i];
            printf("Sums: %d, Counts:%d  means: %d\n", something, something1, mean);
        }
    }

    if (id < k)
    {
        shared_means[id] = means[id];
    }
    __syncthreads();

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
    if (copy_to_assign)
    {
        d_assignments[index] = best_cluster;
    }

    atomicAdd(thrust::raw_pointer_cast(sums + best_cluster), point);
    atomicAdd(thrust::raw_pointer_cast(counts + best_cluster), 1);

    // printf("This is kernel\n");
}

__global__ void calcualte_new_means(thrust::device_ptr<int> means,
                                    thrust::device_ptr<int> sums,
                                    thrust::device_ptr<int> counts)

{
    int index = threadIdx.x;
    const int count = max(counts[index], 1);
    means[index] = sums[index] / counts[index];
}

DataFrame k_means_cuda(const DataFrame &data, int *initial_means, size_t k,
                       size_t number_of_iterations, vector<size_t> &assign)
{
    int data_size = data.size();
    thrust::host_vector<int> h_points;
    thrust::host_vector<int> h_assignments(data.size(), 1);

    thrust::host_vector<int> h_means;
    DataFrame ret_value;
    thrust::device_vector<int> d_new_sums(k);
    thrust::device_vector<int> d_counts(k, 0);

    dim3 grid((data_size + MAX_THREADS - 1) / MAX_THREADS, 1, 1);

    // cout << "Checking: " << (data_size + MAX_THREADS - 1) / MAX_THREADS << endl;
    // cout << "Data size: " << data.size() << endl;
    dim3 block(MAX_THREADS, 1, 1);

    for (int i = 0; i < data.size(); i++)
    {
        h_points.push_back(data[i].x);
    }
    for (int i = 0; i < k; i++)
    {
        h_means.push_back(initial_means[i]);
    }

    thrust::device_vector<int> d_points = h_points;
    thrust::device_vector<int> d_means = h_means;
    thrust::device_vector<int> d_assignments(h_points.size(), 1);

    for (int i = 0; i < number_of_iterations; i++)
    {
        thrust::fill(d_new_sums.begin(), d_new_sums.end(), 0);
        thrust::fill(d_counts.begin(), d_counts.end(), 0);

        bool is_last = (i == number_of_iterations - 1) ? true : false;

        assign_clusters_to_points<<<grid, block>>>(
            d_points.data(),
            d_assignments.data(),
            d_means.data(),
            d_new_sums.data(),
            d_counts.data(),
            k,
            data_size,
            is_last);
        cudaDeviceSynchronize();

        calcualte_new_means<<<1, k>>>(
            d_means.data(),
            d_new_sums.data(),
            d_counts.data());
        cudaDeviceSynchronize();

        cout << "--------------------X--------------------" << endl;
    }

    h_assignments = d_assignments;
    h_means = d_means;

    std::vector<size_t> assignments(data.size());
    for (int i = 0; i < data.size(); i++)
    {
        assignments[i] = h_assignments[i];
        // cout << "Checking something awesome: " << h_assignments[i] << endl;
    }
    assign = assignments;

    for (int i = 0; i < k; i++)
    {
        Point point = {.x = h_means[i]};
        ret_value.push_back(point);
    }

    // cout  << "Chec"
    return ret_value;
}
