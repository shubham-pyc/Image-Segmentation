#include <iostream>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "point.h"
#include <cuda_fp16.h>

using namespace std;
using DataFrame = vector<Point>;
int MAX_THREADS = 1024;

__constant__ int HYPER_CLUSTER_RADIUS = 0.5;

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
    /*
        Function to calculate the best clusters for the points
        Param: 
            d_points: pixel data
            d_assignments: point for assignments for the image pixel
            means: initial means
            sums: sum of the values from the same cluster
            coutns: coutns for the clusters
            k: number of clusters
            size: data size
            copy_to_assign: on the laster iteration pixels are assigned to 
                clusters and assigned to global memory
    */

    extern __shared__ int shared_means[];
    const int id = threadIdx.x;
    const int index = blockIdx.x * blockDim.x + id;

    if (index >= size)
        return;

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
    /*
        Cuda implementaion for kmeans clustering alogirhtm
        Param:
            data: pixel data
            inital_means: means
            k: number
            number_of_iterations: for how long
            assign: vector points for new assigned clusters
    */


    int data_size = data.size();
    thrust::host_vector<int> h_points;
    thrust::host_vector<int> h_assignments(data.size(), 1);

    thrust::host_vector<int> h_means;
    DataFrame ret_value;
    thrust::device_vector<int> d_new_sums(k);
    thrust::device_vector<int> d_counts(k, 0);


    dim3 grid((data_size + MAX_THREADS - 1) / MAX_THREADS, 1, 1);
    dim3 block(MAX_THREADS, 1, 1);


    //converting std::vector to thrust
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
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < number_of_iterations; i++)
    {
        thrust::fill(d_new_sums.begin(), d_new_sums.end(), 0);
        thrust::fill(d_counts.begin(), d_counts.end(), 0);

        bool is_last = (i == number_of_iterations - 1) ? true : false;

        assign_clusters_to_points<<<grid, block,k>>>(
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
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Checking cuda calculation time" << milliseconds << endl;

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

__device__ double equation_cuda(int Xn, int Xi, int r)
{
    return expf(((-4 * Xn) - (Xi * Xi)) / (r * r));
}

__global__ void calculate_potentials(thrust::device_ptr<double> data, thrust::device_ptr<double> output, int size)
{
    const int id = threadIdx.x;
    const int index = blockIdx.x * blockDim.x + id;

    if (index >= size)
        return;

    const int Xn = data[id];

    double potential = 0;
    for (int Xi = 0; Xi < size; Xi++)
    {
        potential += equation_cuda(Xn, data[Xi], HYPER_CLUSTER_RADIUS);
    }

    output[index] = potential;
}

vector<double> get_potentials(DataFrame data)
{
    /*
        Implementaions of equation fro getting the potential for a point for being a cluster
        Param: 
            data: pixel data
    */
    thrust::host_vector<double> h_potent(data.size());
    thrust::host_vector<double> h_data(data.size());

    thrust::device_vector<double> d_potent(data.size());
    thrust::device_vector<double> d_data(data.size());

    vector<double> potentials(data.size(), 0);
    int data_size = data.size();

    for (int i = 0; i < data_size; i++)
    {
        h_data[i] = data[i].x;
    }
    d_data = h_data;

    dim3 grid((data_size + MAX_THREADS - 1) / MAX_THREADS, 1, 1);
    dim3 block(MAX_THREADS, 1, 1);
    calculate_potentials<<<grid, block>>>(d_data.data(), d_potent.data(), data_size);
    cudaDeviceSynchronize();

    h_potent = d_potent;

    for (int i = 0; i < data_size; i++)
    {
        potentials[i] = h_potent[i];
    }

    return potentials;
}
