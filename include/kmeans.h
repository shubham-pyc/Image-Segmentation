#include <algorithm>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>
#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <iterator>

using namespace std;
using DataFrame = vector<Point>;

int THREAD_NUM = 4;

DataFrame k_means_cuda(const DataFrame &data, int *means_, size_t k,
					   size_t number_of_iterations, vector<size_t> &assign);

DataFrame k_means(const DataFrame &data, int *means_,
				  size_t k,
				  size_t number_of_iterations, vector<size_t> &assign)
{
	/*
	Serial implementation of k means algorithm
	Params: 
		data: vector of pixel data
		*means_ : initial array of means
		k: number of centroids
		number_of_iterations: k means number of iterations
		assign: vector of assigned clusters of each pixels 
	*/

	DataFrame means;
	for (int i = 0; i < k; i++)
	{
		Point p = {.x = means_[i]};
		means.push_back(p);
	}

	vector<size_t> assignments(data.size());

	// Iterations
	for (size_t iteration = 0; iteration < number_of_iterations; ++iteration)
	{
		// Find the cluster for each cluster
		for (size_t point = 0; point < data.size(); ++point)
		{
			double least_distance = numeric_limits<double>::max();
			size_t best_cluster = 0;
			for (size_t cluster = 0; cluster < k; ++cluster)
			{
				const int distance =
					squared_euclidean_distance(data[point], means[cluster]);
				if (distance < least_distance)
				{
					least_distance = distance;
					best_cluster = cluster;
				}
			}
			assignments[point] = best_cluster;
		}

		// Totalling all the clusters and counting
		DataFrame new_means(k);
		vector<size_t> counts(k, 0);
		for (size_t point = 0; point < data.size(); ++point)
		{
			const auto cluster = assignments[point];
			new_means[cluster].x += data[point].x;
			counts[cluster] += 1;
			// cout << cluster << endl;
		}

		// Getting new centroids
		for (size_t cluster = 0; cluster < k; ++cluster)
		{
			// Avoiding divide by zero
			const auto count = max<size_t>(1, counts[cluster]);
			means[cluster].x = new_means[cluster].x / count;
			// cout << "Iteration " << iteration << " cluster: " << cluster << " init_means: " << means[cluster].x << endl;
		}
	}
	assign = assignments;

	return means;
}

DataFrame k_means_shared(const DataFrame &data, int *means_, size_t k,
						 size_t number_of_iterations, vector<size_t> &assign)
{

	/*
	OMP multi threaded implementation of k means algorithm
	Params: 
		data: vector of pixel data
		*means_ : initial array of means
		k: number of centroids
		number_of_iterations: k means number of iterations
		assign: vector of assigned clusters of each pixels 
	*/

	DataFrame means;
	for (int i = 0; i < k; i++)
	{
		Point p = {.x = means_[i]};
		means.push_back(p);
	}

	vector<size_t> assignments(data.size());

	for (size_t iteration = 0; iteration < number_of_iterations; ++iteration)
	{

		DataFrame new_means(k);
		vector<size_t> counts(k, 0);

		vector<DataFrame> local_new_means;
		vector<vector<size_t>> local_counts;
#pragma omp parallel num_threads(THREAD_NUM)
		{

			const int nthreads = omp_get_num_threads();
			const int ithread = omp_get_thread_num();
#pragma omp single
			{

				for (int i = 0; i < nthreads; i++)
				{
					DataFrame local_mean(k);
					vector<size_t> local_count(k, 0);
					local_new_means.push_back(local_mean);
					local_counts.push_back(local_count);
				}
			}

// Find assignments.
#pragma omp for
			for (size_t point = 0; point < data.size(); ++point)
			{

				double least_distance = numeric_limits<double>::max();
				size_t best_cluster = 0;
				for (size_t cluster = 0; cluster < k; ++cluster)
				{
					const int distance =
						squared_euclidean_distance(data[point], means[cluster]);
					if (distance < least_distance)
					{
						least_distance = distance;
						best_cluster = cluster;
					}
				}
				assignments[point] = best_cluster;
			}

			// Sum up and count points for each cluster.
#pragma omp for
			for (size_t point = 0; point < data.size(); ++point)
			{

				const auto cluster = assignments[point];
				local_new_means[ithread][cluster].x += data[point].x;
				local_counts[ithread][cluster] += 1;
			}
#pragma omp single
			{
				for (int i = 0; i < nthreads; i++)
				{

					DataFrame local_new_mean = local_new_means[i];
					vector<size_t> local_count = local_counts[i];

					for (int j = 0; j < k; j++)
					{
						new_means[j].x += local_new_mean[j].x;
						counts[j] += local_count[j];
					}
				}
			}

// Divide sums by counts to get new centroids.
#pragma omp for
			for (size_t cluster = 0; cluster < k; ++cluster)
			{
				// Turn 0/0 into 0/1 to avoid zero division.
				const auto count = max<size_t>(1, counts[cluster]);
				means[cluster].x = new_means[cluster].x / count;
			}
		}
	}

	assign = assignments;

	return means;
}

DataFrame k_means_distributed(const DataFrame &data, int *means_, size_t k,
							  size_t number_of_iterations, vector<size_t> &assign)
{
	DataFrame return_value;
	int my_rank, total_processes;
	int iterations = number_of_iterations;
	int *init_means = new int[k]();

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &total_processes);
	int work_per_process = data.size() / total_processes;
	int *total_data;
	if (my_rank == 0)
	{
		total_data = convert_to_array(data, data.size());
		for (int i = 0; i < k; i++)
			init_means[i] = means_[i];
	}
	MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(init_means, k, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&work_per_process, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int data_size = work_per_process * total_processes;
	int *sub_data = new int[work_per_process];
	int *local_assignments = new int[work_per_process];
	int *assignments;

	if (my_rank == 0)
	{
		// cout << "Checking data size: " << data_size << " Chcking wpp: " << work_per_process << endl;
		assignments = new int[data_size];
	}

	MPI_Scatter(total_data, work_per_process, MPI_INT, sub_data, work_per_process, MPI_INT, 0, MPI_COMM_WORLD);

	for (int itr = 0; itr < iterations; itr++)
	{
		int *new_means = new int[k]();
		int *counts = new int[k]();
		int *gathered_means;
		int *gathered_counts;
		// Find the cluster for each cluster
		for (int point = 0; point < work_per_process; ++point)
		{
			double least_distance = numeric_limits<double>::max();
			int best_cluster = 0;
			for (int cluster = 0; cluster < k; ++cluster)
			{
				const int distance =
					squared_euclidean_distance_int(sub_data[point], init_means[cluster]);
				if (distance < least_distance)
				{
					least_distance = distance;
					best_cluster = cluster;
				}
			}
			local_assignments[point] = best_cluster;
		}

		if (my_rank == 0)
		{
			gathered_means = new int[k * total_processes]();
			gathered_counts = new int[k * total_processes]();
		}

		for (int point = 0; point < work_per_process; ++point)
		{
			const auto cluster = local_assignments[point];
			new_means[cluster] += sub_data[point];
			counts[cluster] += 1;
		}
		//Gather data

		MPI_Gather(new_means, k, MPI_INT, gathered_means, k, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Gather(counts, k, MPI_INT, gathered_counts, k, MPI_INT, 0, MPI_COMM_WORLD);

		if (my_rank == 0)
		{
			new_means = new int[k]();
			counts = new int[k]();

			for (int i = 0; i < k * total_processes; i++)
			{
				new_means[i % k] += gathered_means[i];
				counts[i % k] += gathered_counts[i];
			}

			for (int cluster = 0; cluster < k; ++cluster)
			{
				// Avoiding divide by zero
				const auto count = max<int>(1, counts[cluster]);
				init_means[cluster] = new_means[cluster] / count;
			}
		}
		MPI_Bcast(init_means, k, MPI_INT, 0, MPI_COMM_WORLD);

		if (itr == iterations - 1)
		{
			MPI_Gather(local_assignments, work_per_process, MPI_INT, assignments, work_per_process, MPI_INT, 0, MPI_COMM_WORLD);
		}
	}

	if (my_rank == 0)
	{
		for (int i = 0; i < k; i++)
		{
			Point p = {.x = init_means[i]};
			return_value.push_back(p);
		}
		for (int i = 0; i < data.size(); i++)
		{
			assign.push_back(assignments[i]);
		}
		
	}

	return return_value;
}
