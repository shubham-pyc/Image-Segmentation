#include <algorithm>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>
#include <iostream>
// #include <omp.h>

using namespace std;
using DataFrame = vector<Point>;

int THREAD_NUM = 4;

DataFrame k_means_cuda(const DataFrame &data, int *means_, size_t k,
					   size_t number_of_iterations, vector<size_t> &assign);

DataFrame k_means(const DataFrame &data, int *means_,
				  size_t k,
				  size_t number_of_iterations, vector<size_t> &assign)
{
	DataFrame means;
	for (int i = 0; i < k; i++)
	{
		Point p = {.x = means_[i]};
		means.push_back(p);
	}

	vector<size_t> assignments(data.size());
	for (size_t iteration = 0; iteration < number_of_iterations; ++iteration)
	{
		// Find assignments.
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
		DataFrame new_means(k);
		vector<size_t> counts(k, 0);
		for (size_t point = 0; point < data.size(); ++point)
		{
			const auto cluster = assignments[point];
			new_means[cluster].x += data[point].x;
			counts[cluster] += 1;
			// cout << cluster << endl;
		}

		// Divide sums by counts to get new centroids.
		for (size_t cluster = 0; cluster < k; ++cluster)
		{
			// Turn 0/0 into 0/1 to avoid zero division.
			const auto count = max<size_t>(1, counts[cluster]);
			means[cluster].x = new_means[cluster].x / count;
		}
	}
	assign = assignments;

	return means;
}
/*
DataFrame k_means_shared(const DataFrame &data, int *means_, size_t k,
						 size_t number_of_iterations, vector<size_t> &assign)
{
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
				// DataFrame local_new_mean = local_new_means[ithread];
				// vector<size_t> local_count = local_counts[ithread];

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
*/