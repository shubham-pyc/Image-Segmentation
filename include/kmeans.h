#include <algorithm>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>
#include <iostream>
#include <omp.h>

using DataFrame = std::vector<Point>;

int THREAD_NUM = 4;

DataFrame k_means(const DataFrame &data, int *means_,
				  size_t k,
				  size_t number_of_iterations, std::vector<size_t> &assign)
{
	// static std::random_device seed;
	// static std::mt19937 random_number_generator(seed());
	// std::uniform_int_distribution<size_t> indices(0, data.size() - 1);

	// Pick centroids as random points from the dataset.
	DataFrame means;
	for (int i = 0; i < k; i++)
	{
		Point p = {.x = means_[i]};
		means.push_back(p);
	}

	std::vector<size_t> assignments(data.size());
	for (size_t iteration = 0; iteration < number_of_iterations; ++iteration)
	{
		// Find assignments.
		for (size_t point = 0; point < data.size(); ++point)
		{
			double best_distance = std::numeric_limits<double>::max();
			size_t best_cluster = 0;
			for (size_t cluster = 0; cluster < k; ++cluster)
			{
				const int distance =
					squared_l2_distance(data[point], means[cluster]);
				if (distance < best_distance)
				{
					best_distance = distance;
					best_cluster = cluster;
				}
			}
			assignments[point] = best_cluster;
		}

		// Sum up and count points for each cluster.
		DataFrame new_means(k);
		std::vector<size_t> counts(k, 0);
		for (size_t point = 0; point < data.size(); ++point)
		{
			const auto cluster = assignments[point];
			new_means[cluster].x += data[point].x;
			counts[cluster] += 1;
		}

		// Divide sums by counts to get new centroids.
		for (size_t cluster = 0; cluster < k; ++cluster)
		{
			// Turn 0/0 into 0/1 to avoid zero division.
			const auto count = std::max<size_t>(1, counts[cluster]);
			means[cluster].x = new_means[cluster].x / count;
		}
	}
	assign = assignments;

	return means;
}

DataFrame k_means_shared(const DataFrame &data, int *means_, size_t k,
						 size_t number_of_iterations, std::vector<size_t> &assign)
{
	DataFrame means;
	for (int i = 0; i < k; i++)
	{
		Point p = {.x = means_[i]};
		means.push_back(p);
	}

	std::vector<size_t> assignments(data.size());

	for (size_t iteration = 0; iteration < number_of_iterations; ++iteration)
	{
// Find assignments.
#pragma omp parallel for num_threads(THREAD_NUM)
		for (size_t point = 0; point < data.size(); ++point)
		{
			double best_distance = std::numeric_limits<double>::max();
			size_t best_cluster = 0;
			for (size_t cluster = 0; cluster < k; ++cluster)
			{
				const int distance =
					squared_l2_distance(data[point], means[cluster]);
				if (distance < best_distance)
				{
					best_distance = distance;
					best_cluster = cluster;
				}
			}
			assignments[point] = best_cluster;
		}

		// Sum up and count points for each cluster.
		DataFrame new_means(k);
		std::vector<size_t> counts(k, 0);
		for (size_t point = 0; point < data.size(); ++point)
		{
			const auto cluster = assignments[point];
			new_means[cluster].x += data[point].x;
			counts[cluster] += 1;
		}

// Divide sums by counts to get new centroids.
#pragma omp parallel for num_threads(THREAD_NUM)
		for (size_t cluster = 0; cluster < k; ++cluster)
		{
			// Turn 0/0 into 0/1 to avoid zero division.
			const auto count = std::max<size_t>(1, counts[cluster]);
			means[cluster].x = new_means[cluster].x / count;
		}
	}
	assign = assignments;

	return means;
}