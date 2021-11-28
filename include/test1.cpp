#include <iostream>
#include <random>
#include <vector>

using namespace std;

int main()
{
	// static std::random_device seed;
	// static std::mt19937 random_number_generator(seed());
	// std::uniform_int_distribution<size_t> indices(0, 10 - 1);

	// for (size_t i = 0; i < 10; i++)
	// {
	// cout << indices(random_number_generator) << endl;
	// }

	vector<int> arr;

	for (int i = 0; i < 10; i++)
	{
		arr.push_back(i);
	}

	// for (auto i = arr.begin(); i != arr.end(); i++)
	// {
	// cout << i << endl;
	// }

	for (int i = 0; i < arr.size(); i++)
	{
		cout << arr[i] << " ";
	}

	return 0;
}