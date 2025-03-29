#include <iostream>
#include <filesystem>
#include <mpi.h>
#include <fstream>
#include <thread>
#include "Sorting.h"

std::vector<int> ReadFromFile(const std::string& filename) 
{
	std::vector<int> numbers;
	std::ifstream inFile(filename);

	if (inFile.is_open()) 
	{
		int num;
		while (inFile >> num) 
			numbers.push_back(num);
		inFile.close();

	}
	else 
		std::cerr << "Error opening file!" << std::endl;

	return numbers;
}

int main(int argc, char** argv)
{
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	double time = MPI_Wtime();
	std::vector<int> numbers;

	if (rank == 0)
	{
	numbers = ReadFromFile("input.txt");
	time = MPI_Wtime() - time;
	std::cout << "Time to read file: " << time << std::endl;
	}

	time = MPI_Wtime();

	Sorting::MPI_Bucket_sort(numbers);

	time = MPI_Wtime() - time;

	if (rank == 0)
	{
		std::cout << "Time to sort: " << time << std::endl;
	}

	if (rank == 0)
		for (int i = 0; i < numbers.size() - 1; i++)
			if (numbers[i] > numbers[i + 1])
				std::cout << "Error at index " << i << std::endl;

	MPI_Finalize();

	return 0;
}