#include <iostream>
#include <filesystem>
#include <mpi.h>
#include <fstream>
#include <thread>

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
	MPI_Init(&argc, &argv);

	double time = MPI_Wtime();

	std::vector<int> numbers = readFromFile("input.txt");

	time = MPI_Wtime() - time;

	std::cout << "Time to read file: " << time << std::endl;

	return 0;
}