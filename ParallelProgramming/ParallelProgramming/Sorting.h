#include <vector>
#include <functional>
#include <mpi.h>

class Sorting
{
public:	
	static void OddEvenSequential(std::vector<int>& vecData);

	static void RankingSortSequential(std::vector<int>& vecInitialData);

	static void MPI_OddEven(std::vector< int >& vecData, int nRank, int nSize);

	static void MPI_OddEvenReplace(std::vector< int >& vecData, int nRank, int nSize);

	static void MPI_Sort(const std::vector< int >& vecData, int nRank, int nSize, std::function< void(std::vector< int >&) > sortFunction);

	static void MPI_Bucket_sort(std::vector< int >& vecData, int nRank, int nSize);
	
	static void MPI_RankingSort(std::vector<int>& vecData, int nRank, int nSize);

	static void MPI_Sort(std::vector<int>& vecData, int nRank, int nSize, std::function< void(std::vector< int >&) > sortFunction);

	static void MergeSort(std::vector< int >& vec);

	static void BubbleSort(std::vector< int >& vec);

	static void BucketSortSequential(std::vector< int >& vec);

	static void ShellSortSequential(std::vector< int >& vec);

	static std::vector< int > MergeArrays(const std::vector< int >& a, const std::vector< int >& b);

	static void MPI_ShellSort(std::vector<int>& data, int rank, int size);

private:
    bool is_sorted_global(std::vector< int >& local_data, int p, int rank);
};