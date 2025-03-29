#include <vector>
#include <functional>

class Sorting
{
public:	
	static void OddEvenSecvential(std::vector<int>& vecData);

	static void MPI_OddEven(std::vector<int>& vecData, int nRank, int nSize);

	static void MPI_OddEvenReplace(std::vector<int>& vecData, int nRank, int nSize);

	static void MPI_Sort(const std::vector<int>& vecData, int nRank, int nSize, std::function< void(std::vector< int >&) > sortFunction);

	static void MergeSort(std::vector< int >& vec);

	static void BubbleSort(std::vector< int >& vec);

	static std::vector< int > MergeArrays(const std::vector< int >& a, const std::vector< int >& b);
};