#include "Sorting.h"

#include <vector>
#include <algorithm>
#include <mpi.h>

void Sorting::OddEvenSecvential(std::vector<int>& vecData)
{
	bool bSorted = false;
	while (!bSorted)
	{
		bSorted = true;
		for (int i = 1; i < vecData.size() - 1; i += 2)
		{
			if (vecData[i] > vecData[i + 1])
			{
				std::swap(vecData[i], vecData[i + 1]);
				bSorted = false;
			}
		}

		for (int i = 0; i < vecData.size() - 1; i += 2)
		{
			if (vecData[i] > vecData[i + 1])
			{
				std::swap(vecData[i], vecData[i + 1]);
				bSorted = false;
			}
		}
	}
}

void Sorting::MPI_OddEven(std::vector<int>& vecData, int nRank, int nSize)
{
	int n = vecData.size();
	int nLocalDataSize = n / nSize;

	std::vector<int> vecLocalData(nLocalDataSize);

	MPI_Scatter(vecData.data(), nLocalDataSize, MPI_INT, vecLocalData.data(), nLocalDataSize, MPI_INT, 0, MPI_COMM_WORLD);

	OddEvenSecvential(vecLocalData);

	for (int step = 0; step < nSize; step++)
	{
		int nPartner = (step % 2 == 0) ? nRank ^ 1 : nRank ^ 0;

		if (nPartner >= 0 && nPartner < nSize)
		{
			std::vector<int> vecRecvData(vecLocalData);

			MPI_Sendrecv(vecLocalData.data(), nLocalDataSize, MPI_INT, nPartner, 0, vecRecvData.data(), nLocalDataSize, MPI_INT, nPartner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			std::vector<int> vecMergeData(nLocalDataSize * 2);

			std::merge(vecLocalData.begin(), vecLocalData.end(), vecRecvData.begin(), vecRecvData.end(), vecMergeData.begin());

			if (nRank < nPartner)
			{
				std::copy(vecMergeData.begin(), vecMergeData.begin() + nLocalDataSize, vecLocalData.begin());
			}
			else
			{
				std::copy(vecMergeData.begin() + nLocalDataSize, vecMergeData.end(), vecLocalData.begin());
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Gather(vecLocalData.data(), nLocalDataSize, MPI_INT, vecData.data(), nLocalDataSize, MPI_INT, 0, MPI_COMM_WORLD);
}

void Sorting::MPI_OddEvenReplace(std::vector<int>& vecData, int nRank, int nSize)
{
	int n = vecData.size();
	int nLocalDataSize = n / nSize;

	std::vector<int> vecLocalData(nLocalDataSize);

	MPI_Scatter(vecData.data(), nLocalDataSize, MPI_INT, vecLocalData.data(), nLocalDataSize, MPI_INT, 0, MPI_COMM_WORLD);

	OddEvenSecvential(vecLocalData);

	for (int step = 0; step < nSize; step++)
	{
		if ((nRank + step) % 2 == 0)
		{
			if (nRank + 1 < nSize)
				MPI_Sendrecv_replace(vecLocalData.data(), nLocalDataSize, MPI_INT, nRank + 1, 0, nRank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		else
		{
			if (nRank > 0)
				MPI_Sendrecv_replace(vecLocalData.data(), nLocalDataSize, MPI_INT, nRank - 1, 0, nRank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		}

		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Gather(vecLocalData.data(), nLocalDataSize, MPI_INT, vecData.data(), nLocalDataSize, MPI_INT, 0, MPI_COMM_WORLD);
}

void Sorting::MPI_Sort(const std::vector<int>& vecData, int nRank, int nSize, std::function< void(std::vector< int >&) > sortFunction)
{
	int nGlobalSize = vecData.size();
	int nLocalSize = nGlobalSize / nSize;

	std::vector< int > vecLocalData(nLocalSize);

	MPI_Scatter(&vecData[0], nLocalSize, MPI_INT, &vecLocalData[0], nLocalSize, MPI_INT, 0, MPI_COMM_WORLD);

	sortFunction(vecLocalData);

	std::vector< int > vecGatherData(nGlobalSize);

	MPI_Gather(vecLocalData.data(), nLocalSize, MPI_INT, vecGatherData.data(), nLocalSize, MPI_INT, 0, MPI_COMM_WORLD);

	if (!nRank)
	{
		std::vector< int > vecSortedData(nLocalSize);
		int nStartIdx;
		for (int i = 0; i < nSize; i++)
		{
			nStartIdx = i * nLocalSize;
			std::vector< int > vecNextChunk(vecGatherData.begin() + nStartIdx, vecGatherData.begin() + nStartIdx + nLocalSize);

			vecSortedData = MergeArrays(vecSortedData, vecNextChunk);
		}
	}
}

void Sorting::MergeSort(std::vector< int >& vec)
{
	if (vec.size() <= 1)
		return;

	if (vec.size() == 2)
	{
		if (vec[0] > vec[1])
			std::swap(vec[0], vec[1]);

		return;
	}

	size_t mid = vec.size() / 2;
	std::vector< int > left(vec.begin(), vec.begin() + mid);
	std::vector< int > right(vec.begin() + mid, vec.end());

	MergeSort(left);
	MergeSort(right);

	vec = MergeArrays(left, right);
}

void Sorting::BubbleSort(std::vector< int >& vec)
{
	if (vec.size() <= 1)
		return;

	bool swapped;
	do
	{
		swapped = false;
		for (size_t i = 1; i < vec.size(); ++i)
		{
			if (vec[i - 1] > vec[i])
			{
				std::swap(vec[i - 1], vec[i]);
				swapped = true;
			}
		}
	} while (swapped);
}

std::vector< int > Sorting::MergeArrays(const std::vector< int >& a, const std::vector< int >& b)
{
	std::vector< int > c(a.size() + b.size());

	size_t i = 0, j = 0, k = 0;
	while (i < a.size() && j < b.size())
	{
		if (a[i] <= b[j]) {
			c[k++] = a[i++];
		}
		else {
			c[k++] = b[j++];
		}
	}
	while (i < a.size()) c[k++] = a[i++];
	while (j < b.size()) c[k++] = b[j++];

	return c;
}
