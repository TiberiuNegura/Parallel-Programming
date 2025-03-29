#include "Sorting.h"

#include <vector>
#include <iostream>
#include <tuple>
#include <queue>
#include <algorithm>
#include <mpi.h>

void Sorting::OddEvenSequential(std::vector<int>& vecData)
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

void Sorting::RankingSortSequential(std::vector<int>& vecInitialData)
{
	std::vector<int> vecOutData(vecInitialData.size());
	const int n = vecInitialData.size();

	for (int i = 0; i < n; i++) 
	{
		int nRank = 0;
		for (int j = 0; j < n; j++) 
		{
			if (vecInitialData[j] < vecInitialData[i] || (vecInitialData[j] == vecInitialData[i] && j < i))
				nRank++;
		}
		vecOutData[nRank] = vecInitialData[i];
	}

	vecInitialData = vecOutData;
}

void Sorting::MPI_OddEven(std::vector<int>& vecData, int nRank, int nSize)
{
	int n = vecData.size();
	int nLocalDataSize = n / nSize;

	std::vector<int> vecLocalData(nLocalDataSize);

	MPI_Scatter(vecData.data(), nLocalDataSize, MPI_INT, vecLocalData.data(), nLocalDataSize, MPI_INT, 0, MPI_COMM_WORLD);

	OddEvenSequential(vecLocalData);

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

	OddEvenSequential(vecLocalData);

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

void Sorting::MPI_RankingSort(std::vector<int>& vecData, int nRank, int nSize)
{
	int n = vecData.size();

	std::vector<int> send_counts(nSize, n / nSize);
	std::vector<int> displs(nSize, 0);
	int remainder = n % nSize;

	if (nRank == 0) 
	{
		for (int i = 0; i < remainder; i++) 
			send_counts[i]++;
		for (int i = 1; i < nSize; i++) 
			displs[i] = displs[i - 1] + send_counts[i - 1];
	}

	MPI_Bcast(send_counts.data(), nSize, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(displs.data(), nSize, MPI_INT, 0, MPI_COMM_WORLD);

	int local_size = send_counts[nRank];
	std::vector<int> vecLocalData(local_size);

	MPI_Scatterv(vecData.data(), send_counts.data(), displs.data(), MPI_INT,
		vecLocalData.data(), local_size, MPI_INT,
		0, MPI_COMM_WORLD);

	RankingSortSequential(vecLocalData);

	std::vector<int> recv_counts(nSize);
	MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

	std::vector<int> final_data;
	std::vector<int> displs_gather(nSize, 0);

	if (nRank == 0) 
	{
		final_data.resize(n);
		for (int i = 1; i < nSize; i++)
			displs_gather[i] = displs_gather[i - 1] + recv_counts[i - 1];
	}

	MPI_Gatherv(vecLocalData.data(), local_size, MPI_INT, final_data.data(), recv_counts.data(), displs_gather.data(), MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);

	if (nRank == 0)
	{
		std::vector<std::pair<int*, int>> subarrays;
		int current_pos = 0;

		for (int i = 0; i < nSize; i++)
		{
			subarrays.emplace_back(final_data.data() + current_pos, recv_counts[i]);
			current_pos += recv_counts[i];
		}

		std::priority_queue<std::pair<int, std::pair<int*, int*>>, std::vector<std::pair<int, std::pair<int*, int*>>>, std::greater<> > pq;

		for (auto& [start, size] : subarrays)
		{
			if (size > 0)
				pq.push({ *start, {start, start + size} });
		}

		std::vector<int> merged;
		merged.reserve(n);

		while (!pq.empty())
		{
			auto [val, ptrs] = pq.top();
			pq.pop();
			merged.push_back(val);

			if (++ptrs.first < ptrs.second)
				pq.push({ *ptrs.first, {ptrs.first, ptrs.second} });
		}

		vecData = merged;
	}
}

void Sorting::MPI_Sort(std::vector<int>& vecData, int nRank, int nSize, std::function< void(std::vector< int >&) > sortFunction)
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

		vecData = vecSortedData;
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

void Sorting::BucketSortSequential(std::vector<int>& arr) {
	int n = arr.size();
	if (n <= 1) return;

	// Find the maximum value to determine range
	int maxVal = *std::max_element(arr.begin(), arr.end());

	// Number of buckets (adjustable)
	int bucketCount = 10;
	std::vector<std::vector<int>> buckets(bucketCount);

	// Insert elements into respective buckets
	for (int num : arr) {
		int index = (num * bucketCount) / (maxVal + 1); // Normalize to bucket index
		buckets[index].push_back(num);
	}

	// Sort individual buckets
	for (auto& bucket : buckets) {
		Sorting::BubbleSort(bucket);
		// TODO EXPERIMENT WITH std::sort(bucket.begin(), bucket.end());
	}

	// Concatenate all buckets into arr
	int idx = 0;
	for (const auto& bucket : buckets) {
		for (int val : bucket) {
			arr[idx++] = val;
		}
	}
}

void Sorting::MPI_Bucket_sort(std::vector<int>& vecData, int nRank, int nSize) 
{
	// Broadcast the max value from array
	int local_max = nRank == 0 ? *std::max_element(vecData.begin(), vecData.end()) : 0;
	MPI_Bcast(&local_max, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// Broadcast input array
	int n = vecData.size();
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	std::vector<int> local_a;
	if (nRank == 0) local_a = vecData;
	else local_a.resize(n);

	MPI_Bcast(local_a.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

	// Collect bucket elements
	std::vector<int> bucket;
	const int bucket_lower = nRank * local_max / nSize;
	const int bucket_upper = (nRank + 1) * local_max / nSize;

	for (int val : local_a) {
		if (val >= bucket_lower && val < bucket_upper) {
			bucket.push_back(val);
		}
	}

	double time = MPI_Wtime();

	// Sort local bucket
	std::sort(bucket.begin(), bucket.end());

	time = MPI_Wtime() - time;

	std::cout << "Processor " << nRank << " has sorted " << bucket.size() << " elements in " << time << " seconds." << std::endl;

	// Gather bucket sizes
	std::vector<int> counts(nSize), displs(nSize);
	int local_count = bucket.size();
	MPI_Gather(&local_count, 1, MPI_INT,
		counts.data(), 1, MPI_INT,
		0, MPI_COMM_WORLD);

	// Prepare output buffer
	if (nRank == 0) {
		displs[0] = 0;
		for (int i = 1; i < nSize; ++i) {
			displs[i] = displs[i - 1] + counts[i - 1];
		}
		vecData.resize(displs.back() + counts.back());
	}

	// Gather sorted buckets
	MPI_Gatherv(bucket.data(), local_count, MPI_INT,
		vecData.data(), counts.data(), displs.data(), MPI_INT,
		0, MPI_COMM_WORLD);
}

void Sorting::ShellSortSequential(std::vector<int>& vec)
{
	int n = vec.size();
	if (n <= 1) return;

	// Start with a big gap, then reduce the gap
	for (int gap = n / 2; gap > 0; gap /= 2)
	{
		// Do a gapped insertion sort for this gap size.
		// The first gap elements arr[0..gap-1] are already in gapped order
		// keep adding one more element until the entire array is gap sorted
		for (int i = gap; i < n; i += 1)
		{
			// add arr[i] to the elements that have been gap sorted
			// save arr[i] in temp and make a hole at position i
			int temp = vec[i];

			// shift earlier gap-sorted elements up until the correct location for arr[i] is found
			int j;
			for (j = i; j >= gap && vec[j - gap] > temp; j -= gap)
				vec[j] = vec[j - gap];

			// put temp (the original arr[i]) in its correct location
			vec[j] = temp;
		}
	}
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

bool Sorting::is_sorted_global(std::vector<int>& local_data, int rank, int size)
{
	std::vector<int> first_elements(size, 0);
	std::vector<int> last_elements(size, 0);
	int first = local_data.empty() ? 0 : local_data.front();
	int last = local_data.empty() ? 0 : local_data.back();

	MPI_Gather(&first, 1, MPI_INT, first_elements.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(&last, 1, MPI_INT, last_elements.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

	bool isSorted = true;

	if (rank == 0)
	{
		for (int i = 0; i < size - 1; i++)
		{
			if (first_elements[i] > last_elements[i + 1])
			{
				isSorted = false;
				break;
			}
		}
	}

	MPI_Bcast(&isSorted, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
	return isSorted;
}
