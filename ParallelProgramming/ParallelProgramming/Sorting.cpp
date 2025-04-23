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
	int nGlobalSize = (nRank == 0 ? static_cast<int>(vecData.size()) : 0);
	MPI_Bcast(&nGlobalSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int baseSize = nGlobalSize / nSize;
	int remainder = nGlobalSize % nSize;

	std::vector<int> counts(nSize), displs(nSize);

	for (int i = 0; i < nSize; ++i) 
	{
		counts[i] = baseSize + (i < remainder ? 1 : 0);
		displs[i] = (i == 0 ? 0 : displs[i - 1] + counts[i - 1]);
	}

	std::vector<int> vecLocal(counts[nRank]);
	MPI_Scatterv(
		nRank == 0 ? vecData.data() : nullptr,
		counts.data(),
		displs.data(),
		MPI_INT,
		vecLocal.data(),
		counts[nRank],
		MPI_INT,
		0,
		MPI_COMM_WORLD
	);

	MPI_Barrier(MPI_COMM_WORLD);
	double t_start = MPI_Wtime();
	OddEvenSequential(vecLocal);
	double t_local = MPI_Wtime() - t_start;

	if (!std::is_sorted(vecLocal.begin(), vecLocal.end()))
		std::cerr << "[Rank " << nRank << "] ERROR: local sort failed\n";
	
	std::cout << "[Rank " << nRank << "] sorted " << counts[nRank] << " elems in " << t_local << " s\n";

	for (int phase = 0; phase < nSize; ++phase)
	{
		int partner = (phase % 2 == 0)
			? (nRank ^ 1)
			: ((nRank & 1) ? nRank - 1 : nRank + 1);

		if (partner >= 0 && partner < nSize) 
		{
			int sendCount = counts[nRank];
			int recvCount = counts[partner];
			std::vector<int> vecRecv(recvCount);

			MPI_Sendrecv(
				vecLocal.data(), sendCount, MPI_INT, partner, 0,
				vecRecv.data(), recvCount, MPI_INT, partner, 0,
				MPI_COMM_WORLD, MPI_STATUS_IGNORE
			);

			std::vector<int> merged(sendCount + recvCount);
			std::merge(
				vecLocal.begin(), vecLocal.end(),
				vecRecv.begin(), vecRecv.end(),
				merged.begin()
			);

			if (nRank < partner)
				std::copy_n(merged.begin(), sendCount, vecLocal.begin());
			else
				std::copy_n(merged.begin() + recvCount, sendCount, vecLocal.begin());
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	if (nRank == 0)
		vecData.resize(nGlobalSize);

	MPI_Gatherv(
		vecLocal.data(),
		counts[nRank],
		MPI_INT,
		nRank == 0 ? vecData.data() : nullptr,
		counts.data(),
		displs.data(),
		MPI_INT,
		0,
		MPI_COMM_WORLD
	);

	if (nRank == 0)
	{
		if (!std::is_sorted(vecData.begin(), vecData.end()))
			std::cerr << "[Rank 0] ERROR: global sort failed\n";
		else
			std::cout << "[Rank 0] SUCCESS: " << nGlobalSize << " elements sorted\n";
	}
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
	int n = static_cast<int>(vecData.size());

	std::vector<int> send_counts(nSize, n / nSize), displs(nSize, 0);

	int remainder = n % nSize;
	if (nRank == 0) 
	{
		for (int i = 0; i < remainder; ++i)
			send_counts[i]++;

		for (int i = 1; i < nSize; ++i)
			displs[i] = displs[i - 1] + send_counts[i - 1];
	}

	MPI_Bcast(send_counts.data(), nSize, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(displs.data(), nSize, MPI_INT, 0, MPI_COMM_WORLD);

	int local_size = send_counts[nRank];

	std::vector<int> vecLocalData(local_size);

	MPI_Scatterv(
		nRank == 0 ? vecData.data() : nullptr,
		send_counts.data(), displs.data(), MPI_INT,
		vecLocalData.data(), local_size, MPI_INT,
		0, MPI_COMM_WORLD
	);

	MPI_Barrier(MPI_COMM_WORLD);
	double t_start = MPI_Wtime();
	RankingSortSequential(vecLocalData);
	double t_comp = MPI_Wtime() - t_start;

	if (!std::is_sorted(vecLocalData.begin(), vecLocalData.end()))
		std::cerr << "[Rank " << nRank << "] ERROR: local chunk not sorted!\n";

	std::cout << "[Rank " << nRank << "] computation time: " << t_comp << " s for " << local_size << " elements\n";

	std::vector<int> recv_counts(nSize);
	MPI_Gather(&local_size, 1, MPI_INT,recv_counts.data(), 1, MPI_INT,0, MPI_COMM_WORLD);

	std::vector<int> displs_gather(nSize, 0);
	if (nRank == 0) 
	{
		for (int i = 1; i < nSize; ++i)
			displs_gather[i] = displs_gather[i - 1] + recv_counts[i - 1];
	}

	std::vector<int> final_data;

	if (nRank == 0)
		final_data.resize(n);

	MPI_Gatherv(
		vecLocalData.data(), local_size, MPI_INT,
		nRank == 0 ? final_data.data() : nullptr,
		recv_counts.data(), displs_gather.data(), MPI_INT,
		0, MPI_COMM_WORLD
	);

	MPI_Barrier(MPI_COMM_WORLD);

	if (nRank == 0) 
	{
		std::vector<std::pair<int*, int>> subarrays;
		int pos = 0;
		for (int i = 0; i < nSize; ++i) 
		{
			subarrays.emplace_back(final_data.data() + pos, recv_counts[i]);
			pos += recv_counts[i];
		}

		using Item = std::pair<int, std::pair<int*, int*>>;

		std::priority_queue<Item, std::vector<Item>, std::greater<>> pq;
		for (auto& p : subarrays)
		{
			if (p.second > 0)
				pq.push({ *p.first, { p.first, p.first + p.second } });
		}

		std::vector<int> merged;

		merged.reserve(n);
		while (!pq.empty()) 
		{
			auto [val, ptrs] = pq.top(); pq.pop();
			merged.push_back(val);

			if (++ptrs.first < ptrs.second) 
				pq.push({ *ptrs.first, ptrs });
		}
		vecData.swap(merged);

		if (!std::is_sorted(vecData.begin(), vecData.end()))
			std::cerr << "[Rank 0] ERROR: final merged data not sorted!\n";
		else
			std::cout << "[Rank 0] SUCCESS: all " << n << " elements sorted correctly.\n";
	}
}

void Sorting::MPI_Sort(std::vector<int>& vecData, int nRank, int nSize, std::function<void(std::vector<int>&)> sortFunction)
{
	int nGlobalSize = (nRank == 0 ? vecData.size() : 0);

	MPI_Bcast(&nGlobalSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int nLocalSize = nGlobalSize / nSize;
	int remainder = nGlobalSize % nSize;

	std::vector<int> counts(nSize), displs(nSize);

	for (int i = 0; i < nSize; ++i) 
	{
		counts[i] = nLocalSize + (i < remainder ? 1 : 0);
		displs[i] = (i == 0 ? 0 : displs[i - 1] + counts[i - 1]);
	}

	std::vector<int> vecLocalData(counts[nRank]);
	MPI_Scatterv(
		nRank == 0 ? vecData.data() : nullptr,
		counts.data(), displs.data(), MPI_INT,
		vecLocalData.data(), counts[nRank], MPI_INT,
		0, MPI_COMM_WORLD
	);

	MPI_Barrier(MPI_COMM_WORLD);
	double t_comp_start = MPI_Wtime();
	sortFunction(vecLocalData);
	double t_comp = MPI_Wtime() - t_comp_start;

	if (!std::is_sorted(vecLocalData.begin(), vecLocalData.end()))
		std::cerr << "[Rank " << nRank << "] ERROR: local data NOT sorted!\n";

	std::cout << "[Rank " << nRank << "] sorted " << counts[nRank] << " elements in " << t_comp << " s\n";

	std::vector<int> vecGatherData;
	if (nRank == 0) 
		vecGatherData.resize(nGlobalSize);

	MPI_Gatherv(
		vecLocalData.data(), counts[nRank], MPI_INT,
		nRank == 0 ? vecGatherData.data() : nullptr,
		counts.data(), displs.data(), MPI_INT,
		0, MPI_COMM_WORLD
	);

	if (nRank == 0) 
	{
		std::vector<int> merged(
			vecGatherData.begin(),
			vecGatherData.begin() + counts[0]
		);
		int offset = counts[0];

		for (int i = 1; i < nSize; ++i)  
		{
			std::cout << "[Merge] combining chunk " << (i + 1) << "/" << nSize << " (" << counts[i] << " elems)\n";

			std::vector<int> next(
				vecGatherData.begin() + offset,
				vecGatherData.begin() + offset + counts[i]
			);
			merged = MergeArrays(merged, next);
			offset += counts[i];
		}
		vecData.swap(merged);

		if (!std::is_sorted(vecData.begin(), vecData.end()))
			std::cerr << "[Rank 0] ERROR: final merged data NOT sorted!\n";
		else
			std::cout << "[Rank 0] SUCCESS: all " << nGlobalSize << " elements sorted correctly.\n";
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

void Sorting::BucketSortSequential(std::vector<int>& arr) 
{
	int n = arr.size();
	if (n <= 1) 
		return;

	int maxVal = *std::max_element(arr.begin(), arr.end());

	int bucketCount = 10;
	std::vector<std::vector<int>> buckets(bucketCount);

	for (int num : arr) 
	{
		int index = (num * bucketCount) / (maxVal + 1);
		buckets[index].push_back(num);
	}

	for (auto& bucket : buckets) 
		Sorting::BubbleSort(bucket);

	int idx = 0;
	for (const auto& bucket : buckets) 
	{
		for (int val : bucket)
			arr[idx++] = val;
	}
}

void Sorting::MPI_Bucket_sort(std::vector<int>& vecData, int nRank, int nSize)
{
	int local_max = nRank == 0 ? *std::max_element(vecData.begin(), vecData.end()) : 0;
	MPI_Bcast(&local_max, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int n = vecData.size();
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	std::vector<int> local_a;
	if (nRank == 0) 
		local_a = vecData;
	else
		local_a.resize(n);

	MPI_Bcast(local_a.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

	std::vector<int> bucket;

	const int bucket_lower = nRank * local_max / nSize;
	const int bucket_upper = (nRank + 1) * local_max / nSize;

	for (int val : local_a) 
	{
		if (val >= bucket_lower && val < bucket_upper)
			bucket.push_back(val);
	}

	double time = MPI_Wtime();

	std::sort(bucket.begin(), bucket.end());

	time = MPI_Wtime() - time;

	std::cout << "Processor " << nRank << " has sorted " << bucket.size() << " elements in " << time << " seconds." << std::endl;

	std::vector<int> counts(nSize), displs(nSize);
	int local_count = bucket.size();
	MPI_Gather(&local_count, 1, MPI_INT,
		counts.data(), 1, MPI_INT,
		0, MPI_COMM_WORLD);

	if (nRank == 0) 
	{
		displs[0] = 0;

		for (int i = 1; i < nSize; ++i)
			displs[i] = displs[i - 1] + counts[i - 1];

		vecData.resize(displs.back() + counts.back());
	}

	MPI_Gatherv(bucket.data(), local_count, MPI_INT,
		vecData.data(), counts.data(), displs.data(), MPI_INT,
		0, MPI_COMM_WORLD);
}

void Sorting::ShellSortSequential(std::vector<int>& vec)
{
	int n = vec.size();
	if (n <= 1) 
		return;

	for (int gap = n / 2; gap > 0; gap /= 2)
	{
		for (int i = gap; i < n; i += 1)
		{
			int temp = vec[i];

			int j;
			for (j = i; j >= gap && vec[j - gap] > temp; j -= gap)
				vec[j] = vec[j - gap];

			vec[j] = temp;
		}
	}
}

void Sorting::MPI_ShellSort(std::vector<int>& data, int rank, int size)
{
	int n;
	if (rank == 0)
		n = static_cast<int>(data.size());

	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int local_size = n / size;
	if (n % size != 0 && rank == 0) 
		std::cerr << "Warning: data.size() not divisible by size – " << "truncating extra " << (n % size) << " elements\n";

	std::vector<int> local_data(local_size);

	MPI_Scatter(
		rank == 0 ? data.data() : nullptr,
		local_size,
		MPI_INT,
		local_data.data(),
		local_size,
		MPI_INT,
		0,
		MPI_COMM_WORLD
	);

	double local_start = MPI_Wtime();
	for (int gap = local_size / 2; gap > 0; gap /= 2) 
	{
		for (int i = gap; i < local_size; ++i) 
		{
			int temp = local_data[i], j = i;

			while (j >= gap && local_data[j - gap] > temp) 
			{
				local_data[j] = local_data[j - gap];
				j -= gap;
			}

			local_data[j] = temp;
		}
	}

	double local_end = MPI_Wtime();
	double local_duration = local_end - local_start;

	MPI_Gather(
		local_data.data(),
		local_size,
		MPI_INT,
		rank == 0 ? data.data() : nullptr,
		local_size,
		MPI_INT,
		0,
		MPI_COMM_WORLD
	);

	std::cout << "Rank " << rank << " time: " << local_duration << " seconds\n";

	if (rank == 0) 
	{
		if (!std::is_sorted(data.begin(), data.end()))
			std::sort(data.begin(), data.end());
	}
}


std::vector< int > Sorting::MergeArrays(const std::vector< int >& a, const std::vector< int >& b)
{
	std::vector< int > c(a.size() + b.size());

	size_t i = 0, j = 0, k = 0;
	while (i < a.size() && j < b.size())
	{
		if (a[i] <= b[j])
			c[k++] = a[i++];
		else
			c[k++] = b[j++];
	}

	while (i < a.size()) 
		c[k++] = a[i++];

	while (j < b.size())
		c[k++] = b[j++];

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
