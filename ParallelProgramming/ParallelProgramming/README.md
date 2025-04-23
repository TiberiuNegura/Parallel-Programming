**README for Parallel Sort Investigation**

**Student Details**  
- Name: _Your Name_  
- Program: PPDC 2025  
- Group: _Your Group_

**Computer Configuration**  
- CPU: _e.g., Intel Core i7-9700K @ 3.60GHz_  
- GPU (if used): _e.g., NVIDIA GTX 1080 Ti_  
- Memory: _e.g., 16 GB DDR4_  
- Compiler and MPI Version: _e.g., gcc 9.4.0, OpenMPI 4.0.5_
- Operating System: _e.g., Ubuntu 20.04.3 LTS_

---

## 1. Dataset Preparation
- Description of dataset (size, distribution, source file name):  
  - Number of elements: _e.g., 10,000,000_  
  - Data characteristics: _random, sorted, reverse-sorted, etc._  
- File format: plain text with one number per line ("data.txt").  

---

## 2. Methodology
- How the array is read and distributed across MPI processes.  
- Timing methodology:  
  - Wall-clock measurement (MPI_Wtime)  
  - Computation time vs. communication time (use MPI_Barrier, separate timers).  
- Repeated runs and averaging strategy (e.g., 5 runs per algorithm).  

---

## 3. Performance Results

| Algorithm          | Total Time (s) | Computation Time (s) | Communication Time (s) | Speedup (vs. sequential) | Efficiency (%) |
|--------------------|----------------|----------------------|------------------------|--------------------------|----------------|
| Direct Sort        | _t1_           | _c1_                 | _m1_                   | _s1_                     | _e1_           |
| Bucket Sort        | _t2_           | _c2_                 | _m2_                   | _s2_                     | _e2_           |
| Odd-Even Sort      | _t3_           | _c3_                 | _m3_                   | _s3_                     | _e3_           |
| Ranking Sort       | _t4_           | _c4_                 | _m4_                   | _s4_                     | _e4_           |
| Shell Sort         | _t5_           | _c5_                 | _m5_                   | _s5_                     | _e5_           |

*Note: Speedup = T_seq / T_p; Efficiency = Speedup / #processes × 100%*

---

## 4. Scalability Analysis
- Experimental setup: list of process counts tested (e.g., 2, 4, 8, 16).  
- Observed scaling behavior for each algorithm (include plots if available).  
- Discussion of whether each algorithm shows strong or weak scaling.

---

## 5. Comparative Analysis
- Summarize which algorithm performed best in total time.  
- Communication overhead sensitivity:
  - Which sorting method suffered most from increased messaging?
  - Which was least sensitive?
- Identify computational bottlenecks for each method.

---

## 6. Speedup and Efficiency Discussion
- Theoretical vs. actual speedup gaps: formulate Amdahl’s Law where applicable.  
- Explain factors causing deviation (load imbalance, communication latency, synchronization).

---

## 7. Suggestions for Improvement
- Load balancing strategies (e.g., dynamic bucket allocation).  
- Reducing communication (e.g., aggregating messages).  
- Algorithmic optimizations (e.g., improved pivot selection for direct sort).

---

## 8. Conclusion
- Key takeaways about parallel sorting performance.  
- Recommendations for future work.

---

*End of README Template. Fill in each placeholder (`_..._`) with your measured data and observations.*

