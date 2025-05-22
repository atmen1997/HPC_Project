# HPC_Project
## MPI-Based Parallel Branch-and-Bound for the Wandering Salesman Problem (WSP)

This project implements a parallelized version of the Branch-and-Bound algorithm using **MPI** to solve a variation of the Traveling Salesman Problem: the **Wandering Salesman Problem**, where the route does **not** return to the starting city.

---

## 🚀 Problem Summary

Given 17 cities, find the shortest possible route that visits each city **exactly once**, without returning to the origin. This is a combinatorial optimization problem with a factorial search space.

---

## 🧱 Implementations

### 🔸 Serial
- Classic recursive Branch-and-Bound in C
- No pruning based on lower bounds, only upper bound cutoff

### 🔸 Parallel MPI — SPMD (No Communication)
- Static round-robin task distribution
- Processes compute independently
- Final result merged with `MPI_Allreduce`
- Extremely low communication overhead, but suffers from load imbalance

### 🔸 Parallel MPI — Master-Worker (With Communication)
- Dynamic task distribution via a shared task queue
- Workers request new tasks and report better paths
- Global best path shared during runtime to improve pruning
- Introduces communication but improves efficiency

---

## 📊 Benchmark Results

### Serial Baseline:
- 17 cities
- Best distance: `278`
- Execution time: `~201.39 s`

### Best Parallel Results:

| Model         | Tasks     | Processes | Time (s) | Speedup | Efficiency |
|---------------|-----------|-----------|----------|---------|------------|
| SPMD (No Comm)| 524,160   | 64        | 19.76    | 10.19×  | 16%        |
| Master-Worker | 524,160   | 64        | 4.46     | 45.12×  | 70%        |

> ✅ Dynamic communication enabled better pruning and load balancing, resulting in a **4x faster** solution than static distribution.

---

## 📈 Performance Analysis

### Speedup (with communication)
- Peaks at **45.12× speedup** using 64 processes
- Achieved **70.5% of ideal speedup**

### Efficiency
- Efficiency decreases with more processes in static models
- Master-Worker approach maintains ~70% even at 64 processes

### Communication Overhead
- No-Comm model has hidden overhead from load imbalance
- With communication: overhead remains below **5%**

---

## ⚙️ Software and Tools

- **Language**: C
- **Parallelization**: MPI (Message Passing Interface)
- **Cluster**: Crescent 2 HPC (Intel MPI, PBS Scheduler)

---

## 🧠 Key Takeaways

- Static work distribution is simple but inefficient at scale
- Dynamic task queues with shared best solutions greatly improve pruning and performance
- Trade-off: introducing communication overhead yields better load balance and faster convergence
