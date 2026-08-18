# HPC Project: Parallelizing Neural Network Operations using Sequential, OpenMP, and CUDA

## Overview

This project was developed to explore the concepts of **High Performance Computing (HPC)** by implementing some fundamental neural network operations using three different execution models:

* **Sequential Programming** (Single-threaded CPU)
* **OpenMP** (Multi-threaded CPU)
* **CUDA** (GPU Parallel Computing)

The main goal is to understand how parallel programming improves execution time and to compare the performance of CPU and GPU implementations through benchmarking and visualization.

Instead of focusing on deep learning frameworks like TensorFlow or PyTorch, this project implements the core operations manually in C/CUDA to gain a better understanding of how these computations work internally.

---

# Objectives

The primary objectives of this project are:

* Implement basic neural network operations from scratch.
* Compare Sequential, OpenMP, and CUDA implementations.
* Measure execution time for each approach.
* Calculate the speedup achieved through parallelization.
* Verify that all implementations produce identical results.
* Visualize benchmark results using graphs.

---

# Operations Implemented

The project implements three commonly used neural network operations.

## 1. Convolution

Convolution is one of the most important operations used in Convolutional Neural Networks (CNNs). It extracts useful features from an input image by sliding a kernel over the image.

Files:

```text
conv_seq.c
conv_omp.c
conv_cuda.cu
```

Three versions of the same algorithm are implemented:

* Sequential CPU
* OpenMP Parallel CPU
* CUDA GPU

---

## 2. Fully Connected Layer

The Fully Connected (FC) layer performs matrix multiplication between the input and the weight matrix to generate the final output.

Files:

```text
fc_seq.c
fc_omp.c
fc_cuda.cu
```

Again, the same computation is implemented using Sequential, OpenMP, and CUDA.

---

## 3. Max Pooling

Pooling reduces the spatial dimensions of feature maps while preserving important information.

Files:

```text
pool_seq.c
pool_omp.c
pool_cuda.cu
```

The project compares CPU and GPU implementations of this operation.

---

# Project Structure

```text
HPC-project/
│
├── common.h                    # Shared definitions and helper functions
│
├── conv_seq.c                  # Sequential Convolution
├── conv_omp.c                  # OpenMP Convolution
├── conv_cuda.cu                # CUDA Convolution
│
├── fc_seq.c                    # Sequential Fully Connected Layer
├── fc_omp.c                    # OpenMP Fully Connected Layer
├── fc_cuda.cu                  # CUDA Fully Connected Layer
│
├── pool_seq.c                  # Sequential Max Pooling
├── pool_omp.c                  # OpenMP Max Pooling
├── pool_cuda.cu                # CUDA Max Pooling
│
├── run_benchmarks.py           # Runs all implementations and collects timings
├── plot_results.py             # Generates performance graphs
├── check_equivalence.py        # Verifies correctness of outputs
├── test_real_image.py          # Tests the implementations on a sample image
│
├── benchmark_data.csv          # Benchmark results
│
├── conv_execution_time.png
├── conv_speedup.png
├── fc_execution_time.png
├── fc_speedup.png
├── pool_execution_time.png
├── pool_speedup.png
│
└── README.md
```

---

# Workflow

The project follows a simple workflow.

```text
                 run_benchmarks.py
                         │
                         ▼
      Runs Sequential, OpenMP and CUDA Programs
                         │
                         ▼
            Measures Execution Time
                         │
                         ▼
             benchmark_data.csv
                         │
                         ▼
                plot_results.py
                         │
                         ▼
        Execution Time & Speedup Graphs
```

Whenever benchmarking is required, you only need to execute the benchmark script.

---

# Main Components

## 1. run_benchmarks.py

This is the **main orchestrator** of the project.

It automatically:

* Executes every Sequential implementation.
* Executes every OpenMP implementation.
* Executes every CUDA implementation.
* Measures execution time.
* Stores all benchmark values in **benchmark_data.csv**.

This script eliminates the need to manually execute every program one by one.

---

## 2. plot_results.py

This script reads the benchmark data and generates graphs showing:

* Execution Time Comparison
* Speedup Comparison

The graphs help visualize the benefits of parallel programming.

---

## 3. check_equivalence.py

Performance is meaningless if the outputs are incorrect.

This script compares the outputs produced by the Sequential, OpenMP, and CUDA implementations to ensure they all generate identical results.

---

## 4. test_real_image.py

This script demonstrates the implemented operations on a real image instead of randomly generated data.

It serves as a practical example of using the project.

---

# Technologies Used

* C
* CUDA C
* OpenMP
* Python
* NumPy
* Matplotlib

---

# Compilation

## Sequential

```bash
gcc conv_seq.c -o conv_seq
gcc fc_seq.c -o fc_seq
gcc pool_seq.c -o pool_seq
```

---

## OpenMP

```bash
gcc -fopenmp conv_omp.c -o conv_omp
gcc -fopenmp fc_omp.c -o fc_omp
gcc -fopenmp pool_omp.c -o pool_omp
```

---

## CUDA

```bash
nvcc conv_cuda.cu -o conv_cuda
nvcc fc_cuda.cu -o fc_cuda
nvcc pool_cuda.cu -o pool_cuda
```

---

# Running Benchmarks

Execute

```bash
python run_benchmarks.py
```

The script runs all implementations automatically and stores the timing results inside

```text
benchmark_data.csv
```

---

# Generating Graphs

After benchmarking,

```bash
python plot_results.py
```

This generates graphs for:

* Execution Time
* Speedup

for every implemented operation.

---

# Correctness Verification

To verify that every implementation produces identical outputs,

```bash
python check_equivalence.py
```

---

# Performance Evaluation

The project compares three execution models.

### Sequential

* Single CPU thread
* Simple implementation
* Baseline for comparison

### OpenMP

* Multiple CPU threads
* Faster than Sequential
* Uses shared-memory parallelism

### CUDA

* Executes on GPU
* Massive parallelism
* Provides the highest performance for computationally intensive tasks

The benchmark results clearly demonstrate the improvement achieved through parallel computing.

---

# Learning Outcomes

Through this project, I gained practical experience with:

* High Performance Computing concepts
* CPU parallel programming using OpenMP
* GPU programming using CUDA
* Performance benchmarking
* Speedup analysis
* Parallel algorithm design
* Result visualization using Python

---

# Future Improvements

Some possible extensions include:

* Support for larger datasets
* Batch processing
* Additional neural network layers
* Better CUDA memory optimization
* Multi-GPU execution
* Automated build system using Makefile or CMake

---

# Author

**Atul Kumar Tiwari**

This project was created as part of my learning journey in **High Performance Computing (HPC)** to understand and compare Sequential, OpenMP, and CUDA programming through practical implementations and performance analysis.
