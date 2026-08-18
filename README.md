# HPC-project

A simple High Performance Computing (HPC) project that compares **Sequential**, **OpenMP**, and **CUDA** implementations of fundamental neural network operations. The objective is to analyze the performance improvements achieved through CPU multi-threading and GPU acceleration.

This project was developed to understand parallel programming concepts, benchmark execution times, and visualize the speedup obtained using different computing paradigms.

---

## Features

* Sequential CPU implementation
* OpenMP-based parallel CPU implementation
* CUDA-based GPU implementation
* Benchmark automation using Python
* Performance comparison through graphs
* Correctness verification between implementations

---

## Operations Implemented

### 1. Convolution

* Sequential (`conv_seq.c`)
* OpenMP (`conv_omp.c`)
* CUDA (`conv_cuda.cu`)

### 2. Fully Connected Layer

* Sequential (`fc_seq.c`)
* OpenMP (`fc_omp.c`)
* CUDA (`fc_cuda.cu`)

### 3. Max Pooling

* Sequential (`pool_seq.c`)
* OpenMP (`pool_omp.c`)
* CUDA (`pool_cuda.cu`)

---

## Project Structure

```text
HPC-project/
│
├── common.h
│
├── conv_seq.c
├── conv_omp.c
├── conv_cuda.cu
│
├── fc_seq.c
├── fc_omp.c
├── fc_cuda.cu
│
├── pool_seq.c
├── pool_omp.c
├── pool_cuda.cu
│
├── run_benchmarks.py
├── plot_results.py
├── check_equivalence.py
├── test_real_image.py
│
├── benchmark_data.csv
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

## Technologies Used

* C
* CUDA C
* OpenMP
* Python
* NumPy
* Matplotlib

---

## Running the Project

### Compile Sequential

```bash
gcc conv_seq.c -o conv_seq
gcc fc_seq.c -o fc_seq
gcc pool_seq.c -o pool_seq
```

### Compile OpenMP

```bash
gcc -fopenmp conv_omp.c -o conv_omp
gcc -fopenmp fc_omp.c -o fc_omp
gcc -fopenmp pool_omp.c -o pool_omp
```

### Compile CUDA

```bash
nvcc conv_cuda.cu -o conv_cuda
nvcc fc_cuda.cu -o fc_cuda
nvcc pool_cuda.cu -o pool_cuda
```

---

## Benchmarking

Run the benchmark script:

```bash
python run_benchmarks.py
```

The benchmark data will be stored in:

```text
benchmark_data.csv
```

---

## Plot Performance Graphs

```bash
python plot_results.py
```

The script generates graphs comparing execution time and speedup for all implementations.

---

## Verify Correctness

```bash
python check_equivalence.py
```

This checks whether the outputs produced by the Sequential, OpenMP, and CUDA implementations are equivalent.

---

## Performance Analysis

The project compares three execution models:

* Sequential CPU
* OpenMP Parallel CPU
* CUDA GPU

Performance is evaluated using:

* Execution Time
* Speedup
* Correctness of Results

The generated graphs provide a visual comparison of the improvements achieved through parallelization.

---

## Future Improvements

* Support larger image datasets
* Add batch processing
* Optimize CUDA memory access
* Implement additional neural network layers
* Extend benchmarking to multi-GPU systems

---

## Author

**Atul Kumar Tiwari**

If you found this project useful or interesting, feel free to star the repository.
