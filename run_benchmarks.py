import ctypes
import numpy as np
import subprocess
import os
import time
import csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNELS_DIR = os.path.join(PROJECT_ROOT, "kernels")
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(BUILD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_RUNS = 5
WARMUP_RUNS = 2  # extra untimed calls before timing, to absorb CUDA context init cost

F_P = ctypes.POINTER(ctypes.c_float)
INT = ctypes.c_int

CONV_ARGTYPES = [F_P, INT, INT, INT, F_P, INT, INT, INT, INT, INT, F_P, INT, INT]
POOL_ARGTYPES = [F_P, INT, INT, INT, INT, INT, INT, F_P, INT, INT]
FC_ARGTYPES = [F_P, INT, F_P, INT, F_P, F_P]


def compile_all():
    libs = {}

    def build_c(name, src, extra_flags=None):
        so = os.path.join(BUILD_DIR, f"lib{name}.so")
        cmd = ["gcc", "-shared", "-fPIC", "-O2"] + (extra_flags or []) + \
              ["-I", KERNELS_DIR, os.path.join(KERNELS_DIR, src), "-o", so]
        subprocess.run(cmd, check=True)
        return so

    def build_cuda(name, src):
        so = os.path.join(BUILD_DIR, f"lib{name}.so")
        cmd = ["nvcc", "-shared", "-Xcompiler", "-fPIC", "-O2", "-ccbin", "gcc",
               "-I", KERNELS_DIR, os.path.join(KERNELS_DIR, src), "-o", so]
        subprocess.run(cmd, check=True)
        return so

    print("Compiling all kernels...")
    libs["conv_seq"] = build_c("conv_seq", "conv_seq.c")
    libs["conv_omp"] = build_c("conv_omp", "conv_omp.c", ["-fopenmp"])
    libs["conv_cuda"] = build_cuda("conv_cuda", "conv_cuda.cu")

    libs["pool_seq"] = build_c("pool_seq", "pool_seq.c")
    libs["pool_omp"] = build_c("pool_omp", "pool_omp.c", ["-fopenmp"])
    libs["pool_cuda"] = build_cuda("pool_cuda", "pool_cuda.cu")

    libs["fc_seq"] = build_c("fc_seq", "fc_seq.c")
    libs["fc_omp"] = build_c("fc_omp", "fc_omp.c", ["-fopenmp"])
    libs["fc_cuda"] = build_cuda("fc_cuda", "fc_cuda.cu")

    return libs


def get_func(so_path, func_name, argtypes):
    lib = ctypes.CDLL(so_path)
    func = getattr(lib, func_name)
    func.argtypes = argtypes
    func.restype = None
    return func


def time_call(call_fn, num_runs=NUM_RUNS, warmup=WARMUP_RUNS):
    for _ in range(warmup):
        call_fn()
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        call_fn()
        end = time.perf_counter()
        times.append(end - start)
    return min(times), sum(times) / len(times)


def benchmark_conv(libs, results):
    print("\n=== Benchmarking Convolution ===")
    conv_seq = get_func(libs["conv_seq"], "conv2d_seq", CONV_ARGTYPES)
    conv_omp = get_func(libs["conv_omp"], "conv2d_omp", CONV_ARGTYPES)
    conv_cuda = get_func(libs["conv_cuda"], "conv2d_cuda", CONV_ARGTYPES)

    np.random.seed(42)
    configs = [
        {"in_channels": 3,  "in_h": 32,  "in_w": 32,  "out_channels": 16},
        {"in_channels": 16, "in_h": 64,  "in_w": 64,  "out_channels": 32},
        {"in_channels": 32, "in_h": 128, "in_w": 128, "out_channels": 64},
        {"in_channels": 64, "in_h": 224, "in_w": 224, "out_channels": 64},
    ]
    kernel_h = kernel_w = 3
    stride, padding = 1, 1

    for cfg in configs:
        ic, ih, iw, oc = cfg["in_channels"], cfg["in_h"], cfg["in_w"], cfg["out_channels"]
        oh = (ih + 2 * padding - kernel_h) // stride + 1
        ow = (iw + 2 * padding - kernel_w) // stride + 1

        inp = np.random.randn(ic * ih * iw).astype(np.float32)
        wts = np.random.randn(oc * ic * kernel_h * kernel_w).astype(np.float32)
        out = np.zeros(oc * oh * ow, dtype=np.float32)

        def call(func=None):
            func(inp.ctypes.data_as(F_P), ic, ih, iw,
                 wts.ctypes.data_as(F_P), oc, kernel_h, kernel_w,
                 stride, padding,
                 out.ctypes.data_as(F_P), oh, ow)

        print(f"\nInput({ic}x{ih}x{iw}) -> Output({oc}x{oh}x{ow})")

        seq_best, seq_avg = time_call(lambda: call(conv_seq))
        print(f"  Sequential: best={seq_best*1000:.3f}ms")

        omp_best, omp_avg = time_call(lambda: call(conv_omp))
        print(f"  OpenMP:     best={omp_best*1000:.3f}ms  speedup={seq_best/omp_best:.2f}x")

        cuda_best, cuda_avg = time_call(lambda: call(conv_cuda))
        print(f"  CUDA:       best={cuda_best*1000:.3f}ms  speedup={seq_best/cuda_best:.2f}x")

        results.append({
            "operation": "conv", "config": f"{ic}x{ih}x{iw}->{oc}x{oh}x{ow}",
            "seq_ms": seq_best * 1000, "omp_ms": omp_best * 1000, "cuda_ms": cuda_best * 1000,
            "omp_speedup": seq_best / omp_best, "cuda_speedup": seq_best / cuda_best,
        })


def benchmark_pool(libs, results):
    print("\n=== Benchmarking Max Pooling ===")
    pool_seq = get_func(libs["pool_seq"], "maxpool2d_seq", POOL_ARGTYPES)
    pool_omp = get_func(libs["pool_omp"], "maxpool2d_omp", POOL_ARGTYPES)
    pool_cuda = get_func(libs["pool_cuda"], "maxpool2d_cuda", POOL_ARGTYPES)

    np.random.seed(43)
    configs = [
        {"channels": 16, "in_h": 64,  "in_w": 64},
        {"channels": 32, "in_h": 128, "in_w": 128},
        {"channels": 64, "in_h": 224, "in_w": 224},
    ]
    pool_h = pool_w = stride = 2

    for cfg in configs:
        c, ih, iw = cfg["channels"], cfg["in_h"], cfg["in_w"]
        oh = (ih - pool_h) // stride + 1
        ow = (iw - pool_w) // stride + 1

        inp = np.random.randn(c * ih * iw).astype(np.float32)
        out = np.zeros(c * oh * ow, dtype=np.float32)

        def call(func=None):
            func(inp.ctypes.data_as(F_P), c, ih, iw,
                 pool_h, pool_w, stride,
                 out.ctypes.data_as(F_P), oh, ow)

        print(f"\nInput({c}x{ih}x{iw}) -> Output({c}x{oh}x{ow})")

        seq_best, _ = time_call(lambda: call(pool_seq))
        print(f"  Sequential: best={seq_best*1000:.3f}ms")

        omp_best, _ = time_call(lambda: call(pool_omp))
        print(f"  OpenMP:     best={omp_best*1000:.3f}ms  speedup={seq_best/omp_best:.2f}x")

        cuda_best, _ = time_call(lambda: call(pool_cuda))
        print(f"  CUDA:       best={cuda_best*1000:.3f}ms  speedup={seq_best/cuda_best:.2f}x")

        results.append({
            "operation": "pool", "config": f"{c}x{ih}x{iw}->{c}x{oh}x{ow}",
            "seq_ms": seq_best * 1000, "omp_ms": omp_best * 1000, "cuda_ms": cuda_best * 1000,
            "omp_speedup": seq_best / omp_best, "cuda_speedup": seq_best / cuda_best,
        })


def benchmark_fc(libs, results):
    print("\n=== Benchmarking Fully Connected ===")
    fc_seq = get_func(libs["fc_seq"], "fc_seq", FC_ARGTYPES)
    fc_omp = get_func(libs["fc_omp"], "fc_omp", FC_ARGTYPES)
    fc_cuda = get_func(libs["fc_cuda"], "fc_cuda", FC_ARGTYPES)

    np.random.seed(44)
    configs = [
        {"in_features": 4096, "out_features": 1000},
        {"in_features": 8192, "out_features": 4096},
        {"in_features": 16384, "out_features": 4096},
    ]

    for cfg in configs:
        inf, outf = cfg["in_features"], cfg["out_features"]

        inp = np.random.randn(inf).astype(np.float32)
        wts = np.random.randn(outf * inf).astype(np.float32)
        bias = np.random.randn(outf).astype(np.float32)
        out = np.zeros(outf, dtype=np.float32)

        def call(func=None):
            func(inp.ctypes.data_as(F_P), inf,
                 wts.ctypes.data_as(F_P), outf,
                 bias.ctypes.data_as(F_P),
                 out.ctypes.data_as(F_P))

        print(f"\nInput({inf}) -> Output({outf})")

        seq_best, _ = time_call(lambda: call(fc_seq))
        print(f"  Sequential: best={seq_best*1000:.3f}ms")

        omp_best, _ = time_call(lambda: call(fc_omp))
        print(f"  OpenMP:     best={omp_best*1000:.3f}ms  speedup={seq_best/omp_best:.2f}x")

        cuda_best, _ = time_call(lambda: call(fc_cuda))
        print(f"  CUDA:       best={cuda_best*1000:.3f}ms  speedup={seq_best/cuda_best:.2f}x")

        results.append({
            "operation": "fc", "config": f"{inf}->{outf}",
            "seq_ms": seq_best * 1000, "omp_ms": omp_best * 1000, "cuda_ms": cuda_best * 1000,
            "omp_speedup": seq_best / omp_best, "cuda_speedup": seq_best / cuda_best,
        })


def main():
    libs = compile_all()
    results = []

    benchmark_conv(libs, results)
    benchmark_pool(libs, results)
    benchmark_fc(libs, results)

    csv_path = os.path.join(RESULTS_DIR, "benchmark_data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nAll results saved to {csv_path}")


if __name__ == "__main__":
    main()