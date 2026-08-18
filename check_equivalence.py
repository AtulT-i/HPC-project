import ctypes
import numpy as np
import subprocess
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNELS_DIR = os.path.join(PROJECT_ROOT, "kernels")
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")
os.makedirs(BUILD_DIR, exist_ok=True)

TOLERANCE = 1e-3


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


F_P = ctypes.POINTER(ctypes.c_float)
INT = ctypes.c_int

CONV_ARGTYPES = [F_P, INT, INT, INT, F_P, INT, INT, INT, INT, INT, F_P, INT, INT]
POOL_ARGTYPES = [F_P, INT, INT, INT, INT, INT, INT, F_P, INT, INT]
FC_ARGTYPES = [F_P, INT, F_P, INT, F_P, F_P]


def validate_conv(libs):
    print("\n--- Validating Convolution ---")
    conv_seq = get_func(libs["conv_seq"], "conv2d_seq", CONV_ARGTYPES)
    conv_omp = get_func(libs["conv_omp"], "conv2d_omp", CONV_ARGTYPES)
    conv_cuda = get_func(libs["conv_cuda"], "conv2d_cuda", CONV_ARGTYPES)

    np.random.seed(42)
    in_channels, in_h, in_w = 3, 32, 32
    out_channels, kernel_h, kernel_w = 16, 3, 3
    stride, padding = 1, 1
    out_h = (in_h + 2 * padding - kernel_h) // stride + 1
    out_w = (in_w + 2 * padding - kernel_w) // stride + 1

    inp = np.random.randn(in_channels * in_h * in_w).astype(np.float32)
    wts = np.random.randn(out_channels * in_channels * kernel_h * kernel_w).astype(np.float32)

    def run(func):
        out = np.zeros(out_channels * out_h * out_w, dtype=np.float32)
        func(inp.ctypes.data_as(F_P), in_channels, in_h, in_w,
             wts.ctypes.data_as(F_P), out_channels, kernel_h, kernel_w,
             stride, padding,
             out.ctypes.data_as(F_P), out_h, out_w)
        return out

    out_seq, out_omp, out_cuda = run(conv_seq), run(conv_omp), run(conv_cuda)

    diff_omp = np.max(np.abs(out_seq - out_omp))
    diff_cuda = np.max(np.abs(out_seq - out_cuda))

    print(f"  seq vs omp:  {diff_omp:.8f}  {'PASS' if diff_omp < TOLERANCE else 'FAIL'}")
    print(f"  seq vs cuda: {diff_cuda:.8f}  {'PASS' if diff_cuda < TOLERANCE else 'FAIL'}")

    return diff_omp < TOLERANCE and diff_cuda < TOLERANCE


def validate_pool(libs):
    print("\n--- Validating Max Pooling ---")
    pool_seq = get_func(libs["pool_seq"], "maxpool2d_seq", POOL_ARGTYPES)
    pool_omp = get_func(libs["pool_omp"], "maxpool2d_omp", POOL_ARGTYPES)
    pool_cuda = get_func(libs["pool_cuda"], "maxpool2d_cuda", POOL_ARGTYPES)

    np.random.seed(43)
    channels, in_h, in_w = 16, 64, 64
    pool_h, pool_w, stride = 2, 2, 2
    out_h = (in_h - pool_h) // stride + 1
    out_w = (in_w - pool_w) // stride + 1

    inp = np.random.randn(channels * in_h * in_w).astype(np.float32)

    def run(func):
        out = np.zeros(channels * out_h * out_w, dtype=np.float32)
        func(inp.ctypes.data_as(F_P), channels, in_h, in_w,
             pool_h, pool_w, stride,
             out.ctypes.data_as(F_P), out_h, out_w)
        return out

    out_seq, out_omp, out_cuda = run(pool_seq), run(pool_omp), run(pool_cuda)

    diff_omp = np.max(np.abs(out_seq - out_omp))
    diff_cuda = np.max(np.abs(out_seq - out_cuda))

    print(f"  seq vs omp:  {diff_omp:.8f}  {'PASS' if diff_omp < TOLERANCE else 'FAIL'}")
    print(f"  seq vs cuda: {diff_cuda:.8f}  {'PASS' if diff_cuda < TOLERANCE else 'FAIL'}")

    return diff_omp < TOLERANCE and diff_cuda < TOLERANCE


def validate_fc(libs):
    print("\n--- Validating Fully Connected ---")
    fc_seq = get_func(libs["fc_seq"], "fc_seq", FC_ARGTYPES)
    fc_omp = get_func(libs["fc_omp"], "fc_omp", FC_ARGTYPES)
    fc_cuda = get_func(libs["fc_cuda"], "fc_cuda", FC_ARGTYPES)

    np.random.seed(44)
    in_features, out_features = 4096, 1000

    inp = np.random.randn(in_features).astype(np.float32)
    wts = np.random.randn(out_features * in_features).astype(np.float32)
    bias = np.random.randn(out_features).astype(np.float32)

    def run(func):
        out = np.zeros(out_features, dtype=np.float32)
        func(inp.ctypes.data_as(F_P), in_features,
             wts.ctypes.data_as(F_P), out_features,
             bias.ctypes.data_as(F_P),
             out.ctypes.data_as(F_P))
        return out

    out_seq, out_omp, out_cuda = run(fc_seq), run(fc_omp), run(fc_cuda)

    diff_omp = np.max(np.abs(out_seq - out_omp))
    diff_cuda = np.max(np.abs(out_seq - out_cuda))

    print(f"  seq vs omp:  {diff_omp:.8f}  {'PASS' if diff_omp < TOLERANCE else 'FAIL'}")
    print(f"  seq vs cuda: {diff_cuda:.8f}  {'PASS' if diff_cuda < TOLERANCE else 'FAIL'}")

    return diff_omp < TOLERANCE and diff_cuda < TOLERANCE


def main():
    libs = compile_all()

    conv_ok = validate_conv(libs)
    pool_ok = validate_pool(libs)
    fc_ok = validate_fc(libs)

    print("\n=== Summary ===")
    print(f"Convolution: {'PASS' if conv_ok else 'FAIL'}")
    print(f"Max Pooling: {'PASS' if pool_ok else 'FAIL'}")
    print(f"Fully Connected: {'PASS' if fc_ok else 'FAIL'}")

    if conv_ok and pool_ok and fc_ok:
        print("\nAll kernels validated successfully.")
    else:
        print("\nOne or more kernels failed validation - review before benchmarking.")


if __name__ == "__main__":
    main()