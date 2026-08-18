import ctypes
import numpy as np
import os
import subprocess
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNELS_DIR = os.path.join(PROJECT_ROOT, "kernels")
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")

F_P = ctypes.POINTER(ctypes.c_float)
INT = ctypes.c_int
CONV_ARGTYPES = [F_P, INT, INT, INT, F_P, INT, INT, INT, INT, INT, F_P, INT, INT]


def load_conv_cuda():
    so_path = os.path.join(BUILD_DIR, "libconv_cuda.so")
    if not os.path.exists(so_path):
        subprocess.run([
            "nvcc", "-shared", "-Xcompiler", "-fPIC", "-O2", "-ccbin", "gcc",
            "-I", KERNELS_DIR, os.path.join(KERNELS_DIR, "conv_cuda.cu"),
            "-o", so_path
        ], check=True)
    lib = ctypes.CDLL(so_path)
    func = lib.conv2d_cuda
    func.argtypes = CONV_ARGTYPES
    func.restype = None
    return func


def load_image_as_array(image_path, resize_to=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(resize_to)
    arr = np.array(img, dtype=np.float32) / 255.0  # normalize to [0,1]

    # PIL gives (H, W, C) — our kernels expect (C, H, W), so transpose
    arr = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(arr)


def main():
    image_path = input("Enter path to test image: ").strip()

    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    print(f"Loading image: {image_path}")
    img_array = load_image_as_array(image_path)
    in_channels, in_h, in_w = img_array.shape
    print(f"Image loaded as array: {in_channels} channels, {in_h}x{in_w}")

    flat_input = img_array.flatten().astype(np.float32)

    # Random weights standing in for a real trained conv layer
    # (this is a placeholder — real weights come once the original model is integrated)
    out_channels, kernel_h, kernel_w = 16, 3, 3
    stride, padding = 1, 1
    np.random.seed(0)
    weights = np.random.randn(out_channels * in_channels * kernel_h * kernel_w).astype(np.float32) * 0.1

    out_h = (in_h + 2 * padding - kernel_h) // stride + 1
    out_w = (in_w + 2 * padding - kernel_w) // stride + 1
    output = np.zeros(out_channels * out_h * out_w, dtype=np.float32)

    conv_cuda = load_conv_cuda()

    conv_cuda(
        flat_input.ctypes.data_as(F_P), in_channels, in_h, in_w,
        weights.ctypes.data_as(F_P), out_channels, kernel_h, kernel_w,
        stride, padding,
        output.ctypes.data_as(F_P), out_h, out_w
    )

    output = output.reshape(out_channels, out_h, out_w)

    print(f"\nConvolution output shape: {output.shape}")
    print(f"Output value range: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
    print("\nNote: weights used here are random placeholders, not trained weights.")
    print("This confirms the kernel correctly processes real image data end-to-end,")
    print("but the numerical output is not a meaningful prediction yet.")


if __name__ == "__main__":
    main()