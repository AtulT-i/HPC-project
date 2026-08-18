#ifndef COMMON_H
#define COMMON_H

// All tensors are stored as flat 1D float arrays in row-major (NCHW-style) order.
// For a single image: index = channel * (height * width) + row * width + col

// Computes output spatial dimension for a convolution/pooling operation
static inline int compute_output_dim(int input_dim, int kernel_dim, int stride, int padding) {
    return ((input_dim + 2 * padding - kernel_dim) / stride) + 1;
}

#endif