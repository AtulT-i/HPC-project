#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

// CUDA kernel: one thread computes one output element (oc, oh, ow)
__global__ void conv2d_cuda_kernel(
    const float* input, int in_channels, int in_h, int in_w,
    const float* weights, int out_channels, int kernel_h, int kernel_w,
    int stride, int padding,
    float* output, int out_h, int out_w)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;

    if (ow >= out_w || oh >= out_h || oc >= out_channels) return;

    float sum = 0.0f;

    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {

                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw;

                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    int input_idx = ic * (in_h * in_w) + ih * in_w + iw;
                    int weight_idx = oc * (in_channels * kernel_h * kernel_w)
                                    + ic * (kernel_h * kernel_w)
                                    + kh * kernel_w + kw;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    int output_idx = oc * (out_h * out_w) + oh * out_w + ow;
    output[output_idx] = sum;
}

// Host wrapper: handles memory allocation, transfer, kernel launch, and cleanup
extern "C" void conv2d_cuda(
    const float* h_input, int in_channels, int in_h, int in_w,
    const float* h_weights, int out_channels, int kernel_h, int kernel_w,
    int stride, int padding,
    float* h_output, int out_h, int out_w)
{
    size_t input_size = in_channels * in_h * in_w * sizeof(float);
    size_t weights_size = out_channels * in_channels * kernel_h * kernel_w * sizeof(float);
    size_t output_size = out_channels * out_h * out_w * sizeof(float);

    float *d_input, *d_weights, *d_output;

    cudaMalloc((void**)&d_input, input_size);
    cudaMalloc((void**)&d_weights, weights_size);
    cudaMalloc((void**)&d_output, output_size);

    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, weights_size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (out_w + blockDim.x - 1) / blockDim.x,
        (out_h + blockDim.y - 1) / blockDim.y,
        out_channels
    );

    conv2d_cuda_kernel<<<gridDim, blockDim>>>(
        d_input, in_channels, in_h, in_w,
        d_weights, out_channels, kernel_h, kernel_w,
        stride, padding,
        d_output, out_h, out_w
    );

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
}

// Standalone test driver
#ifdef TEST_CONV_CUDA
int main() {
    int in_channels = 1, in_h = 5, in_w = 5;
    int out_channels = 1, kernel_h = 3, kernel_w = 3;
    int stride = 1, padding = 0;

    int out_h = compute_output_dim(in_h, kernel_h, stride, padding);
    int out_w = compute_output_dim(in_w, kernel_w, stride, padding);

    float input[25] = {
        1,2,3,4,5,
        6,7,8,9,10,
        11,12,13,14,15,
        16,17,18,19,20,
        21,22,23,24,25
    };

    float weights[9] = {
        1,0,-1,
        1,0,-1,
        1,0,-1
    };

    float* output = (float*)malloc(out_h * out_w * sizeof(float));

    conv2d_cuda(input, in_channels, in_h, in_w,
                weights, out_channels, kernel_h, kernel_w,
                stride, padding,
                output, out_h, out_w);

    printf("Output (%d x %d):\n", out_h, out_w);
    for (int i = 0; i < out_h; i++) {
        for (int j = 0; j < out_w; j++) {
            printf("%.1f ", output[i * out_w + j]);
        }
        printf("\n");
    }

    free(output);
    return 0;
}
#endif