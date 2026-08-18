#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

// CUDA kernel: one thread computes one output element (c, oh, ow)
__global__ void maxpool2d_cuda_kernel(
    const float* input, int channels, int in_h, int in_w,
    int pool_h, int pool_w, int stride,
    float* output, int out_h, int out_w)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c  = blockIdx.z;

    if (ow >= out_w || oh >= out_h || c >= channels) return;

    float max_val = -1e30f;

    for (int ph = 0; ph < pool_h; ph++) {
        for (int pw = 0; pw < pool_w; pw++) {

            int ih = oh * stride + ph;
            int iw = ow * stride + pw;

            if (ih < in_h && iw < in_w) {
                int input_idx = c * (in_h * in_w) + ih * in_w + iw;
                if (input[input_idx] > max_val) {
                    max_val = input[input_idx];
                }
            }
        }
    }

    int output_idx = c * (out_h * out_w) + oh * out_w + ow;
    output[output_idx] = max_val;
}

// Host wrapper: handles memory allocation, transfer, kernel launch, and cleanup
extern "C" void maxpool2d_cuda(
    const float* h_input, int channels, int in_h, int in_w,
    int pool_h, int pool_w, int stride,
    float* h_output, int out_h, int out_w)
{
    size_t input_size = channels * in_h * in_w * sizeof(float);
    size_t output_size = channels * out_h * out_w * sizeof(float);

    float *d_input, *d_output;

    cudaMalloc((void**)&d_input, input_size);
    cudaMalloc((void**)&d_output, output_size);

    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (out_w + blockDim.x - 1) / blockDim.x,
        (out_h + blockDim.y - 1) / blockDim.y,
        channels
    );

    maxpool2d_cuda_kernel<<<gridDim, blockDim>>>(
        d_input, channels, in_h, in_w,
        pool_h, pool_w, stride,
        d_output, out_h, out_w
    );

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

// Standalone test driver
#ifdef TEST_POOL_CUDA
int main() {
    int channels = 1, in_h = 4, in_w = 4;
    int pool_h = 2, pool_w = 2, stride = 2;

    int out_h = (in_h - pool_h) / stride + 1;
    int out_w = (in_w - pool_w) / stride + 1;

    float input[16] = {
        1, 3, 2, 4,
        5, 6, 7, 8,
        9, 2, 1, 0,
        4, 3, 6, 5
    };

    float* output = (float*)malloc(out_h * out_w * sizeof(float));

    maxpool2d_cuda(input, channels, in_h, in_w,
                   pool_h, pool_w, stride,
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