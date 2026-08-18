#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

// CUDA kernel: one thread computes one output neuron
__global__ void fc_cuda_kernel(
    const float* input, int in_features,
    const float* weights, int out_features,
    const float* bias,
    float* output)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;

    if (o >= out_features) return;

    float sum = bias[o];
    for (int i = 0; i < in_features; i++) {
        sum += input[i] * weights[o * in_features + i];
    }
    output[o] = sum;
}

// Host wrapper: handles memory allocation, transfer, kernel launch, and cleanup
extern "C" void fc_cuda(
    const float* h_input, int in_features,
    const float* h_weights, int out_features,
    const float* h_bias,
    float* h_output)
{
    size_t input_size = in_features * sizeof(float);
    size_t weights_size = out_features * in_features * sizeof(float);
    size_t bias_size = out_features * sizeof(float);
    size_t output_size = out_features * sizeof(float);

    float *d_input, *d_weights, *d_bias, *d_output;

    cudaMalloc((void**)&d_input, input_size);
    cudaMalloc((void**)&d_weights, weights_size);
    cudaMalloc((void**)&d_bias, bias_size);
    cudaMalloc((void**)&d_output, output_size);

    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, weights_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, bias_size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (out_features + blockSize - 1) / blockSize;

    fc_cuda_kernel<<<gridSize, blockSize>>>(
        d_input, in_features,
        d_weights, out_features,
        d_bias,
        d_output
    );

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);
}

// Standalone test driver
#ifdef TEST_FC_CUDA
int main() {
    int in_features = 4, out_features = 3;

    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    float weights[12] = {
        1, 0, 0, 1,
        0, 1, 0, 1,
        1, 1, 1, 1
    };

    float bias[3] = {0.5f, -1.0f, 2.0f};

    float* output = (float*)malloc(out_features * sizeof(float));

    fc_cuda(input, in_features, weights, out_features, bias, output);

    printf("Output (%d):\n", out_features);
    for (int i = 0; i < out_features; i++) {
        printf("%.2f ", output[i]);
    }
    printf("\n");

    free(output);
    return 0;
}
#endif