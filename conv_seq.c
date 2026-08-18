#include <stdio.h>
#include <stdlib.h>
#include "common.h"

// input:   [in_channels][in_h][in_w]        flat array, size = in_channels*in_h*in_w
// weights: [out_channels][in_channels][kernel_h][kernel_w]  flat array
// output:  [out_channels][out_h][out_w]      flat array, size = out_channels*out_h*out_w
void conv2d_seq(
    const float* input, int in_channels, int in_h, int in_w,
    const float* weights, int out_channels, int kernel_h, int kernel_w,
    int stride, int padding,
    float* output, int out_h, int out_w)
{
    for (int oc = 0; oc < out_channels; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {

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
                            // else: implicit zero-padding, contributes 0
                        }
                    }
                }

                int output_idx = oc * (out_h * out_w) + oh * out_w + ow;
                output[output_idx] = sum;
            }
        }
    }
}

// ---- Standalone test driver ----
// Compiles and runs this file alone to sanity-check correctness on tiny fixed input
// before we wire it into OpenMP/CUDA versions or the Python pipeline.
#ifdef TEST_CONV_SEQ
int main() {
    // Tiny fixed example: 1 input channel, 5x5 image, 1 output channel, 3x3 kernel, stride 1, no padding
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

    conv2d_seq(input, in_channels, in_h, in_w,
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