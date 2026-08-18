#include <stdio.h>
#include <stdlib.h>
#include "common.h"

// input:  [channels][in_h][in_w]  flat array
// output: [channels][out_h][out_w] flat array
// Max pooling: for each output position, take the max value in the corresponding input window
void maxpool2d_seq(
    const float* input, int channels, int in_h, int in_w,
    int pool_h, int pool_w, int stride,
    float* output, int out_h, int out_w)
{
    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {

                float max_val = -1e30f;  // effectively negative infinity for float comparisons

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
        }
    }
}

// Standalone test driver
#ifdef TEST_POOL_SEQ
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

    maxpool2d_seq(input, channels, in_h, in_w,
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