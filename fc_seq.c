#include <stdio.h>
#include <stdlib.h>
#include "common.h"

// input:   [in_features]                flat array
// weights: [out_features][in_features]  flat array (row-major)
// bias:    [out_features]                flat array
// output:  [out_features]                flat array
// Computes: output[o] = bias[o] + sum_i( input[i] * weights[o][i] )
void fc_seq(
    const float* input, int in_features,
    const float* weights, int out_features,
    const float* bias,
    float* output)
{
    for (int o = 0; o < out_features; o++) {
        float sum = bias[o];
        for (int i = 0; i < in_features; i++) {
            sum += input[i] * weights[o * in_features + i];
        }
        output[o] = sum;
    }
}

// Standalone test driver
#ifdef TEST_FC_SEQ
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

    fc_seq(input, in_features, weights, out_features, bias, output);

    printf("Output (%d):\n", out_features);
    for (int i = 0; i < out_features; i++) {
        printf("%.2f ", output[i]);
    }
    printf("\n");

    free(output);
    return 0;
}
#endif