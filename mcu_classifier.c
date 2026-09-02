/* 49-parameter CosmicWatch event classifier, dependency-free C99.
 * Standardize -> 4x8 ReLU -> 8x1 sigmoid, matching pi_benchmark.py's
 * forward_np exactly. Builds anywhere (host, Pico SDK, Arduino core):
 *   host verify+bench:  cc -O2 -o mcu_classifier mcu_classifier.c -lm
 * On an MCU, call mlp49_score(raw_features) per event. */
#include <math.h>
#include <stdio.h>
#include <time.h>
#include "mlp49_weights.h"
#include "mlp49_test_vectors.h"

float mlp49_score(const float *raw) {
    float x[N_IN], h;
    float o = MLP_B2[0];
    for (int i = 0; i < N_IN; i++)
        x[i] = (raw[i] - STD_MU[i]) / STD_SD[i];
    for (int j = 0; j < N_HID; j++) {
        h = MLP_B1[j];
        for (int i = 0; i < N_IN; i++)
            h += x[i] * MLP_W1[i * N_HID + j];
        if (h > 0.0f)
            o += h * MLP_W2[j];
    }
    return 1.0f / (1.0f + expf(-o));
}

int main(void) {
    float maxdiff = 0.0f;
    for (int t = 0; t < N_TEST; t++) {
        float y = mlp49_score(&TEST_X[t * N_IN]);
        float d = fabsf(y - TEST_Y[t]);
        if (d > maxdiff) maxdiff = d;
    }
    printf("verification: max |C - numpy| over %d vectors = %.3e\n",
           N_TEST, (double)maxdiff);

    const long iters = 10 * 1000 * 1000;
    volatile float sink = 0.0f;
    clock_t t0 = clock();
    for (long i = 0; i < iters; i++)
        sink += mlp49_score(&TEST_X[(i & 31) * N_IN]);
    double s = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("host: %.1f ns/event (%ld iters, sink=%.3f)\n",
           s / iters * 1e9, iters, (double)sink);
    return maxdiff < 1e-5f ? 0 : 1;
}
