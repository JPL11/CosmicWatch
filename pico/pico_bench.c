/* On-device verify + benchmark of the 49-parameter classifier for the
 * Pico 2 W (RP2350, Cortex-M33 FPU); also builds for RP2040 Picos and
 * for the RP2350's RISC-V cores (no FPU -> softfloat comparison).
 *
 * On USB connect it prints ONE JSON line ({"pico_bench": ...}) with the
 * max |C - numpy| over the 32 reference vectors, mean ns and cycles per
 * event, and min/max single-call cycles; read_pico.py saves it. It then
 * alternates 10 s ACTIVE (scoring continuously, MARK_GPIO high) and
 * 10 s IDLE (sleeping, MARK_GPIO low) forever, printing each phase
 * change, so a USB power meter reads active-minus-idle power and
 *   energy/event = (P_active - P_idle) / events_per_s.
 * The CYW43 radio is never initialised, so it draws no Wi-Fi power. */
#include <math.h>
#include <stdio.h>
#include "pico/stdlib.h"
#include "pico/version.h"
#include "hardware/clocks.h"
#if !defined(__riscv)
#include "hardware/structs/systick.h"
#endif
#include "mlp49_weights.h"       /* N_IN */
#include "mlp49_test_vectors.h"

#define MARK_GPIO 15          /* scope / logic-analyzer marker pin */
#define BENCH_ITERS 200000
#define PHASE_MS 10000
#define DETECTOR_HZ 1.38      /* measured CosmicWatch event rate */

float mlp49_score(const float *raw);

static volatile float sink;

#if !defined(__riscv)
/* SysTick counts processor cycles down from a 24-bit reload. */
static void cyc_init(void) {
    systick_hw->rvr = 0xFFFFFF;
    systick_hw->cvr = 0;
    systick_hw->csr = 0x5;    /* enable, clock = processor clock */
}
static inline uint32_t cyc_now(void) { return systick_hw->cvr; }
static inline uint32_t cyc_elapsed(uint32_t a, uint32_t b) {
    return (a - b) & 0xFFFFFF;
}
#endif

static const char *chip(void) {
#if PICO_RP2350
    return "rp2350";
#else
    return "rp2040";
#endif
}

static const char *arch(void) {
#if defined(__riscv)
    return "riscv-hazard3";
#elif PICO_RP2350
    return "cortex-m33";
#else
    return "cortex-m0plus";
#endif
}

int main(void) {
    stdio_init_all();
    gpio_init(MARK_GPIO);
    gpio_set_dir(MARK_GPIO, GPIO_OUT);
    while (!stdio_usb_connected())
        sleep_ms(100);
    sleep_ms(500);            /* let the host-side reader settle */

    /* 1. correctness against the numpy reference outputs */
    float maxdiff = 0.0f;
    for (int t = 0; t < N_TEST; t++) {
        float d = fabsf(mlp49_score(&TEST_X[t * N_IN]) - TEST_Y[t]);
        if (d > maxdiff) maxdiff = d;
    }

    /* 2. mean cost per event over many calls */
    uint32_t hz = clock_get_hz(clk_sys);
    gpio_put(MARK_GPIO, 1);
    uint64_t t0 = time_us_64();
    for (int i = 0; i < BENCH_ITERS; i++)
        sink += mlp49_score(&TEST_X[(i & (N_TEST - 1)) * N_IN]);
    uint64_t us = time_us_64() - t0;
    gpio_put(MARK_GPIO, 0);
    double ns_event = (double)us * 1e3 / BENCH_ITERS;
    double cyc_event = ns_event * 1e-9 * hz;

    /* 3. single-call spread (input-dependent: ReLU branches, expf) */
    long cmin = -1, cmax = -1;
#if !defined(__riscv)
    cyc_init();
    uint32_t a = cyc_now(), b = cyc_now();
    uint32_t overhead = cyc_elapsed(a, b);
    for (int t = 0; t < N_TEST; t++) {
        a = cyc_now();
        sink += mlp49_score(&TEST_X[t * N_IN]);
        b = cyc_now();
        long c = (long)cyc_elapsed(a, b) - (long)overhead;
        if (cmin < 0 || c < cmin) cmin = c;
        if (c > cmax) cmax = c;
    }
#endif

    printf("{\"pico_bench\": {\"chip\": \"%s\", \"arch\": \"%s\", "
           "\"sdk\": \"%s\", \"clk_sys_hz\": %lu, "
           "\"n_test\": %d, \"max_abs_diff_vs_numpy\": %.3e, "
           "\"bench_iters\": %d, \"ns_per_event\": %.1f, "
           "\"cycles_per_event\": %.0f, "
           "\"single_call_cycles_min\": %ld, \"single_call_cycles_max\": %ld, "
           "\"duty_pct_at_detector_rate\": %.6f, \"verify_pass\": %s}}\n",
           chip(), arch(), PICO_SDK_VERSION_STRING, (unsigned long)hz,
           N_TEST, (double)maxdiff, BENCH_ITERS, ns_event, cyc_event,
           cmin, cmax, DETECTOR_HZ * ns_event * 1e-9 * 100.0,
           maxdiff < 1e-5f ? "true" : "false");

    /* 4. power phases for the inline USB meter */
    for (int cycle = 0;; cycle++) {
        uint64_t n = 0, end = time_us_64() + PHASE_MS * 1000ull;
        printf("PHASE active cycle=%d t_ms=%llu\n", cycle,
               (unsigned long long)(time_us_64() / 1000));
        gpio_put(MARK_GPIO, 1);
        while (time_us_64() < end) {
            for (int i = 0; i < 1000; i++)
                sink += mlp49_score(&TEST_X[(i & (N_TEST - 1)) * N_IN]);
            n += 1000;
        }
        gpio_put(MARK_GPIO, 0);
        printf("PHASE idle cycle=%d t_ms=%llu events_per_s=%.0f\n", cycle,
               (unsigned long long)(time_us_64() / 1000),
               n * 1000.0 / PHASE_MS);
        sleep_ms(PHASE_MS);
    }
}
