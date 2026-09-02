# Microcontroller port of the 49-parameter event classifier

`mcu_classifier.c` is a dependency-free C99 implementation of the deployed
classifier (standardize -> 4x8 ReLU -> 8x1 sigmoid), with weights baked in
from the committed `model_weights.json` via `make_mcu_port.py`. Host
verification: max |C - numpy| = 6e-08 over 32 test vectors (float32
rounding, i.e. exact); 10 ns/event on an x86-class host at -O2.

## Why
The benchmark brief's own conclusion: the Pi-class boards' idle power, not
compute, dominates the deployment budget, so genuinely low-power operation
needs a microcontroller-class device. This port makes that a
flash-and-measure exercise instead of a project.

## Feasibility estimates (pending hardware; cycle-count based)
Per event the forward pass costs ~50 float ops plus one expf.

| target | FP hardware | est. cycles/event | est. us/event @ clock | duty @ 1.38 Hz |
|---|---|---|---|---|
| RP2040 (Pico, M0+) | softfloat | ~8-15k | ~60-110 @ 133 MHz | ~0.01% |
| RP2350 (Pico 2, M33) | FPU | ~0.3-1k | ~1.5-4 @ 150 MHz | ~0.0004% |

Even the $4 RP2040 in software floating point clears the detector rate by
four orders of magnitude; a fixed-point path exists if ever needed but the
headroom makes it unnecessary. Next step when a Pico is on hand: drop
`mlp49_score()` into a Pico SDK project, toggle a GPIO around the call,
and read cycles with a logic analyzer or `time_us_64()`; measure sleep vs
active current for the true energy-per-event the Linux boards cannot reach.

## Files
- `make_mcu_port.py` — regenerates `mlp49_weights.h` + `mlp49_test_vectors.h`
- `mcu_classifier.c` — the implementation + host verify/bench main()
- build: `cc -O2 -o mcu_classifier mcu_classifier.c -lm && ./mcu_classifier`
