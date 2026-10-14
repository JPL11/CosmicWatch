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
headroom makes it unnecessary.

## On-device run (Pico 2 W, `pico/`, added 2026-10-14)
`pico/pico_bench.c` links the unmodified `mlp49_score()` (built with
`-DMLP49_NO_MAIN`) and, on USB connect, prints one JSON line: max
|C - numpy| over the 32 reference vectors, mean ns and cycles per event
(200k calls, `time_us_64`), and min/max single-call cycles (SysTick, Arm
only). It then loops 10 s ACTIVE (continuous scoring, GPIO 15 high) /
10 s IDLE (sleep, GPIO 15 low), printing each phase, so the inline USB
meter gives energy/event = (P_active - P_idle) / events_per_s. The
CYW43 radio is never initialised (the Pico 2 W's LED hangs off it, hence
GPIO 15 as the marker).

    # once: Arm toolchain + SDK + prebuilt picotool. The Homebrew cask
    # gcc-arm-embedded 15.3.rel2 had a stale checksum on 2026-10-14 (Arm
    # re-signed the pkg; its .sha256asc matched the download), so the
    # toolchain was taken straight from Arm, verified, and unpacked:
    #   pkgutil --expand-full arm-gnu-toolchain-15.3.rel2-darwin-arm64-arm-none-eabi.pkg x
    #   mv x/Payload ~/arm-gnu-toolchain-15.3.rel2
    git clone --depth 1 -b 2.1.1 https://github.com/raspberrypi/pico-sdk ~/pico-sdk
    git -C ~/pico-sdk submodule update --init --depth 1
    brew install picotool          # avoids the SDK building it with the host cc
    export PATH=~/arm-gnu-toolchain-15.3.rel2/bin:$PATH PICO_SDK_PATH=~/pico-sdk
    # build -> pico/build/pico_bench.uf2 (31 kB text; FPU code: vfma/vdiv.f32)
    cmake -S pico -B pico/build -Dpicotool_DIR=$(brew --prefix picotool)/lib/cmake/picotool
    cmake --build pico/build -j
    # hold BOOTSEL, plug in, copy the .uf2 to the RP2350 drive, then:
    python pico/read_pico.py --out pico_benchmark_rp2350.json --phases 4

Same source, other targets: `-DPICO_BOARD=pico` (RP2040, softfloat M0+)
or `-DPICO_PLATFORM=rp2350-riscv` (Hazard3 cores, no FPU: a softfloat
row on the same chip; needs a RISC-V toolchain). Write each to its own
`pico_benchmark_<target>.json` so the table above can be replaced with
measured rows.

## Files
- `make_mcu_port.py` — regenerates `mlp49_weights.h` + `mlp49_test_vectors.h`
- `mcu_classifier.c` — the implementation + host verify/bench main()
- `pico/` — Pico SDK project (`pico_bench.c`, `CMakeLists.txt`) +
  `read_pico.py` host capture
- build: `cc -O2 -o mcu_classifier mcu_classifier.c -lm && ./mcu_classifier`
