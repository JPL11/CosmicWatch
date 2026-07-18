# Event-Driven Pi and Jetson Deployment

## What “event-driven” means here

`event_gateway.py` blocks in the Linux kernel on a UART, pipe, or file descriptor. No model loop runs while
the detector is quiet. One policy evaluation runs when a complete JSON event arrives, and selected events are
written in batches. UART reception normally wakes Linux through an interrupt, so this is a real
event-triggered software pipeline.

It is not neuromorphic execution. The Pi and Jetson remain powered while blocked, and their CPUs/GPUs execute
ordinary instructions when awakened. The existing rate-coded SNN expands each static event into artificial
steps and should not be described as a hardware event-driven energy result on either board.

## Event gateway

The input contract is one JSON object per line with `adc_value` and, when available, `sipm_mv`,
`temperature_c`, `pressure_pa`, and `coincident` or `coincidence_flag`.

```bash
# Blocking UART input. Install pyserial only for this mode.
python3 -m pip install pyserial
python3 event_gateway.py \
  --serial /dev/ttyACM0 --baud 115200 \
  --policy hybrid --adc-threshold 238 --mlp-threshold 0.390279 \
  --batch-size 64 --output selected_events.jsonl

# A producer can also pipe JSONL to blocking stdin.
detector-json-producer | python3 event_gateway.py --input - --policy hybrid
```

The current thresholds came from the chronological reduction experiment. Recalibrate them after detector
gain or threshold changes. If the detector emits its native non-JSON serial format, put a small parser in the
producer; do not treat malformed serial fields as zero-valued events.

Useful measurements are board idle watts while blocked, average watts during a realistic event stream,
selected bytes/day, batch/network overhead, CPU duty fraction, temperature, and dropped serial records.

## Prepare the FL workload

Run once on the development machine with the local export:

```bash
python3 prepare_fl_hardware_data.py --clients 8 --max-per-client 3000
```

This creates `legacy_fl_hardware.npz`, approximately 2.3 MB, containing chronological train/test arrays for
eight real legacy device IDs. Put the following on each target:

```text
fl_hardware_benchmark.py
legacy_fl_hardware.npz
```

The target needs NumPy and PyTorch. The benchmark times one real client's local training workload; it does
not pretend that cheap server-side FedAvg aggregation is the expensive part.

## Raspberry Pi 4

Use the official power supply; the previous Pi 4 result showed undervoltage throttling.

```bash
python3 fl_hardware_benchmark.py \
  --data legacy_fl_hardware.npz --client-index 0 \
  --rounds 6 --local-epochs 1 --threads 4 \
  --idle-watts 2.7 --load-watts 4.1 \
  --out fl_hardware_benchmark_raspberry_pi_4.json
```

Read idle and average load watts from an inline USB-C meter. Report both total joules and active joules above
idle. A Pi cannot enter deep sleep while acting as an always-listening UART gateway; a Pico or external power
controller is needed if deep sleep is a requirement.

## Jetson Orin Nano

Run both the constrained and maximum-power modes. Exact mode IDs vary by JetPack image, so inspect them first.

```bash
sudo nvpmodel -q
tegrastats --interval 500

python3 fl_hardware_benchmark.py \
  --data legacy_fl_hardware.npz --client-index 0 \
  --rounds 6 --local-epochs 1 --device cuda \
  --idle-watts 4.8 --load-watts 7.0 \
  --out fl_hardware_benchmark_jetson_7w.json
```

Use average `VDD_IN` readings from `tegrastats` for the wattage arguments, then repeat in MAXN mode. CUDA may
not win for this 26,584-parameter autoencoder because launch and transfer overhead are significant; that is a
result to measure, not assume.

### Measured results (2026-07-18, 15W mode, torch 2.13.0+cu130 in `~/torchenv`)

`fl_hardware_benchmark_jetson_15w_cpu.json` / `fl_hardware_benchmark_jetson_15w_cuda.json`. Power is the
INA3221 VDD_IN median (5 Hz sysfs sampling): idle 3.18 W (rock-stable across four samples), load measured
mid-run during long sustained probes (1000 CPU / 700 CUDA rounds), because the 6-round workload finishes
too fast to sample. Note this JetPack-7 PyPI wheel excludes CC 8.7 from its binary kernels, so CUDA runs
via PTX JIT — it works, with a first-use JIT cost but no steady-state penalty observed.

| device | 6-round s/round | steady s/round | load W | active W | steady J/round (total) | steady J/round (active) |
|---|--:|--:|--:|--:|--:|--:|
| CPU (6× A78AE) | 0.438 | 0.063 | 5.95 | 2.77 | 0.37 | 0.17 |
| CUDA (Orin iGPU) | 0.481 | 0.110 | 3.89 | 0.71 | 0.43 | 0.08 |

The hypothesis above is confirmed on throughput: in steady state the **CPU is ~1.8× faster per round** than
the GPU for this 26,584-parameter model (launch/transfer overhead dominates; the quick 6-round comparison is
misleading because round 1 carries ~4 s of one-time warmup). But the GPU trains at barely above idle power,
so it wins **~2.2× on active energy** (0.08 vs 0.17 J/round). Either way the absolute numbers are tiny —
one FL round costs less than half a joule — and, as with inference, the deployment budget is the 3.18 W
idle floor, not the compute. The Jetson's per-round time equals the Pi 4's (0.44 vs 0.90 s/round on the
6-round workload, 6× faster in steady state), so a Jetson FL client is justified by GPU-sized models only,
which this autoencoder is not.

## Power measurement on the Pi 4

The Pi 4 exposes no software-readable power sensor. `vcgencmd` reports core voltage and throttle state but
no current, so no on-board procedure yields board watts; the committed Pi 4 results correctly leave
`measurement_required` set. A Pi 5 reports per-rail current via `vcgencmd pmic_read_adc`, and the Jetson's
INA3221 rails are read by `tegrastats` — only the Pi 4 requires external hardware.

Measurement options, cheapest first:

1. Inline USB-C meter between the official PSU and the Pi (the setup this document assumes). A logging
   meter (FNIRSI FNB48, AVHzY CT-3, RD UM25C) is preferred so the run can be averaged instead of read by
   eye. Measures true 5 V board draw.
2. Energy-monitoring smart plug at the wall with a local API (Shelly Plug S, Tasmota). Roughly 1 s
   resolution and ±0.5 W, and it includes PSU conversion loss — the honest number for a 24/7 deployment's
   energy budget. Fully scriptable: poll the API during a run and average.
3. INA260/INA219 breakout spliced into the 5 V line, read over I²C by a second device. Highest fidelity
   and sample rate; requires a pass-through cable.

Procedure once a meter is attached:

- Idle: gateway blocked on input, average watts for 2–3 minutes → `--idle-watts`.
- FL load: run `fl_hardware_benchmark.py` with enough rounds for a stable average (for example
  `--rounds 200`, about 40 s of steady state on the Pi 4 at ~0.16 s/round), then rerun with
  `--idle-watts` / `--load-watts` to fill total and active joules. The first round pays a one-time ~4 s
  torch lazy-initialization cost; use the per-round history to exclude it or amortize it explicitly.
- Gateway under a realistic stream: replay at the measured event rate for several minutes. The measured
  CPU duty fraction at 1.38 Hz is 0.041%, so the expected result is load ≈ idle; confirming that is the
  substance of the always-on-gateway decision criterion below.

If measured idle lands well above ~3 W, disable HDMI, activity LEDs, and unused USB before attributing
the draw to the workload. Published typical Pi 4 figures (idle ~2.7 W, load ~4.1 W) are sanity checks
only — do not substitute them for measurements in result JSONs.

## Decision criteria

- Prefer the Pi as the always-on gateway if it meets reliability and I/O requirements at lower board power.
- Prefer the Jetson only when image training, a realistic larger GNN, or another GPU-sized workload offsets
  its higher idle power.
- Compare FL update bytes with compressed raw images. The current six-round FedAvg experiment transfers 1.95x
  the raw grayscale bytes, so compute speed alone cannot make its communication case.
- Do not benchmark the synthetic GNN as a hardware claim until detector count, graph-window rate, and edge
  construction match a planned deployment.
