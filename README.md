# Cosmic Ray Sensor Network Notes

Working repo for the CREDO / CosmicWatch Elasticsearch data: data auditing and export, single-node
muon physics, edge-AI prototyping (tiny MLP/SNN + quantization), image clustering, and simulation-only
network prototypes (GNN / federated learning). See `CosmicWatch_Report.pdf` (regenerate with
`make_report.py`) for the consolidated writeup, and `DATA_README.md` / `DATA_DICTIONARY.md` for the
data exports.

## Setup

1. Copy `.env.example` to `.env` and fill in `CREDO_USER` / `CREDO_PASS` (kept out of git).
2. Create the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Scripts

Data / export:
- `data_analysis.py` — field-level profile of all 72 fields across the full index.
- `data_readiness.py`, `cosmicwatch_summary.py`, `multi_node_probe.py` — coverage, summaries, multi-node check.
- `export_data.py` — stream the full index to CSV with a version-separating `partition` column.
- `make_data_dictionary.py` — regenerate `DATA_DICTIONARY.md` / `data_dictionary.csv`.
- `credo_loader.py` — canonical loader across both CosmicWatch schemas (time = `timestamp` else `wall_time`).

Physics (single node, label-free):
- `rate_physics.py`, `adc_physics.py` — rate, Poisson timing, Landau/Moyal ADC fit, dead-time.
- `energy_calibration.py` — MIP-peak ADC→MeV calibration.
- `unsupervised_physics.py` — coincidence as a physics cut: efficiency turn-on curve, drift, anomaly enrichment.
- `time_domain_physics.py` — diurnal cycle, pressure-effect resolution, tilt from recorded accel, GFZ Kp space-weather check, muon-lifetime feasibility (negative).

Machine learning:
- `edge_ai_experiment.py` — tiny MLP + toy SNN vs a tuned ADC threshold (weak-label supervised).
- `edge_efficiency.py` — quantization sweep (32/8/4/2/1-bit) + latency.
- `event_ml.py` — feature study, self-supervised probe, anomaly detection.
- `legacy_images.py` — decode + cluster the 69k CREDO image hit-crops (fully unsupervised).
- `combined_check.py` — conclusions re-validated on the combined ~3.36M events.
- `fl_simulation.py`, `gnn_simulation.py` — **simulation-only** federated-learning and GNN prototypes.
- `federated_legacy.py` — self-supervised FedAvg on real legacy image device IDs, with centralized and
  local-only comparisons.

Extended validation and operations:
- `edge_reduction.py` — chronological transmission-policy simulation: retained coincidence events vs bytes/day.
- `detector_health.py` — daily robust outliers, regime-change alerts, and ingestion-staleness monitoring.
- `legacy_timing.py` — deduplicated cross-device timing search with a device/day time-shift null.
- `legacy_common.py`, `cosmicwatch_common.py` — canonical readers for the local CSV export.
- `extended_analysis_report.md` — consolidated results and the measurements that still require hardware.

Hardware deployment:
- `event_gateway.py` — blocking UART/stdin event gateway with ADC, coincidence, MLP, or hybrid selection.
- `prepare_fl_hardware_data.py` — generate the compact real-device FL dataset for transfer to a target.
- `fl_hardware_benchmark.py` — benchmark one real FL client's local workload on CPU or CUDA, with power hooks.
- `EVENT_DRIVEN_HARDWARE.md` — Pi 4 and Jetson Orin Nano deployment and measurement procedure.

Reporting:
- `make_report.py` — regenerate the consolidated PDF report from the result JSONs.
- `make_onepager.py` — one-page summary PDF.

## Edge deployment & benchmarking (Raspberry Pi 4 / Pi 400, Jetson Orin Nano)

The trained classifier exports to `model_weights.json` (49 parameters) with **torch-free** inference —
`pi_benchmark.py` needs only Python 3 + numpy on the target. Nothing else from this repo is required
on-device.

### One-time, on the dev machine (needs PyTorch + ES access)

```bash
python3 pi_benchmark.py --train        # trains on real events, writes model_weights.json
scp pi_benchmark.py model_weights.json <user>@<device>:~/
```

(`model_weights.json` is committed, so you can skip `--train` and just copy the repo's copy.)

### Raspberry Pi 4 and Pi 400

The Pi 400 is a Pi 4 in a keyboard (same SoC, 1.8 GHz vs 1.5 GHz) — identical steps for both.
Use 64-bit Raspberry Pi OS.

```bash
sudo apt install -y python3-numpy          # or: pip3 install numpy
python3 pi_benchmark.py                    # latency + throughput + CPU temp/freq/throttle state
```

**Power (needs a USB-C inline power meter):** read the meter with the Pi idle, then while the
benchmark runs, and pass both in:

```bash
python3 pi_benchmark.py --idle-watts 2.7 --load-watts 3.4
# -> pi_benchmark.json gains energy_per_inference_uJ (total and active)
```

The script also records `vcgencmd` volts/temp/throttling automatically on Pi hardware.

### Jetson Orin Nano

No USB meter needed — the Orin has **onboard INA3221 power rails**. JetPack's Ubuntu already has
Python; install numpy if missing (`sudo apt install python3-numpy`).

```bash
# 1) check / set the power mode (affects both speed and watts; benchmark both to bracket the envelope)
sudo nvpmodel -q                # current mode
sudo nvpmodel -m 0              # MAXN (or -m 1 for the 7W mode)

# 2) in a second terminal, watch the board power rail while the benchmark runs
tegrastats --interval 500       # note VDD_IN mW at idle, then under load
#   (or: sudo pip3 install jetson-stats && jtop  -> the POWER tab)

# 3) run the benchmark, passing the two VDD_IN readings (in watts)
python3 pi_benchmark.py --idle-watts 4.8 --load-watts 6.1
```

The 49-parameter model doesn't need the GPU — the value of the Orin number is a **flight-computer-class**
benchmark (Orin-class modules fly on smallsats), and its onboard telemetry gives a clean
energy-per-inference figure.

### Recording results

`pi_benchmark.py` writes `pi_benchmark.json` per run. Suggested comparison table:

| Platform | µs/event (numpy) | µs/event (pure py) | sustained events/s | idle W | load W | µJ/inference |
|---|--:|--:|--:|--:|--:|--:|
| x86_64 dev (baseline) | ~5–7 | ~2.4 | ~105M | — | — | — |
| Raspberry Pi 4 (re-measured 2026-07-18, clean supply)¹ | 75.9 | 29.9 | 6.80M | | | |
| Raspberry Pi 400 (measured 2026-07-01) | 78.2 | 29.3 | 6.25M | | | |
| Jetson Orin Nano 7W (measured 2026-07-17)² | 60.7 | 24.3 | 7.72M | 3.58 | 3.98 | 0.52 (0.052 active) |
| Jetson Orin Nano 15W/MAXN (measured 2026-07-17)² | 38.4 | 15.3 | 12.25M | 3.74 | 4.64 | 0.38 (0.074 active) |

¹ The original 2026-07-01 Pi 4 run was under-voltage throttling (`0x50005`, core sagged to 0.85 V —
root cause: powered from a desktop USB-C data port). Re-run 2026-07-18 on a proper supply
(`throttled=0x0` before and after, core 0.916 V): single-event latency was barely affected
(77.0→75.9 µs) but **sustained throughput rose 29%** (5.27M→6.80M ev/s) — under-voltage capped
sustained clocks, not single-shot latency. The clean Pi 4 out-sustains the Pi 400.

² Jetson power is the board-input **VDD_IN rail read from the onboard INA3221** (sysfs, 5 Hz sampling,
median over 25 s idle / 35 s mid-load) — no USB meter. This dev kit's JetPack exposes two modes:
7W (mode 1, 4 cores) and 15W (mode 0, 6 cores, the "MAXN" of the original Orin Nano). µJ/inference is
total load power over sustained throughput; "active" uses (load − idle) W, i.e. the marginal cost with
the desktop session's idle draw excluded. Results: `pi_benchmark_jetson_orin_nano_7w.json` /
`pi_benchmark_jetson_orin_nano_15w.json`.

Two measured cross-platform findings: (a) even a Pi has ~3×10⁶ headroom over the detector rate — at
1.4–2.8 Hz the classifier is ~0.01% CPU duty cycle, so the *board's idle power* is the whole budget;
(b) pure-Python single-event inference beats numpy ~2.6× on every platform (per-call overhead dominates
a 49-parameter model), so on-device single-event paths should skip numpy entirely.

For context: the detector event rate is ~1.4–2.8 Hz, so every platform above has ≥10⁶× headroom —
the scientifically interesting number is **energy per inference** (the flight power budget), not speed.

### Idle-power experiment (Jetson, measured 2026-07-18)

Because the classifier is ~10⁻⁶ duty cycle at real rates, board **idle** power is the whole budget —
so we measured how much of the Orin Nano's idle is software-reducible (`jetson_idle_power_experiment.json`).
Answer: essentially none. Desktop vs headless is a null result (3.70 vs 3.66 W, rails identical —
the GPU rail-gates either way); 7W mode saves only ~0.16 W. The ~3.7 W floor is VDD_SOC (1.17 W) +
CPU/GPU/CV (0.48 W) + ~2 W of dev-kit carrier overhead (regulators, PHYs, fan) that software can't
touch. Idle also wanders 3.1–3.7 W across sessions, so single idle samples carry ±0.3 W uncertainty.
Implication: low-power deployment is a *hardware* decision — custom carrier, duty-cycling/deep sleep,
or an MCU (the pure-Python path exists precisely for that) — not a software-config one.
