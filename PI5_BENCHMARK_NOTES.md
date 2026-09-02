# Raspberry Pi 5 benchmark (2026-09-02) — WP1 of the NAIRR-period plan

Run: pi_benchmark.py (origin/main copy), committed model_weights.json
(49-param torch-free classifier), python 3.13.5 / numpy 2.2.4, 30 s
sustained-load protocol matching the Pi 4 / Jetson baselines.
Host: jl@192.168.1.72, Pi 5 Model B Rev 1.1, 16 GB.

## Five-device matrix (same weights, same harness)

| device | us/ev numpy | us/ev pure-py | sustained ev/s | uJ/inf |
|---|---|---|---|---|
| Pi 5            | 23.69 | 8.62  | 13,585,340 | (needs USB meter) |
| Jetson Orin 15W | 38.35 | 15.28 | 12,252,638 | 0.074 |
| Jetson Orin 7W  | 60.65 | 24.25 |  7,715,362 | 0.052 |
| Pi 4            | 75.90 | 29.88 |  6,799,224 | (needs USB meter) |
| Pi 400          | 78.20 | 29.26 |  6,248,448 | (needs USB meter) |

Headlines: the Pi 5 is the fastest device in the matrix on every latency
metric — 1.6x the Jetson-15W and 3.2x the Pi 4 on numpy per-event, and the
only board to beat the Jetson's sustained throughput. Headroom vs the
2.4 Hz detector event rate: ~9.2 million x. Pure-python (MCU proxy) path:
8.62 us/event.

Missing for the full table: Pi-row energy columns need USB power-meter
idle/load watt readings (rerun with --idle-watts/--load-watts; the
harness computes energy per inference from them). Jetson rows used
onboard INA sensors.

Result JSON: pi_benchmark_raspberry_pi_5.json (this dir, untracked; the
canonical copy should be committed to the GitHub repo, which is 39
commits ahead of this stale local clone).

## Benchmarks 2 + 3 (same session)

### Event gateway (200k-event replay + 1.3757 Hz realistic run)
| policy | Pi4 us/ev | Pi5 us/ev | speedup |
|---|---|---|---|
| coincidence | 17.98 | 3.97 | 4.5x |
| adc | 26.12 | 6.76 | 3.9x |
| mlp | 74.05 | 19.26 | 3.8x |
| hybrid | 75.90 | 20.07 | 3.8x |

Realistic-rate hybrid: CPU duty 0.016% (Pi4: 0.041%); headroom vs detector
rate 36,215x (Pi4: 9,577x). Caveat: the original Pi4 synthetic-stream
generator was never saved (unreproducible); this session's driver
(cw_gateway_bench.py, this dir) IS saved — selection fractions differ
somewhat (hybrid .66 vs .52) which only adds output work, making Pi5
timings slightly pessimistic. Use the saved driver for all future devices.

### FL client workload (client 0, 6 rounds, 1 local epoch, 4 threads)
| device | s/round | img/s | max RSS MiB |
|---|---|---|---|
| Pi 5 (torch 2.14 cpu) | 0.234 | 10,274 | 326 |
| Jetson 15W cpu | 0.438 | 5,486 | 757 |
| Jetson 15W cuda | 0.481 | 4,990 | 1,098 |
| Pi 4 (torch 2.13 cpu) | 0.900 | 2,665 | 323 |

Pi 5 beats the Jetson WITH CUDA by 2.1x on this workload: the 26.6k-param
model is too small for GPU dispatch to pay off. Final-MSE column differs
across devices (torch versions/seeds); hardware numbers are the claim.

### Session summary
Pi 5 tops all three harnesses (inference triage, event gateway, FL
client). Remaining for the complete matrix: Pi-row energy columns
(USB meter; recommend FNIRSI FNB58 or AVHzY CT-3, note Pi 5's 5A PD
negotiation may fall back to 3A through a meter — harmless at these
loads). All Pi 5 result JSONs + the new gateway driver are in this dir,
untracked; commit to the GitHub repo (this clone is a stale snapshot).
