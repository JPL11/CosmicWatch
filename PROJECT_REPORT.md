# CosmicWatch Sensor Network — Project Report

*Last updated 2026-09-02. One-file narrative of what was done, what was
found, and what is blocked on hardware. Every claim points at a
committed artifact.*

## The story in one paragraph

We took a frozen ~3.36M-event CosmicWatch/CREDO archive, audited it to
the field level, and extracted everything the data can support: the
single-node muon physics checks pass, but the archive is
**instrumentation-limited** — the interesting geophysics signals
(barometric coefficient, space-weather response) are confounded by
detector regime changes, so new science needs new sensors. That result
pivots the project's contribution to the **systems side**: a
49-parameter deployed classifier with a full quantization story, an
event-gateway selection stack, a five-device edge benchmark matrix
topped by the Raspberry Pi 5, a dependency-free C port that clears the
detector rate on microcontroller-class softfloat, and — the capstone —
**a real federated-learning deployment over the LAN across physically
distinct machines**, with measured straggler economics that a
simulation cannot give you.

## 1. Data (frozen archive)

- ~3.36M events, 72 fields, two schema generations, plus 69k image
  hit-crops. Field-level profile: `data_analysis.py` /
  `data_analysis_report.md`; canonical loaders `credo_loader.py`,
  `legacy_common.py`; exports documented in `DATA_README.md` and
  `DATA_DICTIONARY.md`.
- No new ingestion: the Elasticsearch index is stale and the
  collaboration is building new sensors. Everything below is honest
  about being bounded by this archive.

## 2. Physics: what the archive can and cannot say

- **Can:** Poisson event timing, Landau/Moyal ADC shape, MIP-peak
  energy calibration, coincidence as a physics cut, diurnal cycle
  (`rate_physics.py`, `adc_physics.py`, `energy_calibration.py`,
  `unsupervised_physics.py`, `time_domain_physics.py`).
- **Cannot:** the barometric coefficient comes out at the right order
  (−0.1 to −0.2 %/hPa) but is contaminated by a detector regime
  change, and Forbush-decrease cross-checks against NMDB neutron
  monitors are inconclusive at this exposure
  (`cw_atmospheric_pull.py`, `PHYSICS_PROBE_NOTES.md`).
- **Weather probe** (`cw_weather_probe.py`): with no geo field in the
  detector docs, the site was located *empirically* — the onboard
  barometer tracks coastal-LA reanalysis weather at r = 0.992
  (near-sea-level, Long Beach grid; negatively correlated with the
  CREDO Poland sites). An independent fit against external pressure
  gives β_p = −0.19 ± 0.20 %/hPa — dead on the literature value but
  SNR ~1: same instrumentation-limited verdict from a second
  instrument. Temperature/humidity coefficients consistent with zero;
  only 4 rainy hours in the stable segment (no precipitation power).
- **Conclusion:** the archive is instrumentation-limited. This is the
  documented argument for the new-sensor work packages in
  `proposal/` — and it is why the rest of the project is systems.

## 3. Edge ML on the events

- 49-parameter MLP classifier (4→8→1) beats a tuned ADC threshold;
  quantization survives to int8 and degrades gracefully below
  (`edge_ai_experiment.py`, `edge_efficiency.py`).
- Event gateway with four selection policies (coincidence / adc / mlp
  / hybrid) as the on-device triage layer (`event_gateway.py`,
  `cw_gateway_bench.py`).
- Torch-free deployment: `model_weights.json` + numpy-only
  `pi_benchmark.py`; pure-Python single-event inference beats numpy
  ~2.6× on every board (per-call overhead dominates a 49-param model).

## 4. Hardware fleet benchmarks

Five-device matrix (Pi 4, Pi 400, Pi 5, Jetson Orin Nano 7W/15W)
across three harnesses — inference triage, event gateway, FL client.
Details: `PI5_BENCHMARK_NOTES.md`, `hardware_benchmark_brief.pdf`.

- **Pi 5 is the fastest board on all three harnesses** (23.7 µs/event,
  12.6M ev/s batched; 3.0× Pi 4, 1.4× Jetson 15W; 1.6× the Jetson's
  own CUDA path on FL rounds — GPU dispatch does not pay at this size).
- Headroom over the real 1.4–2.8 Hz detector rate is ≥10⁶× on every
  board, so **idle power is the whole flight budget**; the Jetson
  idle-power experiment shows that floor is a hardware property
  (~3.7 W, software-irreducible), pointing at MCU-class deployment.
- `mcu_classifier.c` (C99, verified to 6e-08 vs numpy) clears the
  detector rate by ~4 orders of magnitude even on RP2040 softfloat
  (`MCU_PORT.md`). On-device measurement pending a Pico.
- Operational lesson worth keeping: the original Pi 4 numbers were
  silently degraded by USB under-voltage (`0x50005`); a clean supply
  raised sustained throughput 29%. Always check `vcgencmd get_throttled`.

## 5. Federation: from simulation to real machines

- Simulated: `fl_simulation.py`, `federated_legacy.py` (FedAvg on real
  per-device image data), heterogeneity quantified with 1-Wasserstein
  distances (`fl_heterogeneity.py`).
- **Real (`fed_real/`):** synchronous FedAvg over TCP sockets between
  physical hosts on the LAN, 26.6k-param autoencoder, 8 logical
  clients, per-round wall/arrival/straggler/wire accounting.
  - *Config A, heterogeneous* (desktop + Pi 5 + Jetson): MSE
    0.226→0.078 in 6 rounds; 0.19 s/round steady state; the desktop
    idles ~90% of every round waiting on the ARM boards.
  - *Config B, all-Pi network* (Pi 5 server + Pi 4): MSE 0.228→0.119;
    0.34 s/round; straggler gap only ~0.12 s (worst-case idle ~35%).
  - *Config C, asynchronous* (FedAsync-style staleness-discounted
    mixing, both topologies): on the heterogeneous fleet the desktop
    goes from ~90% idle to contributing 865 of 988 updates, and at
    equal wall time async reaches 2.4x lower MSE than sync (0.032 vs
    0.078 at 3.2 s); on the balanced all-Pi network the gain shrinks
    to ~2x, because async's win scales with the idle fraction it
    reclaims. Staleness up to 145 caused no divergence.
  - *Config D, wire compression under async* (fp32/fp16/int8 on both
    directions): fp16 is free — 47% fewer bytes, slightly better final
    MSE, and ~7% more updates/s since smaller payloads serialize
    faster; int8 cuts bytes 71% and matches everyone early, but
    plateaus ~25% above fp32 near convergence — per-mix quantization
    noise sets a floor extra iterations cannot buy back. Error
    feedback (int8ef) recovers ~half that floor; uplink
    delta-transmission with EF recovers ~75%; delta-encoding BOTH
    directions (int8_delta2_ef, per-connection view tracking) closes
    the gap completely — tail MSE statistically identical to fp32 at
    a 73% byte cut. The floor was fully attributed (half uplink, half
    downlink) and fully recovered by the same mechanism.
  - **Findings:** (a) a synchronous round is priced by the slowest
    host, so a fast machine buys idle time, not speed — the
    homogeneous cheap fleet is the efficient shape for sync, and
    async aggregation is what actually monetizes a fast host; (b) at
    106 kB/update the network — even plain Wi-Fi — is invisible next
    to compute under sync (3.9 MB/run), but async converts reclaimed
    idle into bandwidth (215 MB in 12 s), so update compression,
    pointless under sync at this scale, becomes the relevant lever
    exactly when you go async.

## 6. Outreach and next steps

- **Microsoft Discovery:** a CosmicWatch domain agent + starter kit is
  drafted and committed (`discovery-contrib/`), pending the
  collaborator's preference on opening the upstream PR.
- **Open items, each blocked on a specific piece of hardware:**
  energy columns for the Pi rows (inline USB-C meter), Pico
  measurement of the C port, wired-vs-Wi-Fi timing rerun (unmanaged
  switch), and — the science unlock — new sensor data.
- **Natural growth direction:** federate a larger scientific model
  (e.g. a neural-operator surrogate, building on the public
  Caltech/NVIDIA work, cited) on the same fed_real harness, where
  MB-scale updates finally make the communication axis interesting.
