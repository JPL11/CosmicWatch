# Physics Interpretation — CosmicWatch Single-Node Data

Companion to `data_analysis_report.md`, `edge_efficiency_report.md`, and `rate_physics_report.md`.
Grounds the numbers in detector and particle physics, and states honestly where quantum mechanics
does and does not enter.

## What the detector actually measures
A CosmicWatch unit is a plastic **scintillator** read out by a **SiPM** (silicon photomultiplier).
A charged particle — at sea level, almost always a **cosmic-ray muon** — crosses the scintillator and
deposits energy by ionization (Bethe–Bloch). The scintillator converts that to photons; the SiPM
converts photons to an electrical pulse; the pulse height is digitized as **`adc_value`** (and
`sipm_mv`). So **ADC ∝ energy deposited** by the particle.

## Mapping our measurements to physics
- **ADC spectrum (peak ≈180–210, long tail to the 4095 saturation):** the classic **Landau
  distribution** of energy loss for minimum-ionizing particles in a thin absorber. This is textbook
  muon-through-scintillator behaviour — a sanity check that the data is real cosmic rays.
- **Coincident events have higher ADC (p50 334 vs 183):** a CosmicWatch unit is **two stacked sensor
  devices** (a muon telescope); requiring *both* to fire selects genuine through-going particles (real
  tracks, more light) and rejects single-sensor noise. **Energy deposition is the physical
  discriminator** — which is exactly why an ADC threshold already captures most of the signal and a
  fancier model cannot beat it. The ML "ceiling" is physics, not a modeling failure.
- **Mean rate ≈1.2 Hz:** consistent with the expected sea-level muon flux (~1 muon/cm²/min) for a small
  scintillator. The detector is seeing real cosmic rays at a plausible rate.
- **Inter-arrival CV ≈ 1.0 (Poisson):** cosmic-ray arrivals are intrinsically a **Poisson (memoryless,
  random)** process. Using the microsecond `pico_timestamp_s` field, CV = **0.99** — consistent with
  pure Poisson. (An earlier CV = 0.75 was an artifact of the 1-second `timestamp_ms` quantization, not
  dead-time; dead-time is negligible at ~0.07%.)
- **Rate–pressure correlation positive (+2.7%/hPa):** the *opposite* sign of the known **barometric
  muon effect** (≈ −0.1 to −0.3%/hPa). Over only ~16 active days this is almost certainly a confound
  (seasonal/temperature drift, indoor coupling, deployment trend) — **flagged, not claimed**.

## Where quantum mechanics genuinely enters
The *physics being measured* is quantum/particle physics, even though the *analysis is classical*:
- **Muon production & decay:** primaries → pions → muons via the weak interaction; muon decay (2.2 µs)
  is a quantum-probabilistic process; arrivals are quantum-random (hence Poisson).
- **Special relativity:** muons reach the ground only because of relativistic **time dilation** —
  without it they would decay miles up. (Our data does not measure lifetime, so we cannot *demonstrate*
  this here, but it is why there are muons to detect.)
- **Quantum-limited detection:** the SiPM detects **single photons** (photoelectric absorption +
  Geiger-mode avalanche); scintillation is molecular excitation/de-excitation. The `coincident` cut is
  essentially **rejecting quantum dark-count noise** in the SiPM.

## Honest scope (what this is NOT)
- The methods (MLP, SNN, FL, GNN, quantization) are **classical** signal processing. There is **no
  quantum computing and no quantum machine learning** here.
- The data (ADC + coincidence flag) is too coarse to study quantum-detection properties (photon
  statistics, quantum efficiency) directly.
- A **single unit** (two stacked sensor devices, co-located — a telescope, not a spatial array) can do
  rate, energy-spectrum, and Poisson/dead-time physics. It **cannot** do network/ensemble physics
  (air-shower reconstruction, Cosmic-Ray Ensembles, directional reconstruction) — that requires multiple
  spatially-separated *units*, time-synchronized.
- **Usable data is ~3.36M events** (both ingest schemas: parsed `timestamp` + raw `wall_time`), not
  582k — but it is all from this one unit, so the network conclusion is unchanged.

## One-paragraph summary
The data behaves exactly as cosmic-ray physics predicts: a Landau-shaped energy-deposit (ADC) spectrum,
a ~1.2 Hz muon rate, and Poisson (CV ≈ 1.0) arrival statistics, from a two-sensor SiPM scintillator
telescope whose detection physics is genuinely quantum-limited. Because energy deposition *is* the
discriminating variable, a tuned ADC threshold already captures most of the coincidence signal — so the
ML contribution is **edge efficiency** (a tiny, quantizable model), not higher accuracy. All of this is
real single-node science; the network-scale (GNN/ensemble) physics is impossible with one sensor and is
the concrete reason multi-node data is required.
