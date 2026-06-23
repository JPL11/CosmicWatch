# Quantitative Single-Node Physics — ADC, Timing, Pressure

## 1. ADC energy spectrum (Landau / Moyal fit)

- All events: MPV ≈ **131.33 ADC**, width 34.05, R²=0.9117.
- Coincident MPV 277.53 vs non-coincident 123.73 → coincidence selects higher energy deposit.

## 2. Timing & dead-time (pico_timestamp_s)

- Corrected inter-arrival CV = **0.993** (consistent with Poisson (random)); mean 0.7314 s → 1.3672 Hz.
- Dead-time fraction ≈ 0.0007 (small).

## 3. Pressure confound

- Simple r(rate,pressure) = 0.361; standardized partial βs = {'pressure': 0.398, 'temperature': 0.051, 'time_trend': 0.249}.
- pressure retains a partial association after controls — investigate further.

## Findings

- ADC spectrum fits a Landau (Moyal) shape with MPV≈131.33 ADC (R²=0.9117) — the textbook energy-loss distribution of muons in a thin scintillator; confirms real cosmic-ray events.
- Coincident MPV (277.53) sits above non-coincident (123.73) — the coincidence cut selects higher-energy-deposit tracks, exactly why ADC alone is a strong classifier.
- Using pico_timestamp_s (µs) the inter-arrival CV is 0.993 → consistent with Poisson (random); this CORRECTS the earlier 0.75 which was inflated by 1-second timestamp_ms quantization.
- Detector dead-time fraction ≈ 0.0007 (tiny) — so dead-time barely affects the rate; timing structure is dominated by the Poisson process itself.
- Pressure–rate: simple r=0.361, but standardized partial β=0.398 after controlling for temperature+time → pressure retains a partial association after controls — investigate further.
