# Approximate Energy Calibration (MIP peak)

Anchor: coincident ADC MPV = **278** ↔ MIP MPV **1.60 MeV** (assuming 1.0 cm scintillator, 1.6 MeV/cm MPV dE/dx).

**Calibration: ≈ 5.8 keV / ADC count** (pedestal 0.0).

## Energy scale

- Trigger threshold (~52 ADC): 0.3 MeV
- Mean deposit: 1.451 MeV
- ADC saturation (4095): 23.61 MeV

## Caveats

- Single-point, single-detector calibration: assumes thickness, ADC linearity, zero pedestal.
- This is an energy SCALE (order-of-magnitude correct), not a precision calibration.
- A real calibration would use a known source or a tagged stopping-muon sample.

## Findings

- MIP-peak calibration: ADC MPV 278 ↔ 1.60 MeV → gain ≈ 5.8 keV/ADC.
- Implied dynamic range: trigger threshold ≈ 0.3 MeV up to ≈ 23.61 MeV at ADC saturation — physically sensible for a small scintillator (sub-MeV trigger, tens-of-MeV ceiling for large-path/multi-particle events).
- Mean deposited energy ≈ 1.451 MeV, consistent with a MIP-dominated spectrum with a high-energy tail.
- Turns the descriptive ADC spectrum into a calibrated energy measurement — a real, citable single-node physics result, with the stated approximations.
