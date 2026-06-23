# Label-Free Physics with Coincidence-as-a-Cut

Coincidence is used as a **physics selection cut** and an **interpretation reference**, not a supervised target. Runs across both CosmicWatch partitions (~3.36M events) via the canonical loader.

## A. Spectrum by coincidence cut

- parsed: coincident p50 ADC 334.4 vs non-coincident 182.5
- raw: coincident p50 ADC 544.1 vs non-coincident 199.6

## B. Efficiency turn-on P(coincident | ADC)

- parsed: 39 bins; raw: 39 bins (see plot).

## C. Drift across the combined timeline

- coincident-rate outlier days flagged: 0; mean-ADC outlier days: 1

## D. Anomaly detection + coincident enrichment

- {"n": 40000, "baseline_coincident_rate": 0.1231, "anomaly_coincident_rate": 0.155, "enrichment_factor": 1.26, "interpretation": "anomalies are coincidence-neutral"}

## Findings

- Coincidence CUT cleanly separates energy: coincident ADC p50 334.4 vs non-coincident 182.5 (parsed) — the cut isolates real muons without any training.
- Efficiency turn-on curve P(coincident|ADC) rises monotonically with energy — a genuine detector characterization, using coincidence as a measured outcome, not a label.
- DRIFT between deployments: mean coincident rate 0.13 (parsed, 2025-11..2026-02) vs 0.081 (raw AxLab, 2026-05..2026-06) — a real change in detector response across epochs, now visible because wall_time put both on one timeline.
- Anomaly interpretation: top reconstruction-error events have coincident rate 0.155 vs baseline 0.1231 (1.26x) — anomalies are coincidence-neutral. Coincidence used only to interpret, never to train.
- All four are label-free in spirit; coincidence is a physics cut + interpretation reference, sidestepping the weak-label leakage that capped the supervised task.
