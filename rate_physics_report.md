# CosmicWatch Single-Node Physics Analysis

Timestamped events over `2025-11-01T00:00:00Z` → `2026-03-01T00:00:00Z`; 133 active hours.

## 1. Event rate

- Mean hourly rate: **1.2157 Hz** (min 0.0003, max 1.5889).
- Busiest hour: 2025-11-09T06:00:00.000Z at 1.5889 Hz.

## 2. Poisson check (inter-arrival times)

- Clean window: 120,000 events, mean inter-arrival 1247.172 ms → implied rate 0.8018 Hz.
- Coefficient of variation: **0.749** (CV≈1.0 for a Poisson (random) process).
- Verdict: **sub-Poissonian / more regular than random — consistent with detector dead-time**.

## 3. Environmental correlation

- Rate vs pressure: Pearson r = **0.399** over 131 hours; barometric coefficient ≈ **2.709%/hPa**.
- Rate vs temperature: Pearson r = **0.229** over 131 hours.

  (Known physics: muon flux anti-correlates with pressure ~ −0.1 to −0.3%/hPa. A clean indoor single detector over a short span may show a weak/noisy effect.)

## 4. Findings

- Single-detector rate averages ~1.2157 Hz; usable for a clean cosmic-ray rate measurement.
- Inter-arrival CV = 0.749 → sub-Poissonian / more regular than random — consistent with detector dead-time; this is a real statistical-physics result from one node, no labels needed.
- Rate shows a POSITIVE rate–pressure correlation (r=0.399, +2.709%/hPa), which is the OPPOSITE sign of the canonical barometric effect (negative). Over only ~16 active days this is most likely a confound (seasonal/temperature drift, indoor sensor coupling, rate trending with deployment), NOT a clean barometric measurement — flag for investigation, do not over-claim.
- All of this is achievable NOW on the single-node data and strengthens a project / workshop writeup without needing multi-node data.
