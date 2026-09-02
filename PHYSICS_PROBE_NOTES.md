# Atmospheric / space-weather probes on the frozen archive (2026-09-02)

Question: with ingestion frozen (last doc 2026-06-19), is there physics
left to extract? Two probes attempted against public external data.
Time series: hourly counts + avg pressure/temperature via chunked ES
aggregations (cw_atmospheric_pull.py; no bulk export needed). Active
live-time: 324 near-full hours (~13.5 days) in two segments
(2026-05-17; 2026-06-06 -> 06-19), split by the known 06-11 change point.

## Probe B: barometric coefficient — instrumentation-limited, no detection
- Naive whole-segment regression gives beta_p = -3.5 %/hPa: an ARTIFACT.
  The 06-11 instrument step (rate -0.49 Hz, ADC +30) rides on a pressure
  trend and fakes a 20x-too-large slope. Cautionary result worth keeping.
- Corrected (split at the change point, device temperature as covariate):
  post-change segment beta_p = -0.28 +/- 0.30 %/hPa — CONSISTENT with the
  muon literature (-0.1 to -0.2 %/hPa) but not a detection. Pre-change
  segment (+1.6 +/- 0.6) remains drift-confounded over its 4.3 hPa span.
- First-difference estimator: -0.6 +/- 1.1 %/hPa. The physics is out of
  reach at this live time: hourly Poisson noise ~1.1-1.5% vs a
  ~0.08%/hour pressure signal. A real measurement needs months of
  STABLE running — exactly what the new sensors would provide.

## Probe A: Forbush decreases — no event in coverage; sensitivity stated
Oulu neutron monitor (NMDB, public) for 2026-06-04..20: quiet period,
max excursion -1.7% on 06-06 (below the >=3% FD threshold). No FD
occurred during the archive's active windows, so the search resolves to
a sensitivity statement: at 2.4 Hz, daily-binned Poisson sensitivity is
~0.23%/day, but instrument systematics (temp coupling ~1-2%/C, discrete
steps) set the practical floor at ~1-2%; a typical 3-10% FD WOULD have
been visible had one occurred, a moderate one would not.

## Verdict on "focus on physics more?"
The frozen archive is instrumentation-limited, not analysis-limited:
every remaining atmospheric/space-weather measurement needs longer,
stabler live time than exists. That is a WP2 argument, not a gap —
"months of stable running from the new sensors enable beta_p and FD
measurements the archive cannot support" belongs in the full-Pilot
justification. Keep: the -3.5 %/hPa artifact anatomy (a nice
methods cautionary note) and the sensitivity numbers for the report.
