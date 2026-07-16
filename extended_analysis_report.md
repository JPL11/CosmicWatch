# Extended Existing-Data Analysis

This pass pursued every identified direction that can be evaluated with the current Elasticsearch index and
the local `credo_useful.csv` export. It does not substitute simulations for unavailable hardware measurements
or expert labels.

## Corrected validation

- The live index remains unchanged at 3,437,063 documents and stopped ingesting on 2026-06-19.
- True cross-deployment evaluation is materially worse than the former mixed-deployment test: MLP F1 is
  0.3228 parsed-to-raw and 0.3349 raw-to-parsed. The prior generalization claim was not supported because
  target-deployment rows entered training.
- The full 2,773,823-row raw export has inter-arrival CV 1.015 at 2.388 Hz after removing gaps over 60 s,
  consistent with Poisson arrivals. Elasticsearch server-side sorting is inappropriate because `wall_time`
  is mapped as a precision-losing float.
- The multi-node audit now distinguishes one-device CosmicWatch from real multi-device partitions. Legacy has
  57 device IDs and 74 multi-device days; this supports real-client learning and exploratory timing, not an
  automatic air-shower claim.

## Legacy images and real-device federation

- Exact deduplication removes 3,006 of 69,000 legacy rows (4.36%), leaving 65,994 unique images.
- The clustering pipeline now uses a deterministic uniform sample over the full deduplicated set, keeps
  locations aligned with image labels, and selects representatives nearest each fitted centroid.
- Eight high-volume real devices were used as FedAvg clients with chronological 80/20 per-device splits.
- Weighted reconstruction MSE: centralized 0.0141, FedAvg 0.0502, local-only mean 0.0569.
- Six FedAvg rounds move an estimated 1.95 times the bytes of the raw grayscale training images. Federated
  image training therefore does not provide the communication-saving result seen with the 49-parameter event
  model. Privacy or decentralized ownership would have to justify it.
- No expert morphology labels exist, so this is a real-client self-supervised representation result, not a
  labeled particle classifier.

## Cross-device timing

After deduplication, 65,994 events from 57 devices were compared with 200 device/day time-shift null runs.

| Window | Observed pairs | Null mean | Observed/null | Upper-tail p |
|---:|--:|--:|--:|--:|
| 10 ms | 7 | 11.25 | 0.622 | 0.547 |
| 100 ms | 56 | 107.89 | 0.519 | 0.667 |
| 1 s | 594 | 1073.15 | 0.554 | 0.582 |

No cross-device timing excess is supported. Only 33 observed sub-second pairs have locations on both events;
their median separation is 279 km. Phone clock synchronization and acquisition live-time remain unverified.

## Edge data reduction

Policies were selected on the chronological first 70% and evaluated on the final 1,006,768 events.

| Policy | Transmit fraction | Reduction | Coincidence recall |
|---|--:|--:|--:|
| Hardware coincidence only | 0.084 | 11.9x | 1.000 |
| ADC threshold selected for 90% train recall | 0.474 | 2.11x | 0.934 |
| MLP selected for 90% train recall | 0.525 | 1.90x | 0.946 |
| ADC threshold selected for 95% train recall | 0.628 | 1.59x | 0.963 |
| MLP selected for 95% train recall | 0.698 | 1.43x | 0.972 |

The MLP does not create a compelling bandwidth advantage over ADC thresholding. Hardware coincidence gives
the largest reduction but cannot preserve unexpected noncoincident anomalies. The table models payload bytes,
not radio energy or protocol overhead.

## Detector health

- The monitor covers 34 active partition-days and reports 19 screening alerts.
- It identifies the 2025-11-17 to 2025-11-19 high-ADC/high-coincidence regime.
- It detects the 2026-06-11 raw change point: ADC median +30 counts, rate -0.490 Hz, and coincidence fraction
  +0.0157 in one day.
- The latest event is 2026-06-19, producing a 26-day ingestion-stale alert as of 2026-07-15.
- These are robust statistical alerts, not automatic hardware-failure diagnoses.

## Still blocked by external measurements

The existing data cannot provide Pico/MCU sleep current, radio duty-cycle energy, whole-board power, Jetson
rail power, balloon/altitude response, calibration-source response, or expert image labels. Those items require
the corresponding hardware, deployment, or annotation; no numerical result is inferred for them here.
