# CosmicWatch / CREDO — Comprehensive Data Analysis

Index `credo-detections` · **3,437,063** docs · 5 sources · 72 mapped fields.

> Profiled with server-side aggregations. Companion to `data_readiness.py` (timelines) and `multi_node_probe.py` (multi-node check).

## 1. Sources at a glance

| source | docs | active days | time range | geo | images |
|---|--:|--:|---|:--:|:--:|
| `cosmicwatch-v3x` | 3,355,891 | 16 | 2025-11-02 → 2026-02-25 | — | — |
| `legacy` | 69,000 | 145 | 2017-10-04 → 2018-05-06 | yes | yes |
| `credo.science` | 8,999 | 1 | 2018-05-11 → 2018-05-11 | yes | — |
| `phone-camera` | 3,095 | 24 | 2026-05-18 → 2026-06-19 | yes | yes |
| `credo-science` | 77 | 1 | 2025-11-15 → 2025-11-15 | yes | yes |

## 2. Per-source field coverage

### `cosmicwatch-v3x` (3,355,891 docs)

Populated fields (coverage): `adc_value` 100%, `sipm_mv` 100%, `temperature_c` 100%, `pressure_pa` 100%, `deadtime_s` 100%, `coincidence_flag` 100%, `llm_interpretation` 83%, `accel_x_g` 17%, `accel_y_g` 17%, `accel_z_g` 17%, `gyro_x_degs` 17%, `timestamp` 17%, `timestamp_ms` 17%, `pico_timestamp_s` 17%, `coincident` 17%, `device_id` 17%, `detector_name` 17%, `event` 17%

### `legacy` (69,000 docs)

Populated fields (coverage): `altitude` 100%, `accuracy` 100%, `height` 100%, `width` 100%, `timestamp` 100%, `device_id` 100%, `provider` 100%, `team_id` 100%, `user_id` 100%, `visible` 100%, `frame_content` 100%, `location` 71%

### `credo.science` (8,999 docs)

Populated fields (coverage): `energy` 100%, `altitude` 100%, `timestamp` 100%, `device_id` 100%, `latitude` 100%, `longitude` 100%, `particle_type` 100%

### `phone-camera` (3,095 docs)

Populated fields (coverage): `brightness` 100%, `cluster_size` 100%, `device_id` 100%, `timestamp` 100%, `image_b64` 51%, `provider` 51%, `altitude` 51%, `latitude` 51%, `longitude` 51%

### `credo-science` (77 docs)

Populated fields (coverage): `altitude` 100%, `accuracy` 100%, `height` 100%, `width` 100%, `timestamp` 100%, `device_id` 100%, `location` 100%, `provider` 100%, `team_id` 100%, `user_id` 100%, `visible` 100%, `frame_content` 100%

## 3. CosmicWatch-v3x deep dive

- Distinct device_ids: **1** → {'cosmicwatch-001': 582068}
- Rows with timestamp: **582,068**; with lat/lon: **0**
- **Dual schema:** 582,068 parsed (wall-clock `timestamp`) vs 2,773,823 raw (boot-relative `timestamp_s`, detector 'AxLab', not time-correlatable).
- ADC separability — coincident=True: mean 389.82, p50 334.38, p99 1201.95 (n=72,661)
- ADC separability — coincident=False: mean 249.22, p50 182.54, p99 1053.57 (n=509,407)

| field | min | p50 | mean | p99 | max |
|---|--:|--:|--:|--:|--:|
| `adc_value` | 52.0 | 210.808 | 305.359 | 1522.199 | 4095.0 |
| `sipm_mv` | 2.7 | 10.925 | 16.072 | 78.508 | 1000.0 |
| `temperature_c` | 23.7 | 26.861 | 49.265 | 29.016 | 2505832.0 |
| `pressure_pa` | 0.0 | 101270.977 | 101290.603 | 102006.171 | 349705.406 |
| `deadtime_s` | 0.0 | 301.499 | 1309.327 | 795.549 | 361541248.0 |
| `accel_x_g` | -0.886 | 0.006 | 0.003 | 0.025 | 0.484 |
| `accel_y_g` | -0.555 | -0.0 | -0.001 | 0.008 | 0.34 |
| `accel_z_g` | -1.89 | -1.003 | -0.987 | -0.428 | 1.229 |
| `gyro_x_degs` | -154.6 | 0.012 | 0.031 | 0.5 | 84.2 |

## 4. Geo & image availability

- `legacy` geo via location: {"top_left": {"lat": 54.527672245167196, "lon": 16.81281801313162}, "bottom_right": {"lat": 49.80531306937337, "lon": 22.550374483689666}}
- `credo.science` geo via latitude/longitude: {"field": "latitude/longitude", "count": 8999, "lat_range": [0.0, 0.0], "lon_range": [0.0, 0.0]}
- `phone-camera` geo via latitude/longitude: {"field": "latitude/longitude", "count": 1565, "lat_range": [33.91595458984375, 33.91709518432617], "lon_range": [-118.33549499511719, -118.33383178710938]}
- `credo-science` geo via location: {"top_left": {"lat": 53.65106217563152, "lon": 16.29836923442781}, "bottom_right": {"lat": 49.29571317974478, "lon": 22.6199050527066}}

- `legacy` images: frame_content 69,000
- `phone-camera` images: image_b64 1,569
- `credo-science` images: frame_content 77

## 5. Data quality flags

- ⚠️ cosmicwatch `temperature_c`: max 2505832.0 >> p99 29.016 — corrupt/outlier tail, needs clipping
- ⚠️ cosmicwatch `deadtime_s`: max 361541248.0 >> p99 795.549 — corrupt/outlier tail, needs clipping
- ⚠️ cosmicwatch `gyro_x_degs`: max 84.2 >> p99 0.5 — corrupt/outlier tail, needs clipping
- ⚠️ cosmicwatch `adc_value` saturates at 4095 (12-bit ADC ceiling) — clipped events present
- ⚠️ cosmicwatch `pressure_pa` has 0 Pa floor — invalid/missing-as-zero readings
- ⚠️ `credo.science` is degenerate: lat/lon all 0,0, energy all 0, particle_type constant — not a usable geo or labeled source despite fields being 'present'

## 6. Key findings

- CosmicWatch is one physical node: 1 device_id (cosmicwatch-001), no lat/lon — confirms the GNN/FL multi-node blocker.
- CosmicWatch has TWO schemas: 582,068 parsed events with wall-clock `timestamp`+`coincident`, and 2,773,823 raw docs from detector 'AxLab' whose `timestamp_s` is boot-relative (not wall-clock) — the raw 83% is NOT time-correlatable, so the usable set stays ~582k.
- ADC partially separates coincidence: coincident ADC p50=334.38 vs non-coincident p50=182.54 — distributions overlap, so an ADC threshold is near the achievable ceiling (matches the edge experiment).
- `legacy` (69,000 docs, 2017–18) is the real overlooked asset: decodable PNG hit-crops + genuine Poland GPS (≈49.8–54.5°N, 16.8–22.5°E), 71% geo-tagged — the best candidate for a real CV/geo track (the prior CV dismissal only looked at phone-camera).
- `credo.science` (8,999) is DEGENERATE: lat/lon all 0,0, energy all 0, particle_type constant 'cosmic_ray' — field-present but value-empty; not usable as geo or labels. Correct the handoff.
- `phone-camera` (3,095, 2026, ≈Los Angeles single site) has 1,569 real images (51%) — recent and clean but tiny/single-location; toy CV scale.
- All five sources are schema-disjoint AND temporally disjoint (legacy/credo.science 2017–18; cosmicwatch/phone-camera/credo-science 2025–26) — zero possibility of synchronized cross-source coincidence; any cross-source learning is inherently heterogeneous/federated.
