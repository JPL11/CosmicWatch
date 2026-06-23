# CosmicWatch / CREDO Data Export — README

A self-describing, version-separated dump of the CREDO `credo-detections` Elasticsearch
index (3,437,063 docs, verified 2026-06-22). Pair these data files with
[`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) (every column's type, unit, source, and caveats).

## Files

| File | Rows | Size | What it is |
|---|--:|--:|---|
| `credo_export.csv.gz` | 3,437,063 | 183 MB | **Full index** — all 5 sources, all 73 columns, includes base64 images |
| `credo_useful.csv` / `.gz` | 654,240 | 200 / 52 MB | **Curated subset** — usable events + the useful images only |
| `DATA_DICTIONARY.md` / `data_dictionary.csv` | 73 fields | — | Column reference (types, units, caveats) |

> The CSV/`.gz` data files are **not** in git (too large for GitHub); they are shared directly.
> The dictionary and generator scripts **are** in the repo.

## The `partition` column (first column) — how to separate versions/sources

Every row is labeled so you can split cleanly without knowing the schema quirks:

| `partition` | rows | use |
|---|--:|---|
| `cosmicwatch_parsed` | 582,068 | **usable detector events** (wall-clock `timestamp`, `coincident`, ADC/energy) |
| `legacy` | 69,000 | **real image hit-crops** (20×20 PNG) + Poland GPS (2017–18) |
| `phone-camera` | 3,095 | recent images (1,569 with `image_b64`), LA area |
| `credo-science` | 77 | tiny image set |
| `cosmicwatch_raw_axlab` | 2,773,823 | **usable** detector events — key on `wall_time` (real epoch) + `coincidence_flag` |
| `credo.science` | 8,999 | **degenerate** (lat/lon 0,0, energy 0) — avoid |

Both CosmicWatch partitions are usable (~3.36M events total); only `credo.science` is dropped from the
useful set. Use `credo_loader.py` for a canonical view (time = `timestamp` else `wall_time`;
coincidence = `coincident` else `coincidence_flag`).

## Quick start (pandas)

```python
import pandas as pd
df = pd.read_csv("credo_useful.csv.gz")          # gzip read is automatic

events = df[df.partition == "cosmicwatch_parsed"] # 582k usable detector events
images = df[df.partition == "legacy"]             # 69k image crops + GPS

# decode an image crop (base64 PNG)
import base64, io; from PIL import Image
img = Image.open(io.BytesIO(base64.b64decode(images.iloc[0]["frame_content"])))
```

Command-line split (no Python):
```bash
zcat credo_useful.csv.gz | awk -F, 'NR==1 || $1=="cosmicwatch_parsed"' > events.csv
```

## Read-first caveats (full list in the dictionary)

- **Usable events ≈ 3.36M** = `cosmicwatch_parsed` (582k, `timestamp`+`coincident`) **plus**
  `cosmicwatch_raw_axlab` (2.77M, `wall_time`+`coincidence_flag`). `wall_time` is real epoch seconds.
- `timestamp_ms` is **1-second** resolution; `wall_time`/`pico_timestamp_s` are **microsecond** precision.
- The two partitions cover different windows (parsed 2025-11→2026-02; raw 2026-05→2026-06) and the
  detector response drifted between them (coincidence rate 13%→8%).
- `adc_value` ∝ energy, 0–4095 (saturates at 4095); ≈ **5.8 keV/ADC** by approximate MIP calibration.
- `credo.science` lat/lon/energy are degenerate; `visible` is constant `False`; `llm_interpretation`
  is mostly error strings — none are usable labels.
- Nested fields (`location`, `metadata`) are JSON-encoded in their cells; booleans are `true`/`false`.

## Reproduce

```bash
python3 export_data.py --out credo_export.csv          # full export (add --no-images to slim it)
python3 make_data_dictionary.py                         # regenerate the dictionary
```

Both read Elasticsearch credentials from a local `.env` (see `.env.example`).
