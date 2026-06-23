#!/usr/bin/env python3
"""
Generate a data dictionary for the credo-detections export (DATA_DICTIONARY.md + .csv).

Combines the live ES field types, the measured per-source coverage (data_analysis.json),
and curated descriptions/units/caveats into a reference that ships alongside the CSV exports.
"""
import csv
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")
import requests
import urllib3

from credo_config import es_auth, es_settings, verify_certs

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ROOT = Path(__file__).parent

# field -> (unit, description, caveat)
DESC = {
    "partition": ("label", "DERIVED export column: separates the two CosmicWatch schemas and the sources.",
                  "Values: cosmicwatch_parsed | cosmicwatch_raw_axlab | legacy | credo.science | phone-camera | credo-science."),
    # --- identity / routing ---
    "source": ("", "Ingest source / dataset the record came from.", ""),
    "device_id": ("", "Device/detector identifier.", "cosmicwatch-v3x has a single id: cosmicwatch-001."),
    "detector_name": ("", "Human-readable detector name.", "parsed CosmicWatch partition."),
    "detector": ("", "Detector label in the raw partition.", "raw AxLab partition uses 'AxLab'."),
    "doc_id": ("", "Source document id (credo.science).", "inferred."),
    "id": ("", "Record id in the raw partition / image sources.", "inferred."),
    "team_id": ("", "CREDO team identifier (image sources).", ""),
    "user_id": ("", "CREDO user/participant identifier (image sources).", ""),
    "provider": ("", "Location provider (e.g. gps/network) for image sources.", ""),
    "device_model": ("", "Phone/device model string.", "image sources; inferred."),
    "event": ("", "Event counter (parsed CosmicWatch).", ""),
    "event_num": ("", "Event counter (raw partition).", ""),
    "es_synced": ("", "Elasticsearch sync flag/marker.", "inferred."),
    # --- timing ---
    "timestamp": ("ISO-8601 UTC", "Wall-clock event time.", "Only on the 582k PARSED CosmicWatch + image sources; 1-second resolution."),
    "timestamp_ms": ("ms since epoch", "Wall-clock time in epoch milliseconds.", "Quantized to 1 second (ends in 000); use pico_timestamp_s for fine timing."),
    "timestamp_s": ("s (boot-relative)", "Seconds since device power-on (raw partition).", "Boot-relative; use `wall_time` for ABSOLUTE time on the raw partition."),
    "pico_timestamp_s": ("s (boot-relative)", "High-resolution (microsecond) device clock.", "Boot-relative; DIFFERENCES are valid inter-arrival times (Poisson check)."),
    "comp_date": ("", "On-device computed date string.", "parsed CosmicWatch."),
    "comp_time": ("", "On-device computed time string.", "parsed CosmicWatch."),
    "time_received": ("date", "Server-side ingestion time.", "image sources."),
    "wall_time": ("s (epoch)", "REAL wall-clock time: Unix epoch seconds, microsecond precision.", "The absolute-time ANCHOR for the raw AxLab partition (which has no `timestamp`); also on phone-camera. Convert to UTC."),
    "ml_timestamp": ("date", "Time an ML/LLM inference was attached.", ""),
    # --- CosmicWatch detector physics ---
    "adc_value": ("ADC counts", "Pulse height; PROPORTIONAL TO ENERGY DEPOSITED.", "0-4095 (12-bit); saturates at 4095; ~5.8 keV/ADC (approx MIP calibration)."),
    "sipm_mv": ("mV", "SiPM pulse amplitude in millivolts.", ""),
    "coincident": ("bool", "Both internal panels fired (intra-unit coincidence).", "Weak hardware label; ~12% in clean window; parsed partition."),
    "coincidence_flag": ("0/1", "Coincidence as an integer (raw partition).", "Same meaning as coincident."),
    "temperature_c": ("deg C", "On-board temperature.", "Has corrupt outliers (max ~2.5e6); clip to [-50,80]."),
    "pressure_pa": ("Pa", "On-board atmospheric pressure.", "Has 0-Pa floor + corrupt highs; clip to [80000,110000]."),
    "deadtime_s": ("s (cumulative)", "Running total of detector dead-time.", "Monotonic; per-event = difference; ~0.07% of live time."),
    "cosmicwatch_adc": ("ADC counts", "Alias of adc_value in some records.", "inferred duplicate."),
    "cosmicwatch_sipm_mv": ("mV", "Alias of sipm_mv.", "inferred duplicate."),
    "cosmicwatch_coincident": ("bool", "Alias of coincident.", "inferred duplicate."),
    "cosmicwatch_temperature": ("deg C", "Alias of temperature_c.", "inferred duplicate."),
    "cosmicwatch_pressure": ("Pa", "Alias of pressure_pa.", "inferred duplicate."),
    "cosmicwatch_deadtime": ("s", "Alias of deadtime_s.", "inferred duplicate."),
    # --- motion sensors ---
    "accel_x_g": ("g", "Accelerometer X in gravitational units.", "parsed CosmicWatch (~17% coverage)."),
    "accel_y_g": ("g", "Accelerometer Y in gravitational units.", ""),
    "accel_z_g": ("g", "Accelerometer Z in g; ~-1 g at rest = gravity vector (tilt).", ""),
    "accel_x": ("raw", "Raw accelerometer X (raw partition).", ""),
    "accel_y": ("raw", "Raw accelerometer Y.", ""),
    "accel_z": ("raw", "Raw accelerometer Z.", ""),
    "gyro_x_degs": ("deg/s", "Gyroscope X angular rate.", ""),
    "gyro_y_degs": ("deg/s", "Gyroscope Y angular rate.", ""),
    "gyro_z_degs": ("deg/s", "Gyroscope Z angular rate.", ""),
    "gyro_x": ("raw", "Raw gyroscope X (raw partition).", ""),
    "gyro_y": ("raw", "Raw gyroscope Y.", ""),
    "gyro_z": ("raw", "Raw gyroscope Z.", ""),
    # --- geo ---
    "latitude": ("deg", "Latitude.", "credo.science values are all 0.0 (DEGENERATE); phone-camera real (~LA)."),
    "longitude": ("deg", "Longitude.", "credo.science all 0.0 (DEGENERATE); phone-camera real."),
    "altitude": ("m", "Altitude.", "image sources / credo.science."),
    "location": ("geo_point {lat,lon}", "Geo-point coordinate object.", "legacy & credo-science: real Poland coords."),
    "accuracy": ("m", "Location accuracy radius.", "image sources."),
    "gps_accuracy": ("m", "GPS accuracy radius.", "inferred."),
    # --- image / CV ---
    "frame_content": ("base64 PNG", "Image hit-crop (~20x20 RGBA PNG), base64-encoded.", "legacy (69k) & credo-science (77); decodable PNG."),
    "image_b64": ("base64 PNG", "Image, base64-encoded.", "phone-camera: present on 1,569 of 3,095 rows."),
    "frame_width": ("px", "Full source frame width.", ""),
    "frame_height": ("px", "Full source frame height.", ""),
    "width": ("px", "Full image width (e.g. 960).", "image sources; crop itself is ~20x20."),
    "height": ("px", "Full image height (e.g. 720).", "image sources."),
    "brightness": ("0-255", "Hit/region brightness.", "phone-camera."),
    "hit_x": ("px", "Hit x-coordinate within the frame.", ""),
    "hit_y": ("px", "Hit y-coordinate within the frame.", ""),
    "x": ("px", "Hit x-coordinate (credo-science).", ""),
    "y": ("px", "Hit y-coordinate (credo-science).", ""),
    "visible": ("bool", "Whether the hit is flagged visible.", "legacy: constant False -> NOT a usable label."),
    "cluster_size": ("px", "Size of the detected pixel cluster.", "phone-camera."),
    # --- particle / ML ---
    "energy": ("eV (nominal)", "Reconstructed particle energy.", "credo.science only and ALL 0 -> DEGENERATE/unusable."),
    "particle_type": ("", "Particle class label.", "credo.science: constant 'cosmic_ray' (single-valued)."),
    "ml_prediction": ("class id", "ML model predicted class.", "inferred."),
    "ml_probability": ("0-1", "ML model confidence.", "inferred."),
    "llm_interpretation": ("text", "LLM-generated interpretation string.", "Mostly LLM-ERROR strings (e.g. 'LLM unavailable'); not a usable label."),
    "metadata": ("json object", "Free-form nested metadata.", "JSON-encoded in CSV cells."),
}

GROUPS = [
    ("Export / identity", ["partition", "source", "device_id", "device_model", "detector_name", "detector",
                           "doc_id", "id", "team_id", "user_id", "provider", "event", "event_num", "es_synced"]),
    ("Timing", ["timestamp", "timestamp_ms", "timestamp_s", "pico_timestamp_s", "comp_date", "comp_time",
                "time_received", "wall_time", "ml_timestamp"]),
    ("CosmicWatch detector (physics)", ["adc_value", "sipm_mv", "coincident", "coincidence_flag", "temperature_c",
                                        "pressure_pa", "deadtime_s", "cosmicwatch_adc", "cosmicwatch_sipm_mv",
                                        "cosmicwatch_coincident", "cosmicwatch_temperature", "cosmicwatch_pressure",
                                        "cosmicwatch_deadtime"]),
    ("Motion sensors", ["accel_x_g", "accel_y_g", "accel_z_g", "accel_x", "accel_y", "accel_z",
                        "gyro_x_degs", "gyro_y_degs", "gyro_z_degs", "gyro_x", "gyro_y", "gyro_z"]),
    ("Geo", ["latitude", "longitude", "altitude", "location", "accuracy", "gps_accuracy"]),
    ("Image / CV", ["frame_content", "image_b64", "frame_width", "frame_height", "width", "height",
                    "brightness", "hit_x", "hit_y", "x", "y", "visible", "cluster_size"]),
    ("Particle / ML", ["energy", "particle_type", "ml_prediction", "ml_probability", "llm_interpretation", "metadata"]),
]


def es_types():
    es_url, index = es_settings()
    r = requests.get(f"{es_url}/{index}/_mapping", auth=es_auth(), verify=verify_certs(), timeout=60)
    r.raise_for_status()
    props = list(r.json().values())[0]["mappings"]["properties"]
    return {k: v.get("type", "object") for k, v in props.items()}


def field_sources():
    try:
        da = json.load(open(ROOT / "data_analysis.json"))
    except Exception:
        return {}
    out = {}
    for src, info in (da.get("sources") or {}).items():
        for f in (info.get("coverage") or {}):
            out.setdefault(f, []).append(src)
    return out


def main():
    types = es_types()
    srcs = field_sources()
    rows = []  # (field, type, unit, sources, desc, caveat)
    for field in list(types) + ["partition"]:
        unit, desc, caveat = DESC.get(field, ("", "(no description)", ""))
        slist = "partition (derived)" if field == "partition" else ", ".join(sorted(srcs.get(field, []))) or "—"
        rows.append((field, types.get(field, "derived"), unit, slist, desc, caveat))
    by_field = {r[0]: r for r in rows}

    # CSV
    with open(ROOT / "data_dictionary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["field", "es_type", "unit", "populated_in_sources", "description", "caveats"])
        for grp, fields in GROUPS:
            for f in fields:
                if f in by_field:
                    w.writerow(by_field[f])

    # Markdown
    L = ["# Data Dictionary --- `credo-detections` export\n",
         "Companion to `credo_export.csv(.gz)` (full index) and `credo_useful.csv(.gz)` (usable events + "
         "useful images). 73 columns = 72 mapped ES fields + the derived `partition` column.\n",
         "**Reading the export:** the first column `partition` separates the two CosmicWatch schema versions "
         "and the sources. Nested fields (`location`, `metadata`) and lists are JSON-encoded in their cells; "
         "booleans are `true`/`false`; image fields hold base64 PNG text.\n",
         "## Key caveats (read first)\n",
         "- **Both** CosmicWatch partitions are usable (~3.36M events): `cosmicwatch_parsed` (582k) keys on "
         "`timestamp`+`coincident`; `cosmicwatch_raw_axlab` (2.77M) keys on `wall_time`+`coincidence_flag`.",
         "- Canonical fields: time = `timestamp` else `wall_time`; label = `coincident` else `coincidence_flag` "
         "(see `credo_loader.py`).",
         "- `timestamp_ms` is **1-second** resolution; `wall_time`/`pico_timestamp_s` are microsecond.",
         "- `credo.science` is **degenerate**: `latitude`/`longitude` = 0,0, `energy` = 0, `particle_type` constant.",
         "- `visible` is constant `False`; `llm_interpretation` is mostly error strings --- neither is a usable label.",
         "- `adc_value` is 0--4095 (12-bit, saturates at 4095); ~5.8 keV/ADC by approximate MIP calibration.\n"]
    for grp, fields in GROUPS:
        L.append(f"## {grp}\n")
        L.append("| field | type | unit | populated in | description | caveats |")
        L.append("|---|---|---|---|---|---|")
        mq = lambda s: str(s).replace("|", "/")  # avoid breaking the markdown table
        for f in fields:
            if f in by_field:
                fld, typ, unit, slist, desc, caveat = by_field[f]
                L.append(f"| `{fld}` | {typ} | {mq(unit)} | {mq(slist)} | {mq(desc)} | {mq(caveat)} |")
        L.append("")
    (ROOT / "DATA_DICTIONARY.md").write_text("\n".join(L) + "\n")

    print(f"Wrote DATA_DICTIONARY.md and data_dictionary.csv ({len(rows)} fields).")


if __name__ == "__main__":
    main()
