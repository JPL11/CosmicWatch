#!/usr/bin/env python3
"""
Pull and summarize CosmicWatch coincidence events (the true-muon indicator)
from the CREDO Elasticsearch index.

Usage:
    python3 cosmicwatch_summary.py                 # writes cosmicwatch_summary.json
    python3 cosmicwatch_summary.py --events 500    # also dump 500 recent events
    python3 cosmicwatch_summary.py --out foo.json

Credentials can be overridden with env vars CREDO_USER / CREDO_PASS.
"""
import argparse
import json
import sys
import warnings

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")

import urllib3
import requests

from credo_config import es_auth, es_settings, verify_certs

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

COINCIDENT_FILTER = {
    "bool": {
        "filter": [
            {"term": {"source": "cosmicwatch-v3x"}},
            {"term": {"coincident": True}},
        ]
    }
}


def es_search(body):
    es_url, index = es_settings()
    response = requests.post(
        f"{es_url}/{index}/_search",
        auth=es_auth(),
        verify=verify_certs(),
        headers={"Content-Type": "application/json"},
        data=json.dumps(body),
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def summarize():
    _, index = es_settings()
    body = {
        "size": 0,
        "query": COINCIDENT_FILTER,
        "track_total_hits": True,
        "aggs": {
            "adc": {"stats": {"field": "adc_value"}},
            "sipm": {"stats": {"field": "sipm_mv"}},
            # bound temperature to a sane range to drop corrupt readings
            "temp_clean": {
                "filter": {"range": {"temperature_c": {"gte": -50, "lte": 80}}},
                "aggs": {"stats": {"stats": {"field": "temperature_c"}}},
            },
            "pressure": {"stats": {"field": "pressure_pa"}},
            "time_range": {
                "stats": {"field": "timestamp"}
            },
            "adc_histogram": {
                "histogram": {"field": "adc_value", "interval": 250, "min_doc_count": 1}
            },
        },
    }
    res = es_search(body)
    aggs = res["aggregations"]
    total = res["hits"]["total"]["value"]

    def fmt_ts(ms):
        if ms is None:
            return None
        import datetime
        return datetime.datetime.utcfromtimestamp(ms / 1000).isoformat() + "Z"

    return {
        "index": index,
        "filter": "source=cosmicwatch-v3x AND coincident=true (true-muon events)",
        "coincident_event_count": total,
        "adc_value": _stats(aggs["adc"]),
        "sipm_mv": _stats(aggs["sipm"]),
        "temperature_c_cleaned": _stats(aggs["temp_clean"]["stats"]),
        "pressure_pa": _stats(aggs["pressure"]),
        "time_range_utc": {
            "earliest": fmt_ts(aggs["time_range"]["min"]),
            "latest": fmt_ts(aggs["time_range"]["max"]),
        },
        "adc_histogram": [
            {"adc_bin_start": int(b["key"]), "count": b["doc_count"]}
            for b in aggs["adc_histogram"]["buckets"]
        ],
    }


def _stats(s):
    return {
        "count": s.get("count"),
        "min": s.get("min"),
        "max": s.get("max"),
        "avg": round(s["avg"], 2) if s.get("avg") is not None else None,
    }


def recent_events(n):
    res = es_search(
        {
            "size": n,
            "query": COINCIDENT_FILTER,
            "sort": [{"timestamp": {"order": "desc"}}],
            "_source": [
                "timestamp", "adc_value", "sipm_mv", "coincident",
                "temperature_c", "pressure_pa", "detector_name", "device_id",
            ],
        }
    )
    return [h["_source"] for h in res["hits"]["hits"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=0,
                    help="also include N most-recent coincident events")
    ap.add_argument("--out", default="cosmicwatch_summary.json")
    args = ap.parse_args()

    print("Querying Elasticsearch ...", file=sys.stderr)
    out = {"summary": summarize()}
    if args.events:
        out["recent_events"] = recent_events(args.events)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    s = out["summary"]
    print(f"\nCoincident (muon) events: {s['coincident_event_count']:,}")
    print(f"  ADC value   avg={s['adc_value']['avg']}  "
          f"(min {s['adc_value']['min']}, max {s['adc_value']['max']})")
    print(f"  SiPM mV     avg={s['sipm_mv']['avg']}")
    print(f"  Temp C      avg={s['temperature_c_cleaned']['avg']} (cleaned)")
    print(f"  Time range  {s['time_range_utc']['earliest']} "
          f"-> {s['time_range_utc']['latest']}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
