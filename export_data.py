#!/usr/bin/env python3
"""
Stream the full credo-detections index to a CSV (all sources, all fields).

Memory-safe: scrolls the index and writes rows incrementally, so it never holds
3.4M docs in memory. Nested fields (location, metadata) and lists are JSON-encoded
into their cells; booleans become 'true'/'false'. Image blobs (frame_content,
image_b64) are included by default (base64 text).

Usage:
  python3 export_data.py --out credo_export.csv
  python3 export_data.py --sources cosmicwatch-v3x --no-images --max-rows 100000
"""
import argparse
import csv
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")
import requests
import urllib3

from credo_config import es_auth, es_settings, verify_certs

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PRIORITY = ["source", "timestamp", "timestamp_ms", "device_id", "detector_name",
            "adc_value", "sipm_mv", "coincident", "coincidence_flag", "temperature_c",
            "pressure_pa", "deadtime_s", "latitude", "longitude", "particle_type", "energy"]
IMAGE_FIELDS = ["frame_content", "image_b64"]


def es_post(url_path, body, params=None, timeout=180, retries=4):
    es_url, _ = es_settings()
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(f"{es_url}/{url_path.lstrip('/')}", auth=es_auth(), verify=verify_certs(),
                              headers={"Content-Type": "application/json"}, params=params, json=body,
                              timeout=timeout)
            if r.status_code in (429, 502, 503, 504):
                last = r; time.sleep(2 * (attempt + 1)); continue
            r.raise_for_status(); return r.json()
        except requests.exceptions.RequestException as e:
            last = e; time.sleep(2 * (attempt + 1))
    if isinstance(last, requests.Response):
        last.raise_for_status()
    raise last


def all_fields():
    _, index = es_settings()
    m = es_post(f"{index}/_mapping", {}, timeout=60) if False else None
    es_url, _ = es_settings()
    r = requests.get(f"{es_url}/{index}/_mapping", auth=es_auth(), verify=verify_certs(), timeout=60)
    r.raise_for_status()
    props = list(r.json().values())[0]["mappings"]["properties"]
    return sorted(props.keys())


def header(fields, include_images):
    fields = [f for f in fields if include_images or f not in IMAGE_FIELDS]
    rest = [f for f in fields if f not in PRIORITY]
    # `partition` is a DERIVED column so recipients can cleanly split the versions/sources.
    return ["partition"] + [f for f in PRIORITY if f in fields] + rest


def partition(src):
    """Self-separating label: distinguishes the two CosmicWatch schemas, else the source."""
    s = src.get("source")
    if s == "cosmicwatch-v3x":
        return "cosmicwatch_parsed" if src.get("timestamp") else "cosmicwatch_raw_axlab"
    return s or "unknown"


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dict, list)):
        return json.dumps(v, separators=(",", ":"))
    return v


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="credo_export.csv")
    ap.add_argument("--sources", nargs="*", default=None, help="limit to these sources (default: all)")
    ap.add_argument("--page-size", type=int, default=5000)
    ap.add_argument("--scroll", default="5m")
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all")
    ap.add_argument("--no-images", action="store_true", help="drop frame_content/image_b64")
    return ap.parse_args()


def main():
    args = parse_args()
    _, index = es_settings()
    include_images = not args.no_images
    cols = header(all_fields(), include_images)

    query = {"match_all": {}}
    if args.sources:
        query = {"terms": {"source": args.sources}}
    body = {"size": args.page_size, "sort": ["_doc"], "query": query}

    es_url, _ = es_settings()
    r = requests.post(f"{es_url}/{index}/_search", auth=es_auth(), verify=verify_certs(),
                      headers={"Content-Type": "application/json"}, params={"scroll": args.scroll},
                      json=body, timeout=180)
    r.raise_for_status()
    res = r.json()
    total = res.get("hits", {}).get("total", {}).get("value", "?")
    scroll_id = res.get("_scroll_id")

    written = 0
    started = time.time()
    print(f"Exporting index '{index}' (~{total} docs reported) -> {args.out}", flush=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore", restval="")
        writer.writeheader()
        try:
            while True:
                hits = res.get("hits", {}).get("hits", [])
                if not hits:
                    break
                for h in hits:
                    src = h.get("_source", {})
                    row = {k: fmt(src.get(k)) for k in cols if k != "partition"}
                    row["partition"] = partition(src)
                    writer.writerow(row)
                    written += 1
                    if args.max_rows and written >= args.max_rows:
                        raise StopIteration
                if written % 50000 < args.page_size:
                    rate = written / max(1e-9, time.time() - started)
                    print(f"  {written:,} rows  ({rate:,.0f}/s)", flush=True)
                res = es_post("_search/scroll", {"scroll": args.scroll, "scroll_id": scroll_id})
                scroll_id = res.get("_scroll_id", scroll_id)
        except StopIteration:
            pass
        finally:
            if scroll_id:
                try:
                    requests.delete(f"{es_url}/_search/scroll", auth=es_auth(), verify=verify_certs(),
                                    headers={"Content-Type": "application/json"},
                                    json={"scroll_id": [scroll_id]}, timeout=30)
                except requests.RequestException:
                    pass

    import os
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"DONE: wrote {written:,} rows, {len(cols)} columns, {size_mb:,.1f} MB to {args.out} "
          f"in {time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
