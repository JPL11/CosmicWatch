#!/usr/bin/env python3
"""
Unified CosmicWatch loader across BOTH schema partitions.

The detector data lives in two ingest schemas under source=cosmicwatch-v3x:
  - PARSED (582k): wall-clock `timestamp`/`timestamp_ms`, boolean `coincident`, *_g motion.
  - RAW "AxLab" (2.77M): epoch `wall_time` (the real wall-clock!), `coincidence_flag` (0/1), raw motion.

This module maps both onto a canonical record so every analysis can use ~3.36M events:
  time_utc / time_epoch_s   <- timestamp_ms | wall_time | timestamp
  coincident (bool)         <- coincident   | coincidence_flag
  adc_value, sipm_mv, temperature_c, pressure_pa, deadtime_s, accel_z, partition

Usage:
  from credo_loader import fetch, partition_query, COINC
  rows = fetch("both", max_events=80000)
"""
import datetime as dt
import warnings

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")
import requests
import urllib3

from credo_config import es_auth, es_settings, verify_certs

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SOURCE = "cosmicwatch-v3x"
FETCH_FIELDS = ["timestamp", "timestamp_ms", "wall_time", "pico_timestamp_s", "adc_value", "sipm_mv",
                "coincident", "coincidence_flag", "temperature_c", "pressure_pa", "deadtime_s",
                "accel_z_g", "accel_z", "detector", "detector_name"]

# Per-partition coincidence term (parsed uses boolean, raw uses 0/1 long).
COINC = {"parsed": {"term": {"coincident": True}}, "raw": {"term": {"coincidence_flag": 1}}}


def _request(method, path, body=None, params=None, timeout=180, retries=4):
    import time
    es_url, _ = es_settings()
    last = None
    for attempt in range(retries):
        try:
            r = requests.request(method, f"{es_url}/{path.lstrip('/')}", auth=es_auth(),
                                 verify=verify_certs(), headers={"Content-Type": "application/json"},
                                 params=params, json=body, timeout=timeout)
            if r.status_code in (429, 502, 503, 504):
                last = r; time.sleep(2 * (attempt + 1)); continue
            r.raise_for_status(); return r.json()
        except requests.exceptions.RequestException as e:
            last = e; time.sleep(2 * (attempt + 1))
    if isinstance(last, requests.Response):
        last.raise_for_status()
    raise last


def post(path, body, **kw):
    return _request("POST", path, body, **kw)


def partition_query(partition):
    """partition in {'parsed','raw','both'} -> ES query."""
    base = [{"term": {"source": SOURCE}}]
    if partition == "parsed":
        return {"bool": {"filter": base, "must": [{"exists": {"field": "timestamp"}}]}}
    if partition == "raw":
        return {"bool": {"filter": base, "must_not": [{"exists": {"field": "timestamp"}}]}}
    return {"bool": {"filter": base}}


def _to_epoch(doc):
    if doc.get("timestamp_ms") is not None:
        return float(doc["timestamp_ms"]) / 1000.0
    if doc.get("wall_time") is not None:
        return float(doc["wall_time"])
    ts = doc.get("timestamp")
    if isinstance(ts, str):
        t = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
        try:
            return dt.datetime.fromisoformat(t).timestamp()
        except ValueError:
            return None
    if isinstance(ts, (int, float)):
        return float(ts) / 1000.0
    return None


def canonical_row(doc):
    epoch = _to_epoch(doc)
    coinc = doc.get("coincident")
    if coinc is None:
        cf = doc.get("coincidence_flag")
        coinc = bool(cf) if cf is not None else None
    return {
        "partition": "cosmicwatch_parsed" if doc.get("timestamp") else "cosmicwatch_raw_axlab",
        "time_epoch_s": epoch,
        "time_utc": (dt.datetime.fromtimestamp(epoch, dt.timezone.utc).isoformat().replace("+00:00", "Z")
                     if epoch else None),
        "coincident": coinc,
        "adc_value": doc.get("adc_value"),
        "sipm_mv": doc.get("sipm_mv"),
        "temperature_c": doc.get("temperature_c"),
        "pressure_pa": doc.get("pressure_pa"),
        "deadtime_s": doc.get("deadtime_s"),
        "accel_z": doc.get("accel_z_g", doc.get("accel_z")),
    }


def fetch(partition="both", max_events=0, page_size=5000, scroll="3m"):
    """Scroll the chosen partition(s), return canonical rows (sorted by time within the pull)."""
    _, index = es_settings()
    body = {"size": page_size, "sort": ["_doc"], "_source": FETCH_FIELDS,
            "query": partition_query(partition)}
    es_url, _ = es_settings()
    r = requests.post(f"{es_url}/{index}/_search", auth=es_auth(), verify=verify_certs(),
                      headers={"Content-Type": "application/json"}, params={"scroll": scroll},
                      json=body, timeout=180)
    r.raise_for_status()
    res = r.json()
    sid = res.get("_scroll_id")
    rows = []
    try:
        while True:
            hits = res.get("hits", {}).get("hits", [])
            if not hits:
                break
            for h in hits:
                rows.append(canonical_row(h.get("_source", {})))
                if max_events and len(rows) >= max_events:
                    return rows
            res = post("_search/scroll", {"scroll": scroll, "scroll_id": sid})
            sid = res.get("_scroll_id", sid)
    finally:
        if sid:
            try:
                requests.delete(f"{es_url}/_search/scroll", auth=es_auth(), verify=verify_certs(),
                                headers={"Content-Type": "application/json"},
                                json={"scroll_id": [sid]}, timeout=30)
            except requests.RequestException:
                pass
    return rows


def counts():
    """Quick canonical summary across partitions."""
    out = {}
    for p in ("parsed", "raw"):
        out[p] = post(f"{es_settings()[1]}/_count" if False else f"{es_settings()[1]}/_search",
                      {"size": 0, "query": partition_query(p)})["hits"]["total"]["value"]
    return out


if __name__ == "__main__":
    # smoke: show canonical rows from each partition
    for p in ("parsed", "raw"):
        rows = fetch(p, max_events=2)
        print(f"{p}:")
        for r in rows:
            print("  ", {k: r[k] for k in ("partition", "time_utc", "coincident", "adc_value")})
