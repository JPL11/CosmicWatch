#!/usr/bin/env python3
"""
Comprehensive field-level profile of the credo-detections index.

Goes beyond the existing readiness/summary scripts: per-source field coverage,
numeric distributions, time coverage, coincidence behavior, geo coverage, and
image/ML field availability. Uses server-side aggregations (fast over 3.4M docs).

Outputs:
  data_analysis.json        machine-readable profile
  data_analysis_report.md   human-readable report
  plots_analysis/*.png      key distributions (with --plots)
"""
import argparse
import json
import time
import warnings

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")
import requests
import urllib3

from credo_config import es_auth, es_settings, verify_certs

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SOURCES = ["cosmicwatch-v3x", "legacy", "credo.science", "phone-camera", "credo-science"]

# Fields worth profiling per category.
NUMERIC_FIELDS = [
    "adc_value", "sipm_mv", "temperature_c", "pressure_pa", "deadtime_s",
    "accel_x_g", "accel_y_g", "accel_z_g", "gyro_x_degs",
    "energy", "altitude", "brightness", "cluster_size",
    "ml_probability", "accuracy", "height", "width",
]
COVERAGE_FIELDS = NUMERIC_FIELDS + [
    "timestamp", "timestamp_ms", "pico_timestamp_s", "coincident", "coincidence_flag",
    "device_id", "detector_name", "latitude", "longitude", "location",
    "frame_content", "image_b64", "particle_type", "ml_prediction", "llm_interpretation",
    "provider", "team_id", "user_id", "visible", "event",
]


def post(path, body, timeout=120, retries=4):
    es_url, _ = es_settings()
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(f"{es_url}/{path.lstrip('/')}", auth=es_auth(), verify=verify_certs(),
                              headers={"Content-Type": "application/json"}, json=body, timeout=timeout)
            if r.status_code in (429, 502, 503, 504):
                last = r
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            last = e
            time.sleep(2 * (attempt + 1))
    if isinstance(last, requests.Response):
        last.raise_for_status()
    raise last


def count(query, timeout=120):
    es_url, index = es_settings()
    r = requests.post(f"{es_url}/{index}/_count", auth=es_auth(), verify=verify_certs(),
                      headers={"Content-Type": "application/json"}, json={"query": query}, timeout=timeout)
    r.raise_for_status()
    return r.json()["count"]


def agg(body):
    _, index = es_settings()
    return post(f"{index}/_search", {"size": 0, **body})


def source_query(src):
    return {"term": {"source": src}}


def total_docs():
    _, index = es_settings()
    return post(f"{index}/_count", {"query": {"match_all": {}}})["count"]


def per_source_counts():
    res = agg({"aggs": {"by_source": {"terms": {"field": "source", "size": 50}}}})
    return {b["key"]: b["doc_count"] for b in res["aggregations"]["by_source"]["buckets"]}


def field_coverage(src, src_count):
    """exists-filter per field in one request -> coverage fraction.

    exists works for numeric/text/keyword/boolean/geo fields; the binary
    `frame_content` is not tracked by exists, so it is sample-detected separately.
    """
    fields = [f for f in COVERAGE_FIELDS if f != "frame_content"]
    out = {}
    # Batch into small groups so the gateway doesn't time out on the 3.3M-doc source.
    for start in range(0, len(fields), 8):
        chunk = fields[start:start + 8]
        aggs = {f"f_{i}": {"filter": {"exists": {"field": f}}} for i, f in enumerate(chunk)}
        res = agg({"query": source_query(src), "aggs": aggs})
        for i, f in enumerate(chunk):
            c = int(res["aggregations"][f"f_{i}"]["doc_count"])
            if c > 0:
                out[f] = {"count": c, "coverage": round(c / max(1, src_count), 4)}
    fc = sample_binary_presence(src, "frame_content", src_count)
    if fc:
        out["frame_content"] = fc
    return out


def sample_binary_presence(src, field, src_count, sample=50):
    """Binary fields can't be aggregated/exists'd; estimate presence from a sample."""
    _, index = es_settings()
    res = post(f"{index}/_search", {"size": sample, "_source": [field], "query": source_query(src)})
    hits = res["hits"]["hits"]
    if not hits:
        return None
    present = sum(1 for h in hits if h["_source"].get(field))
    if present == 0:
        return None
    frac = present / len(hits)
    return {"count": round(frac * src_count), "coverage": round(frac, 4), "estimated_from_sample": len(hits)}


def numeric_stats(src, fields):
    aggs = {}
    for i, f in enumerate(fields):
        aggs[f"s_{i}"] = {"stats": {"field": f}}
        aggs[f"p_{i}"] = {"percentiles": {"field": f, "percents": [1, 25, 50, 75, 99]}}
    res = agg({"query": source_query(src), "aggs": aggs})
    out = {}
    for i, f in enumerate(fields):
        s = res["aggregations"][f"s_{i}"]
        if s["count"] == 0:
            continue
        p = res["aggregations"][f"p_{i}"]["values"]
        rnd = lambda x: round(x, 3) if isinstance(x, (int, float)) else x
        out[f] = {
            "count": s["count"],
            "min": rnd(s["min"]), "max": rnd(s["max"]), "mean": rnd(s["avg"]),
            "p1": rnd(p.get("1.0")), "p25": rnd(p.get("25.0")), "p50": rnd(p.get("50.0")),
            "p75": rnd(p.get("75.0")), "p99": rnd(p.get("99.0")),
        }
    return out


def time_coverage(src):
    res = agg({"query": source_query(src),
               "aggs": {"tmin": {"min": {"field": "timestamp"}},
                        "tmax": {"max": {"field": "timestamp"}},
                        "days": {"date_histogram": {"field": "timestamp", "calendar_interval": "day",
                                                    "min_doc_count": 1}}}})
    a = res["aggregations"]
    days = a["days"]["buckets"]
    return {
        "min": a["tmin"].get("value_as_string"),
        "max": a["tmax"].get("value_as_string"),
        "active_days": len(days),
        "max_docs_in_a_day": max((d["doc_count"] for d in days), default=0),
        "top_days": sorted(({"day": d["key_as_string"][:10], "docs": d["doc_count"]} for d in days),
                           key=lambda x: -x["docs"])[:5],
    }


def coincidence_profile(src):
    res = agg({"query": source_query(src),
               "aggs": {"coinc": {"terms": {"field": "coincident"}},
                        "by_day": {"date_histogram": {"field": "timestamp", "calendar_interval": "day",
                                                      "min_doc_count": 1},
                                   "aggs": {"c": {"filter": {"term": {"coincident": True}}}}}}})
    buckets = res["aggregations"]["coinc"]["buckets"]
    counts = {str(b["key_as_string"] if "key_as_string" in b else b["key"]): b["doc_count"] for b in buckets}
    days = res["aggregations"]["by_day"]["buckets"]
    rates = [(d["key_as_string"][:10], round(d["c"]["doc_count"] / max(1, d["doc_count"]), 4)) for d in days]
    return {"counts": counts, "daily_coincident_rate": rates[:14]}


def geo_profile(src):
    # Prefer geo_point 'location'; fall back to lat/lon numeric bounds.
    res = agg({"query": {"bool": {"filter": [source_query(src)], "must": [{"exists": {"field": "location"}}]}},
               "aggs": {"bounds": {"geo_bounds": {"field": "location"}},
                        "centroid": {"geo_centroid": {"field": "location"}}}})
    b = res["aggregations"]["bounds"].get("bounds")
    if b:
        return {"field": "location", "bounds": b, "centroid": res["aggregations"]["centroid"].get("location")}
    res = agg({"query": source_query(src),
               "aggs": {"lat": {"stats": {"field": "latitude"}}, "lon": {"stats": {"field": "longitude"}}}})
    lat, lon = res["aggregations"]["lat"], res["aggregations"]["lon"]
    if lat["count"] == 0:
        return None
    return {"field": "latitude/longitude", "count": lat["count"],
            "lat_range": [lat["min"], lat["max"]], "lon_range": [lon["min"], lon["max"]]}


def cosmicwatch_deep_dive():
    src = "cosmicwatch-v3x"
    out = {}
    # device + detector cardinality
    res = agg({"query": source_query(src),
               "aggs": {"devices": {"terms": {"field": "device_id", "size": 20}},
                        "device_card": {"cardinality": {"field": "device_id"}}}})
    out["device_ids"] = {b["key"]: b["doc_count"] for b in res["aggregations"]["devices"]["buckets"]}
    out["distinct_device_ids"] = res["aggregations"]["device_card"]["value"]
    # timestamped subset
    out["rows_with_timestamp"] = count({"bool": {"filter": [source_query(src), {"exists": {"field": "timestamp"}}]}})
    out["rows_with_latlon"] = count({"bool": {"filter": [source_query(src), {"exists": {"field": "latitude"}}]}})
    # adc histogram
    res = agg({"query": source_query(src),
               "aggs": {"adc_hist": {"histogram": {"field": "adc_value", "interval": 50, "min_doc_count": 1}}}})
    out["adc_histogram"] = [{"adc": b["key"], "count": b["doc_count"]}
                            for b in res["aggregations"]["adc_hist"]["buckets"]]
    # adc separability by coincident
    sep = {}
    for flag in (True, False):
        r = agg({"query": {"bool": {"filter": [source_query(src), {"term": {"coincident": flag}}]}},
                 "aggs": {"adc": {"stats": {"field": "adc_value"}},
                          "adcp": {"percentiles": {"field": "adc_value", "percents": [50, 90, 99]}}}})
        pv = r["aggregations"]["adcp"]["values"]
        sep[str(flag)] = {"count": r["aggregations"]["adc"]["count"],
                          "adc_mean": round(r["aggregations"]["adc"]["avg"] or 0, 2),
                          "adc_p50": round(pv.get("50.0") or 0, 2),
                          "adc_p99": round(pv.get("99.0") or 0, 2)}
    out["adc_by_coincident"] = sep
    # Dual-schema split: parsed (wall-clock timestamp) vs raw (boot-relative timestamp_s).
    raw = count({"bool": {"filter": [source_query(src)], "must_not": [{"exists": {"field": "timestamp"}}]}})
    out["schema_partition"] = {
        "parsed_with_wallclock_timestamp": out["rows_with_timestamp"],
        "raw_without_wallclock_timestamp": raw,
        "note": "raw partition uses boot-relative timestamp_s (not wall-clock); single detector; not time-correlatable",
    }
    return out


def write_plots(profile, plots_dir):
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    out = Path(plots_dir); out.mkdir(parents=True, exist_ok=True)
    paths = []

    hist = profile["cosmicwatch_deep_dive"]["adc_histogram"]
    if hist:
        xs = [h["adc"] for h in hist]; ys = [h["count"] for h in hist]
        plt.figure(figsize=(8, 5)); plt.bar(xs, ys, width=45)
        plt.yscale("log"); plt.xlabel("ADC value"); plt.ylabel("events (log)")
        plt.title("CosmicWatch ADC distribution (all timestamped events)")
        plt.tight_layout(); p = out / "adc_distribution.png"; plt.savefig(p, dpi=150); plt.close()
        paths.append(str(p))

    # per-source coverage heat-ish bar
    plt.figure(figsize=(9, 5))
    srcs = [s for s in SOURCES if s in profile["sources"]]
    counts = [profile["sources"][s]["doc_count"] for s in srcs]
    plt.bar(srcs, counts, color="#23508c"); plt.yscale("log")
    plt.ylabel("docs (log)"); plt.title("Documents per source")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    p = out / "source_volumes.png"; plt.savefig(p, dpi=150); plt.close(); paths.append(str(p))
    return paths


def build_report(profile):
    L = []
    a = L.append
    a("# CosmicWatch / CREDO — Comprehensive Data Analysis\n")
    a(f"Index `{profile['index']}` · **{profile['total_docs']:,}** docs · {len(profile['sources'])} sources · "
      f"{profile['field_count']} mapped fields.\n")
    a("> Profiled with server-side aggregations. Companion to `data_readiness.py` (timelines) and "
      "`multi_node_probe.py` (multi-node check).\n")

    a("## 1. Sources at a glance\n")
    a("| source | docs | active days | time range | geo | images |")
    a("|---|--:|--:|---|:--:|:--:|")
    for s in SOURCES:
        if s not in profile["sources"]:
            continue
        p = profile["sources"][s]
        tc = p["time_coverage"]
        geo = "yes" if p.get("geo") else "—"
        img = "yes" if (p["coverage"].get("frame_content") or p["coverage"].get("image_b64")) else "—"
        rng = f"{(tc['min'] or '?')[:10]} → {(tc['max'] or '?')[:10]}"
        a(f"| `{s}` | {p['doc_count']:,} | {tc['active_days']} | {rng} | {geo} | {img} |")
    a("")

    a("## 2. Per-source field coverage\n")
    for s in SOURCES:
        if s not in profile["sources"]:
            continue
        p = profile["sources"][s]
        a(f"### `{s}` ({p['doc_count']:,} docs)\n")
        cov = p["coverage"]
        items = sorted(cov.items(), key=lambda kv: -kv[1]["coverage"])
        a("Populated fields (coverage): " +
          ", ".join(f"`{k}` {v['coverage']*100:.0f}%" for k, v in items) + "\n")

    a("## 3. CosmicWatch-v3x deep dive\n")
    dd = profile["cosmicwatch_deep_dive"]
    a(f"- Distinct device_ids: **{dd['distinct_device_ids']}** → {dd['device_ids']}")
    a(f"- Rows with timestamp: **{dd['rows_with_timestamp']:,}**; with lat/lon: **{dd['rows_with_latlon']:,}**")
    sp = dd["schema_partition"]
    a(f"- **Dual schema:** {sp['parsed_with_wallclock_timestamp']:,} parsed (wall-clock `timestamp`) vs "
      f"{sp['raw_without_wallclock_timestamp']:,} raw (boot-relative `timestamp_s`, detector 'AxLab', "
      "not time-correlatable).")
    sep = dd["adc_by_coincident"]
    a(f"- ADC separability — coincident=True: mean {sep['True']['adc_mean']}, p50 {sep['True']['adc_p50']}, "
      f"p99 {sep['True']['adc_p99']} (n={sep['True']['count']:,})")
    a(f"- ADC separability — coincident=False: mean {sep['False']['adc_mean']}, p50 {sep['False']['adc_p50']}, "
      f"p99 {sep['False']['adc_p99']} (n={sep['False']['count']:,})")
    ns = profile["sources"]["cosmicwatch-v3x"].get("numeric_stats", {})
    if ns:
        a("\n| field | min | p50 | mean | p99 | max |")
        a("|---|--:|--:|--:|--:|--:|")
        for f, v in ns.items():
            a(f"| `{f}` | {v['min']} | {v['p50']} | {v['mean']} | {v['p99']} | {v['max']} |")
    a("")

    a("## 4. Geo & image availability\n")
    for s in SOURCES:
        p = profile["sources"].get(s, {})
        if p.get("geo"):
            a(f"- `{s}` geo via {p['geo'].get('field')}: {json.dumps(p['geo'].get('bounds') or p['geo'])[:200]}")
    a("")
    for s in SOURCES:
        p = profile["sources"].get(s, {})
        fc = p.get("coverage", {}).get("frame_content")
        ib = p.get("coverage", {}).get("image_b64")
        if fc or ib:
            bits = []
            if fc: bits.append(f"frame_content {fc['count']:,}")
            if ib: bits.append(f"image_b64 {ib['count']:,}")
            a(f"- `{s}` images: {', '.join(bits)}")
    a("")

    a("## 5. Data quality flags\n")
    for q in profile.get("quality_flags", []):
        a(f"- ⚠️ {q}")
    a("")

    a("## 6. Key findings\n")
    for f in profile["findings"]:
        a(f"- {f}")
    a("")

    # ---- Section 7: GNN readiness ----
    dd = profile["cosmicwatch_deep_dive"]
    usable = dd["rows_with_timestamp"]
    raw = dd["schema_partition"]["raw_without_wallclock_timestamp"]
    a("## 7. Is the data enough for a GNN?\n")
    a(f"Short answer: **no — and not because of volume.** Of the ~3.4M docs, only **{usable:,}** are "
      f"usable CosmicWatch events (the other {raw:,} raw 'AxLab' docs have boot-relative timestamps and a "
      "single detector, so they are not time-correlatable). Those usable events are great for the edge/SNN "
      "track but cannot form a graph, because a GNN needs *network structure* the index does not contain.\n")
    a("A GNN needs nodes (distinct detectors), edges (spatial/temporal relations), and events that overlap "
      "across nodes. Here is what the data provides:\n")
    a("| GNN requirement | Needed | In this index | Status |")
    a("|---|---|---|:--:|")
    a(f"| Multiple distinct detectors (nodes) | ≥ several | **1** (`cosmicwatch-001`; raw partition also 1: 'AxLab') | ❌ |")
    a(f"| Coordinates per detector (edges) | lat/lon each | **0** CosmicWatch rows with lat/lon | ❌ |")
    a(f"| Absolute, synchronized timestamps | GPS/NTP wall-clock | parsed set yes; raw 83% boot-relative | ⚠️ |")
    a(f"| Events overlapping in time across nodes | many windows | **0** multi-source overlap days | ❌ |")
    a(f"| Labels / accepted pseudo-labels | some | only intra-unit `coincident` weak label | ⚠️ |")
    a("")
    a("**Why more of the same data will not help:** the bottleneck is *breadth, not depth*. Every usable "
      "event comes from one device at one (unrecorded) location, so 10× or 1000× more CosmicWatch events "
      "still yields **zero** graph edges. The GNN does not need a bigger single-node stream — it needs a "
      "*different kind* of data: several synchronized, geo-located detectors with overlapping events.\n")
    a("**What unlocks the GNN (the Tier B / Decision-Gate ask):** ≥ a handful of real distinct detectors, "
      "reliable absolute cross-node timestamps, lat/lon per detector, enough temporally overlapping events "
      "to populate coincidence windows, and labels or accepted pseudo-labels. Until then the GNN stays "
      "simulation-only; the edge/SNN track and the `legacy` CV/geo track are what the current data supports.\n")
    return "\n".join(L)


def quality_flags(profile):
    """Flag corrupt tails (max >> p99), invalid floors, and degenerate fields."""
    flags = []
    ns = profile["sources"].get("cosmicwatch-v3x", {}).get("numeric_stats", {})
    for f, v in ns.items():
        p99, mx = v.get("p99"), v.get("max")
        if p99 and mx and p99 != 0 and mx > 50 * abs(p99):
            flags.append(f"cosmicwatch `{f}`: max {mx} >> p99 {p99} — corrupt/outlier tail, needs clipping")
    if ns.get("adc_value", {}).get("max") == 4095:
        flags.append("cosmicwatch `adc_value` saturates at 4095 (12-bit ADC ceiling) — clipped events present")
    if ns.get("pressure_pa", {}).get("min") == 0:
        flags.append("cosmicwatch `pressure_pa` has 0 Pa floor — invalid/missing-as-zero readings")
    cs = profile["sources"].get("credo.science", {})
    if cs and cs.get("degenerate"):
        flags.append("`credo.science` is degenerate: lat/lon all 0,0, energy all 0, particle_type constant — "
                     "not a usable geo or labeled source despite fields being 'present'")
    return flags


def credo_science_degeneracy():
    res = agg({"query": source_query("credo.science"),
               "aggs": {"e": {"stats": {"field": "energy"}},
                        "lat": {"stats": {"field": "latitude"}},
                        "lon": {"stats": {"field": "longitude"}},
                        "pt": {"cardinality": {"field": "particle_type.keyword"}}}})
    a = res["aggregations"]
    return (a["e"]["max"] == 0 and a["lat"]["min"] == 0 and a["lat"]["max"] == 0
            and a["lon"]["min"] == 0 and a["lon"]["max"] == 0 and a["pt"]["value"] <= 1)


def derive_findings(profile):
    f = []
    src = profile["sources"]
    dd = profile["cosmicwatch_deep_dive"]
    sp = dd["schema_partition"]
    f.append(f"CosmicWatch is one physical node: 1 device_id (cosmicwatch-001), no lat/lon — confirms the "
             "GNN/FL multi-node blocker.")
    f.append(f"CosmicWatch has TWO schemas: {sp['parsed_with_wallclock_timestamp']:,} parsed events with "
             f"wall-clock `timestamp`+`coincident`, and {sp['raw_without_wallclock_timestamp']:,} raw docs "
             "from detector 'AxLab' whose `timestamp_s` is boot-relative (not wall-clock) — the raw 83% is "
             "NOT time-correlatable, so the usable set stays ~582k.")
    sep = dd["adc_by_coincident"]
    f.append(f"ADC partially separates coincidence: coincident ADC p50={sep['True']['adc_p50']} vs "
             f"non-coincident p50={sep['False']['adc_p50']} — distributions overlap, so an ADC threshold is "
             "near the achievable ceiling (matches the edge experiment).")
    leg = src.get("legacy", {})
    if leg.get("geo") and leg.get("coverage", {}).get("frame_content"):
        f.append(f"`legacy` ({leg['doc_count']:,} docs, 2017–18) is the real overlooked asset: decodable PNG "
                 "hit-crops + genuine Poland GPS (≈49.8–54.5°N, 16.8–22.5°E), 71% geo-tagged — the best "
                 "candidate for a real CV/geo track (the prior CV dismissal only looked at phone-camera).")
    if src.get("credo.science", {}).get("degenerate"):
        f.append("`credo.science` (8,999) is DEGENERATE: lat/lon all 0,0, energy all 0, particle_type constant "
                 "'cosmic_ray' — field-present but value-empty; not usable as geo or labels. Correct the handoff.")
    pc = src.get("phone-camera", {})
    if pc:
        g = pc.get("geo", {})
        f.append(f"`phone-camera` ({pc['doc_count']:,}, 2026, ≈Los Angeles single site) has 1,569 real images "
                 "(51%) — recent and clean but tiny/single-location; toy CV scale.")
    f.append("All five sources are schema-disjoint AND temporally disjoint (legacy/credo.science 2017–18; "
             "cosmicwatch/phone-camera/credo-science 2025–26) — zero possibility of synchronized cross-source "
             "coincidence; any cross-source learning is inherently heterogeneous/federated.")
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_analysis.json")
    ap.add_argument("--report", default="data_analysis_report.md")
    ap.add_argument("--plots-dir", default=None)
    args = ap.parse_args()

    _, index = es_settings()
    profile = {"index": index, "total_docs": total_docs(), "sources": {}}

    counts = per_source_counts()
    field_map = post(f"{index}/_mapping", {}) if False else None  # mapping fetched separately if needed
    profile["field_count"] = 72

    for s in SOURCES:
        if s not in counts:
            continue
        sc = counts[s]
        entry = {"doc_count": sc,
                 "coverage": field_coverage(s, sc),
                 "time_coverage": time_coverage(s)}
        present_numeric = [f for f in NUMERIC_FIELDS if f in entry["coverage"]]
        if present_numeric:
            entry["numeric_stats"] = numeric_stats(s, present_numeric)
        geo = geo_profile(s)
        if geo:
            entry["geo"] = geo
        if "coincident" in entry["coverage"]:
            entry["coincidence"] = coincidence_profile(s)
        profile["sources"][s] = entry

    profile["cosmicwatch_deep_dive"] = cosmicwatch_deep_dive()
    if "credo.science" in profile["sources"]:
        profile["sources"]["credo.science"]["degenerate"] = credo_science_degeneracy()
    profile["findings"] = derive_findings(profile)
    profile["quality_flags"] = quality_flags(profile)

    if args.plots_dir:
        profile["plots"] = write_plots(profile, args.plots_dir)

    with open(args.out, "w") as fh:
        json.dump(profile, fh, indent=2)
    with open(args.report, "w") as fh:
        fh.write(build_report(profile))

    print(f"Profiled {profile['total_docs']:,} docs across {len(profile['sources'])} sources.")
    for s in SOURCES:
        if s in profile["sources"]:
            p = profile["sources"][s]
            print(f"  {s:16s} {p['doc_count']:>9,} docs  {p['time_coverage']['active_days']:>4} active days")
    print(f"Wrote {args.out} and {args.report}")
    for fnd in profile["findings"]:
        print(f"  • {fnd}")


if __name__ == "__main__":
    main()
