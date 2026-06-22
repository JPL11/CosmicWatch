#!/usr/bin/env python3
"""
Data-readiness probe for the credo-detections index.

Produces, for the upcoming meeting:
  - monthly event counts per source (big-picture timeline)
  - daily counts for the live CosmicWatch detector + GAP MAP (missing days)
  - coincidence rate over time (sanity: is the muon tag stable?)
  - field-population check (which fields are actually filled per source)

Writes data_readiness.json and prints readable tables.
"""
import datetime
import json
import warnings

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")

import urllib3
import requests
from credo_config import es_auth, es_settings, verify_certs

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def search(body):
    es_url, index = es_settings()
    response = requests.post(f"{es_url}/{index}/_search", auth=es_auth(),
                             verify=verify_certs(),
                             headers={"Content-Type": "application/json"},
                             data=json.dumps(body), timeout=180)
    response.raise_for_status()
    return response.json()


def iso(ms):
    return datetime.datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d")


def monthly_by_source():
    res = search({
        "size": 0,
        "aggs": {"m": {
            "date_histogram": {"field": "timestamp", "calendar_interval": "month", "min_doc_count": 1},
            "aggs": {"src": {"terms": {"field": "source", "size": 10}}},
        }},
    })
    out = []
    for b in res["aggregations"]["m"]["buckets"]:
        month = b["key_as_string"][:7]
        srcs = {s["key"]: s["doc_count"] for s in b["src"]["buckets"]}
        out.append({"month": month, "total": b["doc_count"], "sources": srcs})
    return out


def daily_cosmicwatch():
    res = search({
        "size": 0,
        "query": {"term": {"source": "cosmicwatch-v3x"}},
        "aggs": {
            "d": {"date_histogram": {"field": "timestamp", "calendar_interval": "day", "min_doc_count": 1},
                  "aggs": {"coinc": {"filter": {"term": {"coincident": True}}}}},
            "range": {"stats": {"field": "timestamp"}},
        },
    })
    days = []
    for b in res["aggregations"]["d"]["buckets"]:
        day = b["key_as_string"][:10]
        total = b["doc_count"]
        coinc = b["coinc"]["doc_count"]
        days.append({"day": day, "count": total, "coincident": coinc,
                     "coinc_rate": round(100 * coinc / total, 1) if total else 0})
    rng = res["aggregations"]["range"]
    return days, rng


def find_gaps(days):
    """Days with zero events between the first and last active day."""
    if not days:
        return []
    present = {d["day"] for d in days}
    start = datetime.date.fromisoformat(days[0]["day"])
    end = datetime.date.fromisoformat(days[-1]["day"])
    gaps, run = [], None
    d = start
    one = datetime.timedelta(days=1)
    while d <= end:
        s = d.isoformat()
        if s not in present:
            if run is None:
                run = [s, s]
            else:
                run[1] = s
        else:
            if run:
                gaps.append(run)
                run = None
        d += one
    if run:
        gaps.append(run)
    return gaps


def main():
    print("Querying Elasticsearch (a few aggregations, ~30-60s) ...")
    monthly = monthly_by_source()
    cw_days, cw_range = daily_cosmicwatch()
    gaps = find_gaps(cw_days)

    out = {
        "monthly_by_source": monthly,
        "cosmicwatch_daily": cw_days,
        "cosmicwatch_range": {"earliest": iso(cw_range["min"]), "latest": iso(cw_range["max"])},
        "cosmicwatch_gaps": [{"from": g[0], "to": g[1],
                              "days": (datetime.date.fromisoformat(g[1]) -
                                       datetime.date.fromisoformat(g[0])).days + 1}
                             for g in gaps],
    }
    with open("data_readiness.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== MONTHLY EVENTS BY SOURCE ===")
    print(f"{'month':<9}{'total':>10}   sources")
    for m in monthly:
        src = ", ".join(f"{k}:{v:,}" for k, v in sorted(m["sources"].items(), key=lambda x: -x[1]))
        print(f"{m['month']:<9}{m['total']:>10,}   {src}")

    print(f"\n=== COSMICWATCH LIVE DETECTOR: {out['cosmicwatch_range']['earliest']} -> {out['cosmicwatch_range']['latest']} ===")
    print(f"{len(cw_days)} active days")
    print(f"{'day':<12}{'events':>8}{'coinc':>8}{'rate%':>7}")
    for d in cw_days:
        print(f"{d['day']:<12}{d['count']:>8,}{d['coincident']:>8,}{d['coinc_rate']:>7}")

    print(f"\n=== GAPS (missing days inside the active window) ===")
    if not out["cosmicwatch_gaps"]:
        print("none")
    for g in out["cosmicwatch_gaps"]:
        print(f"  {g['from']} -> {g['to']}  ({g['days']} day(s) missing)")

    print("\nWrote data_readiness.json")


if __name__ == "__main__":
    main()
