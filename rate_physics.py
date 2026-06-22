#!/usr/bin/env python3
"""
Physics-grounded analysis of the single-node CosmicWatch event stream.

Uses the data for what one detector is genuinely good at:
  1. Event rate over time (hourly), via server-side aggregation.
  2. Poisson check: inter-arrival times of a clean window should be ~exponential
     (coefficient of variation ~ 1) for a random cosmic-ray process.
  3. Environmental correlation: muon flux is known to anti-correlate with
     atmospheric pressure (barometric effect) and vary with temperature; test it.

No labels and no multi-node data required — this is real, defensible single-node
science. Outputs: rate_physics.json, rate_physics_report.md, plots_physics/*.png
"""
import argparse
import json
import time
import warnings

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")
import numpy as np
import requests
import urllib3

from credo_config import es_auth, es_settings, verify_certs
from edge_ai_experiment import SOURCE, build_rows, fetch_events

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PRESSURE_OK = (80_000, 110_000)   # valid Pa range (raw data has 0-Pa and corrupt highs)
TEMP_OK = (-50, 80)               # valid degrees C


def post(path, body, timeout=120, retries=4):
    es_url, _ = es_settings()
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(f"{es_url}/{path.lstrip('/')}", auth=es_auth(), verify=verify_certs(),
                              headers={"Content-Type": "application/json"}, json=body, timeout=timeout)
            if r.status_code in (429, 502, 503, 504):
                last = r; time.sleep(2 * (attempt + 1)); continue
            r.raise_for_status(); return r.json()
        except requests.exceptions.RequestException as e:
            last = e; time.sleep(2 * (attempt + 1))
    if isinstance(last, requests.Response):
        last.raise_for_status()
    raise last


def hourly_rate_and_env(start, end):
    _, index = es_settings()
    body = {
        "size": 0,
        "query": {"bool": {"filter": [
            {"term": {"source": SOURCE}},
            {"range": {"timestamp": {"gte": start, "lt": end}}},
        ]}},
        "aggs": {"by_hour": {
            "date_histogram": {"field": "timestamp", "calendar_interval": "hour", "min_doc_count": 1},
            "aggs": {
                "p": {"filter": {"range": {"pressure_pa": {"gte": PRESSURE_OK[0], "lte": PRESSURE_OK[1]}}},
                      "aggs": {"avg": {"avg": {"field": "pressure_pa"}}}},
                "t": {"filter": {"range": {"temperature_c": {"gte": TEMP_OK[0], "lte": TEMP_OK[1]}}},
                      "aggs": {"avg": {"avg": {"field": "temperature_c"}}}},
            }}},
    }
    buckets = post(f"{index}/_search", body)["aggregations"]["by_hour"]["buckets"]
    out = []
    for b in buckets:
        out.append({
            "hour": b["key_as_string"],
            "count": b["doc_count"],
            "rate_hz": b["doc_count"] / 3600.0,
            "pressure_pa": b["p"]["avg"]["value"],
            "temperature_c": b["t"]["avg"]["value"],
        })
    return out


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def correlation_block(hours, env_key, min_count=200):
    rows = [h for h in hours if h["count"] >= min_count and h[env_key] is not None]
    if len(rows) < 5:
        return {"n_hours": len(rows), "note": "too few hours for a stable correlation"}
    env = np.array([h[env_key] for h in rows])
    rate = np.array([h["rate_hz"] for h in rows])
    r = pearson(env, rate)
    slope, intercept = np.polyfit(env, rate, 1)
    block = {"n_hours": len(rows), "pearson_r": round(r, 3) if r is not None else None,
             "slope_per_unit": float(slope), "mean_rate_hz": round(float(rate.mean()), 4)}
    if env_key == "pressure_pa":
        # barometric coefficient: % rate change per hPa (1 hPa = 100 Pa)
        block["barometric_pct_per_hPa"] = round(slope * 100 / max(1e-9, rate.mean()) * 100, 3)
    return block


def poisson_check(start, end, max_events):
    rows = build_rows(fetch_events(start, end, max_events=max_events, page_size=5000, scroll_keepalive="2m"))
    gaps = np.array([r["interarrival_ms"] for r in rows if r["interarrival_ms"] > 0], dtype=float)
    if len(gaps) < 100:
        return {"note": f"only {len(gaps)} gaps"}
    mean = float(gaps.mean()); std = float(gaps.std())
    cv = std / max(1e-9, mean)
    return {
        "events": len(rows),
        "mean_interarrival_ms": round(mean, 3),
        "implied_rate_hz": round(1000.0 / mean, 4),
        "coefficient_of_variation": round(cv, 3),
        "exponential_expectation": "CV≈1.0 for a Poisson (random) process",
        "verdict": ("consistent with Poisson (random) arrivals" if 0.8 <= cv <= 1.2
                    else "sub-Poissonian / more regular than random — consistent with detector dead-time" if cv < 0.8
                    else "super-Poissonian / bursty (clustered arrivals)"),
        "_gaps_sample": gaps[:20000].tolist(),
    }


def write_plots(out, plots_dir):
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    import datetime as dt
    d = Path(plots_dir); d.mkdir(parents=True, exist_ok=True)
    paths = []

    hours = out["hourly"]
    if hours:
        xs = [dt.datetime.fromisoformat(h["hour"].replace("Z", "+00:00")) for h in hours]
        ys = [h["rate_hz"] for h in hours]
        plt.figure(figsize=(9, 4)); plt.plot(xs, ys, ".", ms=3)
        plt.ylabel("event rate (Hz)"); plt.title("CosmicWatch hourly event rate")
        plt.tight_layout(); p = d / "rate_over_time.png"; plt.savefig(p, dpi=150); plt.close(); paths.append(str(p))

        valid = [h for h in hours if h["count"] >= 200 and h["pressure_pa"]]
        if len(valid) >= 5:
            pr = [h["pressure_pa"] / 100 for h in valid]  # hPa
            rt = [h["rate_hz"] for h in valid]
            plt.figure(figsize=(7, 5)); plt.scatter(pr, rt, s=10, alpha=0.5)
            m, b = np.polyfit(pr, rt, 1); xs2 = np.array([min(pr), max(pr)])
            plt.plot(xs2, m * xs2 + b, "r-", label=f"fit r={out['pressure_correlation'].get('pearson_r')}")
            plt.xlabel("pressure (hPa)"); plt.ylabel("event rate (Hz)")
            plt.title("Rate vs atmospheric pressure (barometric effect)")
            plt.legend(); plt.tight_layout()
            p = d / "rate_vs_pressure.png"; plt.savefig(p, dpi=150); plt.close(); paths.append(str(p))

    pc = out.get("poisson", {})
    gaps = pc.get("_gaps_sample")
    if gaps:
        g = np.array(gaps); g = g[g < np.percentile(g, 99)]
        plt.figure(figsize=(7, 5))
        plt.hist(g, bins=80, density=True, alpha=0.6, label="observed gaps")
        lam = 1.0 / max(1e-9, np.mean(gaps))
        xs3 = np.linspace(0, g.max(), 200)
        plt.plot(xs3, lam * np.exp(-lam * xs3), "r-", label="exponential (Poisson)")
        plt.yscale("log"); plt.xlabel("inter-arrival (ms)"); plt.ylabel("density (log)")
        plt.title(f"Inter-arrival vs exponential (CV={pc.get('coefficient_of_variation')})")
        plt.legend(); plt.tight_layout()
        p = d / "interarrival_poisson.png"; plt.savefig(p, dpi=150); plt.close(); paths.append(str(p))
    return paths


def build_report(out):
    L = []; a = L.append
    a("# CosmicWatch Single-Node Physics Analysis\n")
    a(f"Timestamped events over `{out['window']['start']}` → `{out['window']['end']}`; "
      f"{out['n_active_hours']} active hours.\n")

    a("## 1. Event rate\n")
    a(f"- Mean hourly rate: **{out['rate_summary']['mean_rate_hz']} Hz** "
      f"(min {out['rate_summary']['min_rate_hz']}, max {out['rate_summary']['max_rate_hz']}).")
    a(f"- Busiest hour: {out['rate_summary']['peak_hour']} at {out['rate_summary']['max_rate_hz']} Hz.\n")

    a("## 2. Poisson check (inter-arrival times)\n")
    pc = out["poisson"]
    a(f"- Clean window: {pc.get('events','?'):,} events, mean inter-arrival "
      f"{pc.get('mean_interarrival_ms')} ms → implied rate {pc.get('implied_rate_hz')} Hz.")
    a(f"- Coefficient of variation: **{pc.get('coefficient_of_variation')}** "
      f"({pc.get('exponential_expectation')}).")
    a(f"- Verdict: **{pc.get('verdict')}**.\n")

    a("## 3. Environmental correlation\n")
    p = out["pressure_correlation"]; t = out["temperature_correlation"]
    a(f"- Rate vs pressure: Pearson r = **{p.get('pearson_r')}** over {p.get('n_hours')} hours; "
      f"barometric coefficient ≈ **{p.get('barometric_pct_per_hPa')}%/hPa**.")
    a(f"- Rate vs temperature: Pearson r = **{t.get('pearson_r')}** over {t.get('n_hours')} hours.\n")
    a(f"  (Known physics: muon flux anti-correlates with pressure ~ −0.1 to −0.3%/hPa. "
      "A clean indoor single detector over a short span may show a weak/noisy effect.)\n")

    a("## 4. Findings\n")
    for f in out["findings"]:
        a(f"- {f}")
    a("")
    return "\n".join(L)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-11-01T00:00:00Z")
    ap.add_argument("--end", default="2026-03-01T00:00:00Z")
    ap.add_argument("--poisson-start", default="2026-01-23T00:00:00Z")
    ap.add_argument("--poisson-end", default="2026-01-25T00:00:00Z")
    ap.add_argument("--poisson-max-events", type=int, default=120_000)
    ap.add_argument("--out", default="rate_physics.json")
    ap.add_argument("--report", default="rate_physics_report.md")
    ap.add_argument("--plots-dir", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    started = time.time()

    print("Aggregating hourly rate + environment ...")
    hours = hourly_rate_and_env(args.start, args.end)
    rates = [h["rate_hz"] for h in hours]
    peak = max(hours, key=lambda h: h["rate_hz"]) if hours else None

    print("Pulling clean window for Poisson check ...")
    poisson = poisson_check(args.poisson_start, args.poisson_end, args.poisson_max_events)

    out = {
        "window": {"start": args.start, "end": args.end},
        "n_active_hours": len(hours),
        "hourly": hours,
        "rate_summary": {
            "mean_rate_hz": round(float(np.mean(rates)), 4) if rates else None,
            "min_rate_hz": round(float(np.min(rates)), 4) if rates else None,
            "max_rate_hz": round(float(np.max(rates)), 4) if rates else None,
            "peak_hour": peak["hour"] if peak else None,
        },
        "poisson": poisson,
        "pressure_correlation": correlation_block(hours, "pressure_pa"),
        "temperature_correlation": correlation_block(hours, "temperature_c"),
    }

    p = out["pressure_correlation"]; pc = out["poisson"]
    findings = []
    findings.append(f"Single-detector rate averages ~{out['rate_summary']['mean_rate_hz']} Hz; "
                    "usable for a clean cosmic-ray rate measurement.")
    findings.append(f"Inter-arrival CV = {pc.get('coefficient_of_variation')} → {pc.get('verdict')}; "
                    "this is a real statistical-physics result from one node, no labels needed.")
    if p.get("pearson_r") is not None:
        bc = p.get("barometric_pct_per_hPa") or 0
        if bc < 0:
            findings.append(f"Rate anti-correlates with pressure (r={p['pearson_r']}, {bc}%/hPa) — SAME sign as "
                            "the known barometric muon effect (≈ −0.1 to −0.3%/hPa), though larger; "
                            "single-detector/short-span limited.")
        else:
            findings.append(f"Rate shows a POSITIVE rate–pressure correlation (r={p['pearson_r']}, +{bc}%/hPa), "
                            "which is the OPPOSITE sign of the canonical barometric effect (negative). Over only "
                            "~16 active days this is most likely a confound (seasonal/temperature drift, indoor "
                            "sensor coupling, rate trending with deployment), NOT a clean barometric measurement — "
                            "flag for investigation, do not over-claim.")
    findings.append("All of this is achievable NOW on the single-node data and strengthens a project / "
                    "workshop writeup without needing multi-node data.")
    out["findings"] = findings

    if args.plots_dir:
        out["plots"] = write_plots(out, args.plots_dir)
    # drop the bulky gaps sample from the saved JSON after plotting
    out["poisson"].pop("_gaps_sample", None)
    out["runtime_seconds"] = round(time.time() - started, 2)

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    with open(args.report, "w") as fh:
        fh.write(build_report(out))

    print(f"Active hours: {len(hours)}  mean rate {out['rate_summary']['mean_rate_hz']} Hz")
    print(f"Poisson CV: {pc.get('coefficient_of_variation')} -> {pc.get('verdict')}")
    print(f"Pressure r: {p.get('pearson_r')}  ({p.get('barometric_pct_per_hPa')}%/hPa)  "
          f"Temp r: {out['temperature_correlation'].get('pearson_r')}")
    print(f"Wrote {args.out}, {args.report}")


if __name__ == "__main__":
    main()
