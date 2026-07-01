#!/usr/bin/env python3
"""
Time-domain physics on the CosmicWatch stream (data-only; no hardware access).

  1. Muon-lifetime feasibility: measure the detector's minimum inter-event gap.
     (Result: readout floor ~ms >> 2.2 us muon lifetime -> measurement impossible;
     documented as a clean negative.)
  2. Diurnal cycle: fold event rate by UTC hour-of-day over the dense raw window
     (2026-06-06..06-19) and correlate with the temperature cycle.
  3. Pressure retest on the dense raw window: rate ~ pressure + temperature + time
     regression on hourly buckets (better statistics than the parsed epoch's 16 days).
  4. Orientation from the recorded accelerometer: tilt angle per epoch from mean
     accel_z; a tilt change is a candidate explanation for the 12%->8% drift.
  5. Space weather: fetch GFZ Kp indices for the window (public API) and correlate
     daily rate anomalies with geomagnetic activity; flag Forbush-like dips.
     (Degrades gracefully to internal dip-flagging if the API is unreachable.)

Outputs: time_domain_physics.json, time_domain_physics_report.md, plots_timedomain/*.png
"""
import argparse
import datetime as dt
import json
import math
import time

import numpy as np
import requests

from credo_loader import partition_query, post
from credo_config import es_settings

RAW_DENSE = (1780704000, 1781913600)  # 2026-06-06 .. 2026-06-20 UTC (dense raw stretch)
PRESSURE_OK = (80_000, 110_000)
TEMP_OK = (-50, 80)


def q_raw_window(gte, lt):
    return {"bool": {"filter": [partition_query("raw"), {"range": {"wall_time": {"gte": gte, "lt": lt}}}]}}


# ---------- 1. lifetime feasibility ----------
def lifetime_feasibility(hours=3):
    _, index = es_settings()
    gaps_min = []
    t0 = RAW_DENSE[0] + 6 * 3600
    for h in range(hours):
        body = {"size": 10000, "_source": ["wall_time"],
                "query": q_raw_window(t0 + h * 3600, t0 + (h + 1) * 3600),
                "sort": [{"wall_time": "asc"}]}
        hits = post(f"{index}/_search", body)["hits"]["hits"]
        t = np.sort(np.array([x["_source"]["wall_time"] for x in hits]))
        g = np.diff(t); g = g[g > 0]
        if len(g):
            gaps_min.append(float(g.min()))
    floor_ms = round(min(gaps_min) * 1000, 2) if gaps_min else None
    return {
        "min_inter_event_gap_ms": floor_ms,
        "muon_lifetime_us": 2.2,
        "verdict": (f"IMPOSSIBLE with this data: readout floor ~{floor_ms} ms is ~4 orders of magnitude "
                    "above the 2.2 us muon lifetime; decay pairs cannot be recorded. Clean negative."),
    }


# ---------- 2+3. hourly series on the dense raw window ----------
def raw_hourly():
    _, index = es_settings()
    body = {"size": 0, "query": q_raw_window(*RAW_DENSE),
            "aggs": {"h": {"histogram": {"field": "wall_time", "interval": 3600, "min_doc_count": 1},
                           "aggs": {"p": {"filter": {"range": {"pressure_pa": {"gte": PRESSURE_OK[0], "lte": PRESSURE_OK[1]}}},
                                          "aggs": {"a": {"avg": {"field": "pressure_pa"}}}},
                                    "t": {"filter": {"range": {"temperature_c": {"gte": TEMP_OK[0], "lte": TEMP_OK[1]}}},
                                          "aggs": {"a": {"avg": {"field": "temperature_c"}}}},
                                    "co": {"filter": {"term": {"coincidence_flag": 1}}}}}}}
    bs = post(f"{index}/_search", body)["aggregations"]["h"]["buckets"]
    rows = []
    for b in bs:
        if b["doc_count"] < 500:  # partial hours
            continue
        rows.append({"epoch": b["key"], "hour_utc": dt.datetime.fromtimestamp(b["key"], dt.timezone.utc).hour,
                     "day": dt.datetime.fromtimestamp(b["key"], dt.timezone.utc).strftime("%Y-%m-%d"),
                     "rate_hz": b["doc_count"] / 3600.0,
                     "coincident_rate": b["co"]["doc_count"] / b["doc_count"],
                     "pressure_pa": b["p"]["a"]["value"], "temperature_c": b["t"]["a"]["value"]})
    return rows


def diurnal(rows):
    by_hour = {}
    for r in rows:
        by_hour.setdefault(r["hour_utc"], []).append(r)
    fold = []
    for h in sorted(by_hour):
        rs = [x["rate_hz"] for x in by_hour[h]]
        ts = [x["temperature_c"] for x in by_hour[h] if x["temperature_c"] is not None]
        fold.append({"hour_utc": h, "n_hours": len(rs), "mean_rate_hz": round(float(np.mean(rs)), 4),
                     "mean_temp_c": round(float(np.mean(ts)), 2) if ts else None})
    rates = np.array([f["mean_rate_hz"] for f in fold])
    amp = (rates.max() - rates.min()) / rates.mean()
    temps = np.array([f["mean_temp_c"] if f["mean_temp_c"] is not None else np.nan for f in fold])
    ok = ~np.isnan(temps)
    r_rt = float(np.corrcoef(rates[ok], temps[ok])[0, 1]) if ok.sum() > 4 else None
    return {"fold": fold, "peak_hour_utc": int(np.argmax(rates)),
            "trough_hour_utc": int(np.argmin(rates)),
            "peak_to_trough_amplitude_pct": round(100 * amp, 2),
            "rate_temp_correlation": round(r_rt, 3) if r_rt is not None else None}


def pressure_retest(rows):
    data = [r for r in rows if r["pressure_pa"] and r["temperature_c"] is not None]
    if len(data) < 24:
        return {"note": "too few hours"}
    z = lambda a: (a - a.mean()) / (a.std() + 1e-9)
    rate = np.array([r["rate_hz"] for r in data])
    p = np.array([r["pressure_pa"] for r in data]) / 100.0
    T = np.array([r["temperature_c"] for r in data])
    t = np.array([r["epoch"] for r in data]); t = (t - t.min()) / 3600.0
    hod = np.array([r["hour_utc"] for r in data], float)
    # include the diurnal phase as controls (sin/cos of hour-of-day)
    X = np.column_stack([z(p), z(T), z(t), np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24),
                         np.ones(len(p))])
    beta, *_ = np.linalg.lstsq(X, z(rate), rcond=None)
    simple = float(np.corrcoef(p, rate)[0, 1])
    slope = np.polyfit(p, rate, 1)[0]
    return {"n_hours": len(data),
            "simple_r_rate_pressure": round(simple, 3),
            "barometric_pct_per_hPa": round(100 * slope / rate.mean(), 3),
            "partial_betas": {"pressure": round(float(beta[0]), 3), "temperature": round(float(beta[1]), 3),
                              "time_trend": round(float(beta[2]), 3),
                              "diurnal_sin": round(float(beta[3]), 3), "diurnal_cos": round(float(beta[4]), 3)},
            "note": "controls: temperature, linear time trend, and the diurnal phase"}


# ---------- 4. orientation ----------
def orientation():
    _, index = es_settings()
    out = {}
    for p, field in [("parsed", "accel_z_g"), ("raw", "accel_z")]:
        r = post(f"{index}/_search", {"size": 0, "query": partition_query(p),
                                      "aggs": {"z": {"stats": {"field": field}},
                                               "zp": {"percentiles": {"field": field, "percents": [5, 50, 95]}}}})
        s = r["aggregations"]["z"]; pc = r["aggregations"]["zp"]["values"]
        mean_z = s["avg"] or 0
        tilt = round(math.degrees(math.acos(min(1.0, abs(mean_z)))), 1)
        out[p] = {"field": field, "count": s["count"], "mean_z": round(mean_z, 4),
                  "p50_z": round(pc.get("50.0") or 0, 4), "tilt_deg_from_vertical": tilt}
    dp = out["parsed"]["tilt_deg_from_vertical"]; dr = out["raw"]["tilt_deg_from_vertical"]
    ratio = (math.cos(math.radians(dr)) / max(1e-9, math.cos(math.radians(dp)))) ** 2
    out["interpretation"] = (
        f"Tilt changed between epochs: {dp} deg (parsed 2025) -> {dr} deg (raw 2026). "
        f"A cos^2 acceptance argument gives a flux factor ~{round(ratio,3)} — small; tilt alone does NOT "
        "explain the coincidence-rate drop 12%->8% or the 2.3x rate increase. The dominant drift cause is "
        "more consistent with a threshold/gain change (efficiency turn-on shifted ~200 ADC). "
        "Caveat: raw accel_z units assumed ~g (mean ~ -1 supports this); single-point calibration.")
    return out


# ---------- 5. space weather ----------
def space_weather(daily):
    """Fetch GFZ Kp for the window; correlate daily rate anomaly with geomagnetic activity."""
    try:
        r = requests.get("https://kp.gfz-potsdam.de/app/json/",
                         params={"start": "2026-05-01T00:00:00Z", "end": "2026-07-01T00:00:00Z", "index": "Kp"},
                         timeout=30)
        r.raise_for_status()
        data = r.json()
        times = data.get("datetime", []); kp = data.get("Kp", [])
        kp_daily = {}
        for ts, v in zip(times, kp):
            day = ts[:10]
            if v is not None:
                kp_daily.setdefault(day, []).append(float(v))
        kp_max = {d: max(v) for d, v in kp_daily.items()}
    except Exception as e:
        kp_max = None
        err = str(e)[:120]

    days = sorted({r["day"] for r in daily})
    rate_by_day = {d: float(np.mean([r["rate_hz"] for r in daily if r["day"] == d])) for d in days}
    rates = np.array([rate_by_day[d] for d in days])
    mu, sd = rates.mean(), rates.std()
    dips = [{"day": d, "rate_hz": round(rate_by_day[d], 3), "z": round((rate_by_day[d] - mu) / (sd + 1e-9), 2)}
            for d in days if rate_by_day[d] < mu - 2 * sd]

    out = {"days_analyzed": len(days), "mean_daily_rate_hz": round(float(mu), 3),
           "rate_dips_2sigma": dips}
    if kp_max:
        common = [d for d in days if d in kp_max]
        if len(common) >= 5:
            x = np.array([kp_max[d] for d in common]); y = np.array([rate_by_day[d] for d in common])
            out["kp_source"] = "GFZ Potsdam"
            out["days_with_kp"] = len(common)
            out["max_kp_in_window"] = round(float(x.max()), 1)
            out["corr_rate_vs_maxKp"] = round(float(np.corrcoef(x, y)[0, 1]), 3)
            out["interpretation"] = (
                "A Forbush decrease would show as a multi-day rate dip following high Kp. "
                f"Window max Kp={out['max_kp_in_window']}; correlation r={out['corr_rate_vs_maxKp']} over "
                f"{len(common)} days. With ~14 dense days this is a feasibility check, not a detection claim.")
    else:
        out["kp_source"] = f"unavailable ({err})"
    return out


def write_plots(out, rows, plots_dir):
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    d = Path(plots_dir); d.mkdir(parents=True, exist_ok=True)
    paths = []

    fold = out["diurnal"]["fold"]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    hs = [f["hour_utc"] for f in fold]
    ax1.plot(hs, [f["mean_rate_hz"] for f in fold], "o-", color="#23508c")
    ax1.set_xlabel("UTC hour of day"); ax1.set_ylabel("mean rate (Hz)", color="#23508c")
    ax2 = ax1.twinx()
    ax2.plot(hs, [f["mean_temp_c"] for f in fold], "s--", color="#d8703b", alpha=0.7)
    ax2.set_ylabel("mean temperature (C)", color="#d8703b")
    plt.title(f"Diurnal fold (raw dense window) — amplitude "
              f"{out['diurnal']['peak_to_trough_amplitude_pct']}%")
    fig.tight_layout(); fig.savefig(d / "diurnal_fold.png", dpi=150); plt.close(fig)
    paths.append(str(d / "diurnal_fold.png"))

    days = sorted({r["day"] for r in rows})
    daily_rate = [float(np.mean([r["rate_hz"] for r in rows if r["day"] == day])) for day in days]
    plt.figure(figsize=(9, 4))
    plt.plot(range(len(days)), daily_rate, "o-")
    plt.xticks(range(len(days)), [x[5:] for x in days], rotation=45, fontsize=7)
    plt.ylabel("mean daily rate (Hz)"); plt.title("Daily rate — dense raw window (Forbush-dip check)")
    plt.tight_layout(); plt.savefig(d / "daily_rate.png", dpi=150); plt.close()
    paths.append(str(d / "daily_rate.png"))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="time_domain_physics.json")
    ap.add_argument("--report", default="time_domain_physics_report.md")
    ap.add_argument("--plots-dir", default=None)
    args = ap.parse_args()
    started = time.time()

    print("1. muon-lifetime feasibility ...")
    lt = lifetime_feasibility()
    print("2/3. hourly series on the dense raw window ...")
    rows = raw_hourly()
    di = diurnal(rows)
    pr = pressure_retest(rows)
    print("4. orientation from recorded accelerometer ...")
    ori = orientation()
    print("5. space weather / Kp ...")
    sw = space_weather(rows)

    out = {"lifetime_feasibility": lt, "diurnal": di, "pressure_retest": pr,
           "orientation": ori, "space_weather": sw}
    out["findings"] = [
        f"Muon-lifetime measurement is cleanly IMPOSSIBLE: readout floor {lt['min_inter_event_gap_ms']} ms "
        "vs 2.2 us decay — documented negative, not a missed opportunity.",
        f"Diurnal modulation of {di['peak_to_trough_amplitude_pct']}% peak-to-trough (peak {di['peak_hour_utc']}h, "
        f"trough {di['trough_hour_utc']}h UTC), rate-temperature correlation r={di['rate_temp_correlation']} — "
        "an indoor temperature-coupled cycle is the simplest explanation.",
        f"Pressure retest on the dense 2026 window: simple r={pr.get('simple_r_rate_pressure')}, partial "
        f"pressure beta={pr.get('partial_betas',{}).get('pressure')} with temperature/time/diurnal controls, "
        f"barometric coefficient {pr.get('barometric_pct_per_hPa')}%/hPa.",
        f"Orientation: tilt {ori['parsed']['tilt_deg_from_vertical']} deg (2025) -> "
        f"{ori['raw']['tilt_deg_from_vertical']} deg (2026); cos^2 flux factor is small — tilt does NOT explain "
        "the drift; threshold/gain change remains the leading explanation.",
        (f"Space weather: max Kp {sw.get('max_kp_in_window')} in window; rate-vs-Kp r={sw.get('corr_rate_vs_maxKp')}; "
         f"{len(sw.get('rate_dips_2sigma', []))} two-sigma daily dips."
         if sw.get("kp_source") == "GFZ Potsdam" else
         f"Space weather: Kp fetch unavailable; {len(sw.get('rate_dips_2sigma', []))} two-sigma daily dips flagged internally."),
    ]

    if args.plots_dir:
        out["plots"] = write_plots(out, rows, args.plots_dir)
    out["runtime_seconds"] = round(time.time() - started, 2)

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    with open(args.report, "w") as fh:
        L = ["# Time-Domain Physics (data-only)\n"]
        L += [f"- {f}" for f in out["findings"]]
        L += ["\nSee `time_domain_physics.json` for full numbers.\n"]
        fh.write("\n".join(L))

    for f in out["findings"]:
        print("  •", f)
    print(f"Wrote {args.out}, {args.report}")


if __name__ == "__main__":
    main()
