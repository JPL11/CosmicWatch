#!/usr/bin/env python3
"""Daily detector-health baselines and alerts from existing CosmicWatch records."""
import argparse
import collections
import datetime as dt
import json
import time

import numpy as np

from cosmicwatch_common import iter_cosmicwatch, utc_day


def robust_z(values):
    values = np.asarray(values, float); median = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - median))
    if scale < 1e-9: scale = np.std(values) or 1.0
    return (values - median) / scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="credo_useful.csv")
    ap.add_argument("--as-of", default="2026-07-15")
    ap.add_argument("--out", default="detector_health.json")
    ap.add_argument("--report", default="detector_health_report.md")
    args = ap.parse_args(); started = time.time()
    groups = collections.defaultdict(list)
    for row in iter_cosmicwatch(args.csv):
        if row["time_epoch_s"] is not None and row["adc_value"] is not None:
            groups[(utc_day(row["time_epoch_s"]), row["partition"])].append(row)
    daily = []
    for (day, partition), rows in sorted(groups.items()):
        times = np.asarray([r["time_epoch_s"] for r in rows])
        adc = np.asarray([r["adc_value"] for r in rows])
        coinc = np.asarray([r["coincident"] for r in rows])
        span = max(1.0, times.max() - times.min())
        daily.append({"day": day, "partition": partition, "events": len(rows),
                      "active_span_h": round(span / 3600, 3), "rate_hz": len(rows) / span,
                      "adc_p50": float(np.median(adc)), "adc_p90": float(np.quantile(adc, .9)),
                      "coincidence_rate": float(coinc.mean()), "saturation_rate": float((adc >= 4095).mean())})
    alerts = []
    for partition in sorted({row["partition"] for row in daily}):
        subset = [row for row in daily if row["partition"] == partition and row["events"] >= 100]
        for metric in ("rate_hz", "adc_p50", "coincidence_rate", "saturation_rate"):
            z = robust_z([row[metric] for row in subset])
            for row, value in zip(subset, z):
                row[f"{metric}_robust_z"] = round(float(value), 2)
                if abs(value) >= 3:
                    alerts.append({"type": "metric_outlier", "day": row["day"], "partition": partition,
                                   "metric": metric, "value": round(row[metric], 6), "robust_z": round(float(value), 2)})
            adjacent = []
            for previous, current in zip(subset, subset[1:]):
                gap = (dt.date.fromisoformat(current["day"]) - dt.date.fromisoformat(previous["day"])).days
                if gap == 1:
                    adjacent.append((current, current[metric] - previous[metric]))
            if len(adjacent) >= 5:
                delta_z = robust_z([delta for _, delta in adjacent])
                for (row, delta), value in zip(adjacent, delta_z):
                    if abs(value) >= 4:
                        alerts.append({"type": "daily_change_point", "day": row["day"], "partition": partition,
                                       "metric": metric, "delta": round(delta, 6),
                                       "robust_delta_z": round(float(value), 2)})
    latest = max(row["day"] for row in daily)
    offline_days = (dt.date.fromisoformat(args.as_of) - dt.date.fromisoformat(latest)).days
    if offline_days > 1:
        alerts.append({"type": "ingestion_stale", "latest_event_day": latest, "as_of": args.as_of,
                       "days_without_data": offline_days})
    partition_summary = {}
    for partition in sorted({row["partition"] for row in daily}):
        subset = [row for row in daily if row["partition"] == partition]
        partition_summary[partition] = {"days": len(subset), "events": sum(r["events"] for r in subset),
                                        "median_adc_p50": round(float(np.median([r["adc_p50"] for r in subset])), 3),
                                        "median_coincidence_rate": round(float(np.median([r["coincidence_rate"] for r in subset])), 5)}
    result = {"as_of": args.as_of, "latest_event_day": latest, "offline_days": offline_days,
              "partition_summary": partition_summary, "daily": daily, "alerts": alerts,
              "limitations": ["Daily outliers are screening alerts, not diagnosed hardware failures.",
                              "Sparse/partial acquisition days can distort rate."],
              "runtime_seconds": round(time.time() - started, 2)}
    with open(args.out, "w") as fh: json.dump(result, fh, indent=2)
    with open(args.report, "w") as fh:
        fh.write("# Detector Health Audit\n\n")
        fh.write(f"Latest event day: {latest}; stale by {offline_days} days as of {args.as_of}.\n\n")
        fh.write(f"Generated {len(alerts)} robust screening alerts across {len(daily)} active partition-days.\n\n")
        for partition, summary in partition_summary.items(): fh.write(f"- {partition}: {summary}\n")
        fh.write("\nSee JSON for daily metrics and alert details.\n")
    print(f"Wrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()
