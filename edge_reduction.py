#!/usr/bin/env python3
"""Simulate event transmission policies using existing CosmicWatch data."""
import argparse
import json
import time

import numpy as np

from cosmicwatch_common import iter_cosmicwatch
from pi_benchmark import forward_np


def clean(row, means):
    values = [row["adc_value"], row["sipm_mv"], row["temperature_c"], row["pressure_pa"]]
    if values[2] is None or not -50 < values[2] < 80: values[2] = means[2]
    if values[3] is None or not 80000 < values[3] < 110000: values[3] = means[3]
    return [means[i] if value is None else value for i, value in enumerate(values)]


def metrics(selected, labels, event_bytes, event_rate):
    selected = np.asarray(selected, bool); labels = np.asarray(labels, bool)
    retained = int(np.sum(selected & labels)); positives = int(labels.sum())
    fraction = float(selected.mean())
    return {
        "transmit_fraction": round(fraction, 5),
        "data_reduction_x": round(1.0 / max(fraction, 1e-12), 2),
        "coincident_recall": round(retained / max(1, positives), 5),
        "coincident_precision": round(retained / max(1, int(selected.sum())), 5),
        "events_per_day": round(fraction * event_rate * 86400, 1),
        "payload_kib_per_day": round(fraction * event_rate * 86400 * event_bytes / 1024, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="credo_useful.csv")
    ap.add_argument("--weights", default="model_weights.json")
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--event-bytes", type=int, default=32)
    ap.add_argument("--event-rate-hz", type=float, default=1.3757)
    ap.add_argument("--out", default="edge_reduction.json")
    ap.add_argument("--report", default="edge_reduction_report.md")
    args = ap.parse_args(); started = time.time()
    with open(args.weights) as fh: weights = json.load(fh)
    means = weights["standardize_mean"]; stds = np.asarray(weights["standardize_std"], np.float32)
    rows = []
    for row in iter_cosmicwatch(args.csv):
        if row["time_epoch_s"] is not None and row["adc_value"] is not None:
            rows.append(row)
            if args.max_events and len(rows) >= args.max_events: break
    rows.sort(key=lambda row: row["time_epoch_s"])
    x = np.asarray([clean(row, means) for row in rows], np.float32)
    labels = np.asarray([row["coincident"] for row in rows], bool)
    split = int(0.7 * len(rows)); train_y, test_y = labels[:split], labels[split:]
    adc_train, adc_test = x[:split, 0], x[split:, 0]
    scores = forward_np((x - np.asarray(means, np.float32)) / stds, weights)
    policies = {"transmit_all": metrics(np.ones(len(test_y), bool), test_y, args.event_bytes, args.event_rate_hz),
                "hardware_coincidence_only": metrics(test_y, test_y, args.event_bytes, args.event_rate_hz)}
    for target in (0.5, 0.75, 0.9, 0.95, 0.99):
        adc_threshold = float(np.quantile(adc_train[train_y], 1.0 - target))
        score_threshold = float(np.quantile(scores[:split][train_y], 1.0 - target))
        policies[f"adc_target_recall_{int(target*100)}"] = {
            "threshold": round(adc_threshold, 4),
            **metrics(adc_test >= adc_threshold, test_y, args.event_bytes, args.event_rate_hz),
        }
        policies[f"mlp_target_recall_{int(target*100)}"] = {
            "threshold": round(score_threshold, 6),
            **metrics(scores[split:] >= score_threshold, test_y, args.event_bytes, args.event_rate_hz),
        }
    result = {
        "data": {"rows": len(rows), "chronological_train_rows": split, "test_rows": len(rows)-split,
                 "test_coincidence_rate": round(float(test_y.mean()), 5)},
        "assumptions": {"payload_bytes_per_event": args.event_bytes, "event_rate_hz": args.event_rate_hz,
                        "excludes_protocol_headers_radio_retries_and_idle_power": True},
        "policies": policies,
        "interpretation": [
            "This evaluates data volume and weak-label retention, not measured radio energy.",
            "Hardware coincidence is an oracle for the existing weak label but discards unusual noncoincident events.",
            "ADC and MLP thresholds are selected on the chronological training segment only.",
        ],
        "runtime_seconds": round(time.time() - started, 2),
    }
    with open(args.out, "w") as fh: json.dump(result, fh, indent=2)
    with open(args.report, "w") as fh:
        fh.write("# Edge Data-Reduction Policies\n\n")
        fh.write("| policy | transmit | reduction | coincident recall | KiB/day |\n|---|--:|--:|--:|--:|\n")
        for name, value in policies.items():
            fh.write(f"| {name} | {value['transmit_fraction']:.3f} | {value['data_reduction_x']:.1f}x | "
                     f"{value['coincident_recall']:.3f} | {value['payload_kib_per_day']:.1f} |\n")
        fh.write("\nThese are payload simulations; radio and board power require hardware measurement.\n")
    print(f"Wrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()
