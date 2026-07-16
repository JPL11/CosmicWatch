#!/usr/bin/env python3
"""Deduplicated cross-device timing search with a device/day time-shift null."""
import argparse
import collections
import json
import math
import random
import statistics
import time

from legacy_common import iter_legacy


WINDOWS = (10, 100, 1000)
DAY_MS = 86_400_000


def haversine_km(a, b):
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(value))


def count_pairs(rows, collect_distances=False):
    queue = collections.deque(); counts = {window: 0 for window in WINDOWS}
    device_pairs = {window: set() for window in WINDOWS}; distances = []
    for timestamp, device, location in sorted(rows):
        while queue and timestamp - queue[0][0] > WINDOWS[-1]: queue.popleft()
        for old_time, old_device, old_location in queue:
            if old_device == device: continue
            delta = timestamp - old_time
            for window in WINDOWS:
                if delta <= window:
                    counts[window] += 1
                    device_pairs[window].add(tuple(sorted((device, old_device))))
            if collect_distances and delta <= 1000 and location and old_location:
                distances.append(haversine_km(location, old_location))
        queue.append((timestamp, device, location))
    return counts, {window: len(values) for window, values in device_pairs.items()}, distances


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="credo_useful.csv")
    ap.add_argument("--null-runs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="legacy_timing.json")
    ap.add_argument("--report", default="legacy_timing_report.md")
    args = ap.parse_args(); started = time.time()
    rows = []; duplicates = 0
    for row in iter_legacy(args.csv, include_image=False):
        duplicates = row["duplicates_before"]
        rows.append((row["timestamp_ms"], row["device_id"], row["location"]))
    observed, observed_device_pairs, distances = count_pairs(rows, collect_distances=True)
    grouped = collections.defaultdict(list)
    for timestamp, device, _ in rows:
        grouped[(timestamp // DAY_MS, device)].append(timestamp % DAY_MS)
    rng = random.Random(args.seed); null = {window: [] for window in WINDOWS}
    for _ in range(args.null_runs):
        shifted = []
        for (day, device), values in grouped.items():
            magnitude = rng.randint(30_000, 1_800_000)
            offset = magnitude if rng.random() < 0.5 else -magnitude
            shifted.extend((day * DAY_MS + (value + offset) % DAY_MS, device, None) for value in values)
        counts, _, _ = count_pairs(shifted)
        for window in WINDOWS: null[window].append(counts[window])
    tests = {}
    for window in WINDOWS:
        values = null[window]
        tests[str(window)] = {
            "observed_pairs": observed[window], "distinct_device_pairs": observed_device_pairs[window],
            "null_mean": round(statistics.mean(values), 3), "null_sd": round(statistics.stdev(values), 3),
            "null_range": [min(values), max(values)],
            "empirical_upper_tail_p": round((1 + sum(value >= observed[window] for value in values)) / (len(values) + 1), 5),
            "observed_to_null_mean": round(observed[window] / max(1e-9, statistics.mean(values)), 4),
        }
    result = {
        "data": {"unique_events": len(rows), "duplicates_removed": duplicates,
                 "devices": len({row[1] for row in rows}), "device_days": len(grouped)},
        "method": "Within each device/day, circularly shift the complete event train by a random signed 30-1800 s lag.",
        "windows_ms": tests,
        "spatial_for_observed_pairs_within_1s": {
            "pairs_with_two_locations": len(distances),
            "median_distance_km": round(statistics.median(distances), 3) if distances else None,
            "min_distance_km": round(min(distances), 3) if distances else None,
            "max_distance_km": round(max(distances), 3) if distances else None,
        },
        "conclusion": "No cross-device timing excess is established when the upper-tail p-value is not small.",
        "limitations": ["Phone clock synchronization is unverified.", "Acquisition live-time is not explicitly recorded.",
                        "The time-shift null preserves device/day event trains but approximates shared live-time."],
        "runtime_seconds": round(time.time() - started, 2),
    }
    with open(args.out, "w") as fh: json.dump(result, fh, indent=2)
    with open(args.report, "w") as fh:
        fh.write("# Legacy Cross-Device Timing Null Test\n\n")
        fh.write(f"{len(rows):,} unique events from {result['data']['devices']} devices; {duplicates:,} exact duplicates removed.\n\n")
        fh.write("| window | observed | null mean | ratio | upper-tail p |\n|---:|--:|--:|--:|--:|\n")
        for window, value in tests.items():
            fh.write(f"| {window} ms | {value['observed_pairs']} | {value['null_mean']:.1f} | "
                     f"{value['observed_to_null_mean']:.3f} | {value['empirical_upper_tail_p']:.3f} |\n")
        fh.write("\nNo statistically supported cross-device excess is claimed.\n")
    print(f"Wrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()
