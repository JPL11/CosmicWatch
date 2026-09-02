"""Reproducible driver for the event-gateway hardware benchmark.
Replicates the Pi 4 protocol: synthetic JSONL stream (feature scales from
model_weights.json standardization constants, coincidence rate 0.0848),
200k-event as-fast-as-possible runs per policy, then a Poisson-paced
realistic-rate run (1.3757 Hz, ~120 s) measuring CPU duty fraction.
Selection fractions are NOT physics results (see input_caveat).
Usage: python3 cw_gateway_bench.py --out event_gateway_benchmark_<dev>.json
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--events", type=int, default=200000)
ap.add_argument("--rate-hz", type=float, default=1.3757)
ap.add_argument("--realistic-seconds", type=float, default=120.0)
ap.add_argument("--out", default="event_gateway_benchmark.json")
args = ap.parse_args()

w = json.load(open("model_weights.json"))
feats, mu, sd = w["features"], w["standardize_mean"], w["standardize_std"]
rng = np.random.default_rng(7)
COINC_RATE = 0.084755

def gen_stream(path, n):
    with open(path, "w") as fh:
        vals = rng.normal(mu, sd, size=(n, len(feats)))
        coinc = rng.random(n) < COINC_RATE
        for i in range(n):
            ev = {f: round(float(vals[i, j]), 4) for j, f in enumerate(feats)}
            ev["coincident"] = int(coinc[i])
            fh.write(json.dumps(ev) + "\n")

STREAM = "synthetic_stream.jsonl"
gen_stream(STREAM, args.events)
print(f"stream: {args.events} events", flush=True)

out = {"platform": {
    "model": open("/proc/device-tree/model").read().strip("\x00 ")
             if os.path.exists("/proc/device-tree/model") else platform.node(),
    "machine": platform.machine(), "platform": platform.platform(),
    "python": platform.python_version(),
    "throttled": subprocess.run(["vcgencmd", "get_throttled"],
                                capture_output=True, text=True).stdout.strip()
                 if os.path.exists("/usr/bin/vcgencmd") else "n/a"},
    "input_caveat": "Synthetic JSONL stream with feature scales from "
        "model_weights.json standardization constants and the 8.4% "
        "coincidence rate from edge_reduction.json. Selection fractions "
        "here are NOT physics results; only per-event CPU cost, throughput "
        "headroom, and duty fraction are hardware measurements."}

afap = {}
for policy in ("coincidence", "adc", "mlp", "hybrid"):
    sf = f"stats_{policy}.json"
    subprocess.run([sys.executable, "event_gateway.py", "--input", STREAM,
                    "--policy", policy, "--output", "/dev/null",
                    "--stats", sf], check=True,
                   stderr=subprocess.DEVNULL)
    s = json.load(open(sf))
    afap[policy] = {
        "events": s["received"],
        "process_cpu_seconds": s["process_cpu_seconds"],
        "cpu_microseconds_per_event":
            round(1e6 * s["process_cpu_seconds"] / s["received"], 2),
        "cpu_limited_events_per_second":
            round(s["received"] / s["process_cpu_seconds"], 1),
        "selected_fraction_on_synthetic_stream": s["selected_fraction"],
    }
    print(policy, afap[policy], flush=True)
out["throughput_as_fast_as_possible"] = afap

# realistic-rate run: Poisson-paced producer piped into the gateway
n_real = max(20, int(args.rate_hz * args.realistic_seconds))
gaps = rng.exponential(1.0 / args.rate_hz, size=n_real)
proc = subprocess.Popen([sys.executable, "event_gateway.py", "--input", "-",
                         "--policy", "hybrid", "--output", "/dev/null",
                         "--stats", "stats_realistic.json"],
                        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
                        text=True)
vals = rng.normal(mu, sd, size=(n_real, len(feats)))
coinc = rng.random(n_real) < COINC_RATE
t0 = time.monotonic()
for i in range(n_real):
    time.sleep(float(gaps[i]))
    ev = {f: round(float(vals[i, j]), 4) for j, f in enumerate(feats)}
    ev["coincident"] = int(coinc[i])
    proc.stdin.write(json.dumps(ev) + "\n")
    proc.stdin.flush()
    if time.monotonic() - t0 > args.realistic_seconds:
        break
proc.stdin.close(); proc.wait()
s = json.load(open("stats_realistic.json"))
out["realistic_rate_run"] = {
    "policy": "hybrid", "poisson_rate_hz": args.rate_hz,
    "received": s["received"], "wall_seconds": s["wall_seconds"],
    "process_cpu_seconds": s["process_cpu_seconds"],
    "cpu_duty_fraction_during_run": s["cpu_duty_fraction_during_run"]}
out["headroom_hybrid_vs_detector_rate"] = round(
    afap["hybrid"]["cpu_limited_events_per_second"] / args.rate_hz, 1)
out["power_note"] = ("Pass USB-meter idle/load watts to the pi_benchmark "
                     "harness for energy columns; gateway runs report CPU "
                     "duty only.")
json.dump(out, open(args.out, "w"), indent=1)
print("WROTE", args.out, flush=True)
