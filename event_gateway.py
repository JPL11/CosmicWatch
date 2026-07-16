#!/usr/bin/env python3
"""Blocking, event-triggered CosmicWatch gateway for Pi/Jetson/Linux systems.

Input is JSON Lines from stdin/a file, or from a serial device when pyserial is
installed. The process blocks in the kernel while waiting for the next line;
there is no inference polling loop. Selected events are written in batches.
"""
import argparse
import json
import math
import sys
import time

FEATURES = ("adc_value", "sipm_mv", "temperature_c", "pressure_pa")


def forward_pure(features, weights):
    hidden = []
    for column, bias in enumerate(weights["b1"]):
        value = sum(features[row] * weights["W1"][row][column] for row in range(len(features))) + bias
        hidden.append(max(0.0, value))
    output = sum(hidden[row] * weights["W2"][row][0] for row in range(len(hidden))) + weights["b2"][0]
    return 1.0 / (1.0 + math.exp(-output))


def normalized_features(event, weights):
    values = []
    for index, name in enumerate(FEATURES):
        value = event.get(name)
        if value is None:
            value = weights["standardize_mean"][index]
        value = float(value)
        if name == "temperature_c" and not -50 < value < 80:
            value = weights["standardize_mean"][index]
        if name == "pressure_pa" and not 80000 < value < 110000:
            value = weights["standardize_mean"][index]
        values.append((value - weights["standardize_mean"][index]) /
                      weights["standardize_std"][index])
    return values


def select_event(event, policy, weights, adc_threshold, mlp_threshold):
    coincidence = event.get("coincident", event.get("coincidence_flag", False))
    coincidence = str(coincidence).lower() in {"1", "true"}
    adc = float(event.get("adc_value", 0))
    if policy == "all":
        return True, None
    if policy == "coincidence":
        return coincidence, None
    if policy == "adc":
        return adc >= adc_threshold, None
    score = forward_pure(normalized_features(event, weights), weights)
    if policy == "mlp":
        return score >= mlp_threshold, score
    if policy == "hybrid":
        return coincidence or adc >= adc_threshold or score >= mlp_threshold, score
    raise ValueError(f"Unknown policy: {policy}")


def open_input(args):
    if args.serial:
        try:
            import serial
        except ImportError as exc:
            raise SystemExit("Serial input requires: pip install pyserial") from exc
        return serial.Serial(args.serial, args.baud, timeout=None)
    if args.input == "-":
        return sys.stdin.buffer
    return open(args.input, "rb")


def parse_line(raw):
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("event must be a JSON object")
    return value


def flush_batch(batch, output):
    for event in batch:
        output.write(json.dumps(event, separators=(",", ":")) + "\n")
    output.flush()
    batch.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="-", help="JSONL path or - for blocking stdin")
    ap.add_argument("--serial", help="blocking serial device, e.g. /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--output", default="selected_events.jsonl")
    ap.add_argument("--weights", default="model_weights.json")
    ap.add_argument("--policy", choices=("all", "coincidence", "adc", "mlp", "hybrid"), default="hybrid")
    ap.add_argument("--adc-threshold", type=float, default=238.0)
    ap.add_argument("--mlp-threshold", type=float, default=0.390279)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--stats", default="event_gateway_stats.json")
    args = ap.parse_args()
    with open(args.weights) as fh: weights = json.load(fh)
    source = open_input(args); output = sys.stdout if args.output == "-" else open(args.output, "a")
    batch = []; received = selected = invalid = 0
    started_wall = time.monotonic(); started_cpu = time.process_time()
    try:
        while not args.max_events or received < args.max_events:
            raw = source.readline()  # Kernel-blocking UART/pipe/file read: event-driven wait.
            if not raw:
                break
            received += 1
            try:
                event = parse_line(raw)
                keep, score = select_event(event, args.policy, weights,
                                           args.adc_threshold, args.mlp_threshold)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                invalid += 1; continue
            if keep:
                selected += 1
                if score is not None: event["edge_score"] = round(float(score), 6)
                batch.append(event)
                if len(batch) >= args.batch_size: flush_batch(batch, output)
    except KeyboardInterrupt:
        pass
    finally:
        if batch: flush_batch(batch, output)
        if source not in (sys.stdin, sys.stdin.buffer): source.close()
        if output is not sys.stdout: output.close()
    wall = time.monotonic() - started_wall; cpu = time.process_time() - started_cpu
    stats = {
        "policy": args.policy, "received": received, "selected": selected, "invalid": invalid,
        "selected_fraction": round(selected / max(1, received - invalid), 6),
        "wall_seconds": round(wall, 6), "process_cpu_seconds": round(cpu, 6),
        "cpu_duty_fraction_during_run": round(cpu / max(wall, 1e-9), 8),
        "event_driven_contract": "blocking input; inference runs once per received event; batched output",
        "power_caveat": "Linux board idle power continues while blocked; this is not neuromorphic hardware.",
    }
    with open(args.stats, "w") as fh: json.dump(stats, fh, indent=2)
    print(json.dumps(stats), file=sys.stderr)


if __name__ == "__main__":
    main()
