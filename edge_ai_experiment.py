#!/usr/bin/env python3
"""
Prototype the edge-AI angle on the clean CosmicWatch window.

The script pulls CosmicWatch events from Elasticsearch, builds event-level
features, trains a tiny MLP baseline, and optionally trains a small pure-PyTorch
spiking neural network using rate-coded spike inputs.

Default window: 2026-01-23 through 2026-01-24 UTC, the cleanest high-volume
range found by data_readiness.py.
"""
import argparse
import datetime as dt
import json
import math
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")

import numpy as np
import requests
import urllib3

from credo_config import es_auth, es_settings, verify_certs

try:
    import torch
except ImportError:
    torch = None

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SOURCE = "cosmicwatch-v3x"
SOURCE_FIELDS = [
    "timestamp",
    "timestamp_ms",
    "pico_timestamp_s",
    "adc_value",
    "sipm_mv",
    "coincident",
    "coincidence_flag",
    "temperature_c",
    "pressure_pa",
    "detector_name",
    "device_id",
]
FEATURE_NAMES = [
    "adc_value",
    "sipm_mv",
    "log1p_interarrival_ms",
    "temperature_c_clean",
    "pressure_pa_clean",
]


def post_json(path, body, params=None, timeout=180):
    es_url, _index = es_settings()
    response = requests.post(
        f"{es_url}/{path.lstrip('/')}",
        auth=es_auth(),
        verify=verify_certs(),
        headers={"Content-Type": "application/json"},
        params=params,
        json=body,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def clear_scroll(scroll_id):
    if not scroll_id:
        return
    es_url, _index = es_settings()
    try:
        requests.delete(
            f"{es_url}/_search/scroll",
            auth=es_auth(),
            verify=verify_certs(),
            headers={"Content-Type": "application/json"},
            json={"scroll_id": [scroll_id]},
            timeout=30,
        )
    except requests.RequestException:
        pass


def fetch_events(start_utc, end_utc, max_events, page_size, scroll_keepalive):
    _es_url, index = es_settings()
    body = {
        "size": page_size,
        "sort": ["_doc"],
        "_source": SOURCE_FIELDS,
        "query": {
            "bool": {
                "filter": [
                    {"term": {"source": SOURCE}},
                    {"range": {"timestamp": {"gte": start_utc, "lt": end_utc}}},
                ]
            }
        },
    }
    response = post_json(
        f"{index}/_search",
        body,
        params={"scroll": scroll_keepalive},
    )

    events = []
    scroll_id = response.get("_scroll_id")
    try:
        while True:
            hits = response.get("hits", {}).get("hits", [])
            if not hits:
                break

            for hit in hits:
                events.append(hit.get("_source", {}))
                if max_events and len(events) >= max_events:
                    return events

            response = post_json(
                "_search/scroll",
                {"scroll": scroll_keepalive, "scroll_id": scroll_id},
            )
            scroll_id = response.get("_scroll_id", scroll_id)
    finally:
        clear_scroll(scroll_id)

    return events


def to_float(value):
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return None


def parse_timestamp_ms(value):
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return int(parsed.timestamp() * 1000)


def timestamp_iso(timestamp_ms):
    return dt.datetime.fromtimestamp(
        timestamp_ms / 1000,
        tz=dt.timezone.utc,
    ).isoformat().replace("+00:00", "Z")


def clean_temperature(value):
    temperature = to_float(value)
    if temperature is None or temperature < -50 or temperature > 80:
        return None
    return temperature


def clean_pressure(value):
    pressure = to_float(value)
    if pressure is None or pressure < 80_000 or pressure > 110_000:
        return None
    return pressure


def build_rows(events):
    timestamped_events = []
    for event in events:
        timestamp_ms = parse_timestamp_ms(
            event.get("timestamp_ms", event.get("timestamp"))
        )
        if timestamp_ms is not None:
            timestamped_events.append((timestamp_ms, event))

    timestamped_events.sort(key=lambda item: item[0])
    rows = []
    previous_timestamp_ms = None

    for timestamp_ms, event in timestamped_events:
        adc_value = to_float(event.get("adc_value"))
        coincident = to_bool(event.get("coincident", event.get("coincidence_flag")))
        if adc_value is None or coincident is None:
            continue

        if previous_timestamp_ms is None:
            interarrival_ms = 0.0
        else:
            interarrival_ms = max(0.0, float(timestamp_ms - previous_timestamp_ms))
        previous_timestamp_ms = timestamp_ms

        rows.append(
            {
                "timestamp_ms": timestamp_ms,
                "timestamp": timestamp_iso(timestamp_ms),
                "coincident": coincident,
                "adc_value": adc_value,
                "sipm_mv": to_float(event.get("sipm_mv")),
                "interarrival_ms": interarrival_ms,
                "log1p_interarrival_ms": math.log1p(interarrival_ms),
                "temperature_c_clean": clean_temperature(event.get("temperature_c")),
                "pressure_pa_clean": clean_pressure(event.get("pressure_pa")),
                "device_id": event.get("device_id"),
                "detector_name": event.get("detector_name"),
            }
        )

    return rows


def dataset_from_rows(rows):
    raw_features = []
    labels = []
    for row in rows:
        raw_features.append(
            [
                row["adc_value"],
                row["sipm_mv"],
                row["log1p_interarrival_ms"],
                row["temperature_c_clean"],
                row["pressure_pa_clean"],
            ]
        )
        labels.append(1.0 if row["coincident"] else 0.0)

    return np.array(raw_features, dtype=np.float32), np.array(labels, dtype=np.float32)


def fill_missing(train_features, test_features):
    medians = np.nanmedian(train_features, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
    train_filled = np.where(np.isnan(train_features), medians, train_features)
    test_filled = np.where(np.isnan(test_features), medians, test_features)
    return train_filled.astype(np.float32), test_filled.astype(np.float32), medians


def standardize(train_features, test_features):
    means = train_features.mean(axis=0)
    stds = train_features.std(axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)
    return (
        ((train_features - means) / stds).astype(np.float32),
        ((test_features - means) / stds).astype(np.float32),
        means.astype(np.float32),
        stds.astype(np.float32),
    )


def binary_metrics(labels, scores, threshold):
    labels_bool = labels.astype(bool)
    predictions = scores >= threshold
    true_positive = int(np.sum(predictions & labels_bool))
    false_positive = int(np.sum(predictions & ~labels_bool))
    true_negative = int(np.sum(~predictions & ~labels_bool))
    false_negative = int(np.sum(~predictions & labels_bool))

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1_score = 2 * precision * recall / max(1e-12, precision + recall)
    accuracy = (true_positive + true_negative) / max(1, len(labels))

    return {
        "threshold": float(threshold),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1_score), 4),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "auc": round(float(roc_auc(labels, scores)), 4),
    }


def roc_auc(labels, scores):
    labels_bool = labels.astype(bool)
    positive_count = int(np.sum(labels_bool))
    negative_count = int(len(labels) - positive_count)
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start_index = 0
    while start_index < len(scores):
        end_index = start_index + 1
        while end_index < len(scores) and sorted_scores[end_index] == sorted_scores[start_index]:
            end_index += 1
        average_rank = (start_index + 1 + end_index) / 2.0
        ranks[order[start_index:end_index]] = average_rank
        start_index = end_index

    positive_rank_sum = np.sum(ranks[labels_bool])
    auc = (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)
    return auc


def best_threshold(labels, scores):
    candidates = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 97)))
    if len(candidates) == 0:
        return 0.5
    best_candidate = float(candidates[0])
    best_f1_score = -1.0
    for threshold in candidates:
        metric = binary_metrics(labels, scores, float(threshold))
        if metric["f1"] > best_f1_score:
            best_f1_score = metric["f1"]
            best_candidate = float(threshold)
    return best_candidate


def adc_threshold_baseline(train_features, train_labels, test_features, test_labels):
    train_adc = train_features[:, 0]
    test_adc = test_features[:, 0]
    threshold = best_threshold(train_labels, train_adc)
    return {
        "description": "Predict coincident=true when adc_value exceeds a tuned threshold.",
        "adc_threshold": round(float(threshold), 4),
        "train": binary_metrics(train_labels, train_adc, threshold),
        "test": binary_metrics(test_labels, test_adc, threshold),
    }


def model_size(model):
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "parameters": int(parameter_count),
        "float32_bytes": int(parameter_count * 4),
        "int8_bytes": int(parameter_count),
    }


def train_mlp(
    train_features,
    train_labels,
    test_features,
    test_labels,
    epochs,
    batch_size,
    learning_rate,
    seed,
):
    if torch is None:
        return {"available": False, "reason": "PyTorch is not installed."}

    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(train_features.shape[1], 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1),
    )

    positive_count = float(np.sum(train_labels))
    negative_count = float(len(train_labels) - positive_count)
    positive_weight = torch.tensor([negative_count / max(1.0, positive_count)])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_tensor = torch.tensor(train_features, dtype=torch.float32)
    train_label_tensor = torch.tensor(train_labels, dtype=torch.float32)
    test_tensor = torch.tensor(test_features, dtype=torch.float32)

    for _epoch in range(epochs):
        permutation = torch.randperm(len(train_tensor))
        for start_index in range(0, len(train_tensor), batch_size):
            batch_indices = permutation[start_index : start_index + batch_size]
            logits = model(train_tensor[batch_indices]).squeeze(-1)
            loss = loss_fn(logits, train_label_tensor[batch_indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        train_scores = torch.sigmoid(model(train_tensor).squeeze(-1)).numpy()
        test_scores = torch.sigmoid(model(test_tensor).squeeze(-1)).numpy()

    threshold = best_threshold(train_labels, train_scores)
    return {
        "available": True,
        "model": "MLP(input -> 8 ReLU -> 1)",
        "epochs": epochs,
        "size": model_size(model),
        "train": binary_metrics(train_labels, train_scores, threshold),
        "test": binary_metrics(test_labels, test_scores, threshold),
    }


if torch is not None:

    class SurrogateSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, membrane_minus_threshold):
            ctx.save_for_backward(membrane_minus_threshold)
            return (membrane_minus_threshold > 0).float()

        @staticmethod
        def backward(ctx, grad_output):
            (membrane_minus_threshold,) = ctx.saved_tensors
            surrogate_grad = 1.0 / (1.0 + membrane_minus_threshold.abs()).pow(2)
            return grad_output * surrogate_grad


    class TinySNN(torch.nn.Module):
        def __init__(self, feature_count, hidden_count=12, decay=0.85, threshold=1.0):
            super().__init__()
            self.input_layer = torch.nn.Linear(feature_count, hidden_count, bias=False)
            self.output_layer = torch.nn.Linear(hidden_count, 1)
            self.decay = decay
            self.threshold = threshold

        def forward(self, spike_tensor):
            batch_count, step_count, _feature_count = spike_tensor.shape
            membrane = torch.zeros(
                batch_count,
                self.input_layer.out_features,
                device=spike_tensor.device,
            )
            readout = torch.zeros(batch_count, device=spike_tensor.device)

            for step_index in range(step_count):
                membrane = self.decay * membrane + self.input_layer(
                    spike_tensor[:, step_index, :]
                )
                spikes = SurrogateSpike.apply(membrane - self.threshold)
                membrane = membrane * (1.0 - spikes.detach())
                readout = readout + self.output_layer(spikes).squeeze(-1)

            return readout / step_count


def minmax01_from_train(train_features, test_features):
    low = np.percentile(train_features, 1, axis=0)
    high = np.percentile(train_features, 99, axis=0)
    span = np.where(high - low < 1e-6, 1.0, high - low)
    train_scaled = np.clip((train_features - low) / span, 0.0, 1.0)
    test_scaled = np.clip((test_features - low) / span, 0.0, 1.0)
    return train_scaled.astype(np.float32), test_scaled.astype(np.float32)


def rate_code_spikes(features01, step_count):
    thresholds = ((np.arange(step_count, dtype=np.float32) + 0.5) / step_count)
    spikes = features01[:, None, :] >= thresholds[None, :, None]
    return spikes.astype(np.float32)


def train_snn(
    train_features,
    train_labels,
    test_features,
    test_labels,
    epochs,
    batch_size,
    learning_rate,
    step_count,
    seed,
):
    if torch is None:
        return {"available": False, "reason": "PyTorch is not installed."}

    torch.manual_seed(seed)
    train_scaled, test_scaled = minmax01_from_train(train_features, test_features)
    train_spikes = torch.tensor(rate_code_spikes(train_scaled, step_count), dtype=torch.float32)
    test_spikes = torch.tensor(rate_code_spikes(test_scaled, step_count), dtype=torch.float32)
    train_label_tensor = torch.tensor(train_labels, dtype=torch.float32)

    model = TinySNN(feature_count=train_features.shape[1])
    positive_count = float(np.sum(train_labels))
    negative_count = float(len(train_labels) - positive_count)
    positive_weight = torch.tensor([negative_count / max(1.0, positive_count)])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _epoch in range(epochs):
        permutation = torch.randperm(len(train_spikes))
        for start_index in range(0, len(train_spikes), batch_size):
            batch_indices = permutation[start_index : start_index + batch_size]
            logits = model(train_spikes[batch_indices])
            loss = loss_fn(logits, train_label_tensor[batch_indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        train_scores = torch.sigmoid(model(train_spikes)).numpy()
        test_scores = torch.sigmoid(model(test_spikes)).numpy()

    threshold = best_threshold(train_labels, train_scores)
    size = model_size(model)
    size["timesteps"] = int(step_count)
    size["approx_synaptic_ops_per_event"] = int(
        step_count
        * (
            train_features.shape[1] * model.input_layer.out_features
            + model.input_layer.out_features
        )
    )

    return {
        "available": True,
        "model": "rate-coded LIF SNN(input -> 12 spiking neurons -> 1)",
        "epochs": epochs,
        "size": size,
        "train": binary_metrics(train_labels, train_scores, threshold),
        "test": binary_metrics(test_labels, test_scores, threshold),
    }


def write_plots(rows, plots_dir):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    output_dir = Path(plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adc_signal = [row["adc_value"] for row in rows if row["coincident"]]
    adc_background = [row["adc_value"] for row in rows if not row["coincident"]]
    plt.figure(figsize=(8, 5))
    plt.hist(adc_background, bins=80, alpha=0.65, label="non-coincident", density=True)
    plt.hist(adc_signal, bins=80, alpha=0.65, label="coincident", density=True)
    plt.xlabel("ADC value")
    plt.ylabel("density")
    plt.title("CosmicWatch ADC separability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "adc_separability.png", dpi=160)
    plt.close()

    interarrival = [
        row["interarrival_ms"]
        for row in rows
        if row["interarrival_ms"] > 0 and row["interarrival_ms"] < 60_000
    ]
    plt.figure(figsize=(8, 5))
    plt.hist(interarrival, bins=80)
    plt.xlabel("inter-arrival time (ms)")
    plt.ylabel("events")
    plt.title("Event timing distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "interarrival_ms.png", dpi=160)
    plt.close()

    return [
        str(output_dir / "adc_separability.png"),
        str(output_dir / "interarrival_ms.png"),
    ]


def summarize_rows(rows):
    coincident_count = sum(1 for row in rows if row["coincident"])
    noncoincident_count = len(rows) - coincident_count
    interarrival_values = np.array(
        [row["interarrival_ms"] for row in rows if row["interarrival_ms"] > 0],
        dtype=np.float32,
    )
    timestamps = [row["timestamp_ms"] for row in rows]
    duration_seconds = max(1.0, (max(timestamps) - min(timestamps)) / 1000)

    return {
        "rows": len(rows),
        "coincident": coincident_count,
        "noncoincident": noncoincident_count,
        "coincident_rate": round(coincident_count / max(1, len(rows)), 4),
        "time_range_utc": {
            "start": timestamp_iso(min(timestamps)),
            "end": timestamp_iso(max(timestamps)),
        },
        "event_rate_hz": round(len(rows) / duration_seconds, 4),
        "median_interarrival_ms": (
            round(float(np.median(interarrival_values)), 4)
            if len(interarrival_values)
            else None
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-01-23T00:00:00Z")
    parser.add_argument("--end", default="2026-01-25T00:00:00Z")
    parser.add_argument("--max-events", type=int, default=50_000,
                        help="0 means pull the full date window")
    parser.add_argument("--page-size", type=int, default=5_000)
    parser.add_argument("--scroll", default="2m")
    parser.add_argument("--out", default="edge_ai_experiment.json")
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--snn-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--snn-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-train", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    started_at = time.time()
    max_events = args.max_events if args.max_events > 0 else None

    print(f"Pulling {SOURCE} events: {args.start} -> {args.end}")
    events = fetch_events(
        args.start,
        args.end,
        max_events=max_events,
        page_size=args.page_size,
        scroll_keepalive=args.scroll,
    )
    rows = build_rows(events)
    if len(rows) < 100:
        raise SystemExit(f"Only {len(rows)} usable rows found; not enough to train.")

    raw_features, labels = dataset_from_rows(rows)
    split_index = max(1, int(0.8 * len(rows)))
    train_raw = raw_features[:split_index]
    test_raw = raw_features[split_index:]
    train_labels = labels[:split_index]
    test_labels = labels[split_index:]
    train_filled, test_filled, feature_medians = fill_missing(train_raw, test_raw)
    train_standard, test_standard, feature_means, feature_stds = standardize(
        train_filled,
        test_filled,
    )

    output = {
        "query": {
            "source": SOURCE,
            "start": args.start,
            "end": args.end,
            "max_events": args.max_events,
        },
        "data": summarize_rows(rows),
        "split": {
            "method": "chronological 80/20 split",
            "train_rows": int(len(train_labels)),
            "test_rows": int(len(test_labels)),
        },
        "features": {
            "names": FEATURE_NAMES,
            "missing_fill_medians": {
                name: float(value)
                for name, value in zip(FEATURE_NAMES, feature_medians)
            },
            "standardization_mean": {
                name: float(value)
                for name, value in zip(FEATURE_NAMES, feature_means)
            },
            "standardization_std": {
                name: float(value)
                for name, value in zip(FEATURE_NAMES, feature_stds)
            },
        },
        "baselines": {
            "adc_threshold": adc_threshold_baseline(
                train_filled,
                train_labels,
                test_filled,
                test_labels,
            )
        },
        "models": {},
        "caveats": [
            "coincident is a hardware-derived weak label, not independent hand truth",
            "this is an edge-feasibility prototype, not a final physics classifier",
            "current ES data has one hardware node, so network-GNN work needs CREDO multi-node data or simulation",
        ],
    }

    if args.no_train:
        output["models"]["training"] = {"skipped": True}
    else:
        output["models"]["mlp"] = train_mlp(
            train_standard,
            train_labels,
            test_standard,
            test_labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        output["models"]["snn"] = train_snn(
            train_filled,
            train_labels,
            test_filled,
            test_labels,
            epochs=args.snn_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            step_count=args.snn_steps,
            seed=args.seed,
        )

    if args.plots_dir:
        output["plots"] = write_plots(rows, args.plots_dir)

    output["runtime_seconds"] = round(time.time() - started_at, 2)
    with open(args.out, "w") as output_file:
        json.dump(output, output_file, indent=2)

    print(f"Rows: {output['data']['rows']:,}")
    print(f"Coincident rate: {100 * output['data']['coincident_rate']:.2f}%")
    print(f"ADC baseline F1: {output['baselines']['adc_threshold']['test']['f1']}")
    if output["models"].get("mlp", {}).get("available"):
        print(f"MLP test F1: {output['models']['mlp']['test']['f1']}")
    if output["models"].get("snn", {}).get("available"):
        print(f"SNN test F1: {output['models']['snn']['test']['f1']}")
        print(f"SNN int8 size: {output['models']['snn']['size']['int8_bytes']} bytes")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
