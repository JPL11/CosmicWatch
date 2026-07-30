#!/usr/bin/env python3
"""
Multi-modal federation demo: ONE FedAvg orchestration layer, TWO client types.

  - Event clients: the tiny supervised MLP on real CosmicWatch event features
    (ADC/timing, `coincident` weak label). The split into clients is SYNTHETIC
    (one physical detector) — same caveat as fl_simulation.py.
  - Image clients: the 26,584-param autoencoder on legacy CREDO 20x20 hit-crops,
    federated across 8 REAL devices (legacy_fl_hardware.npz).

The point being demonstrated: the federation is the shared layer, not the
representation. Each modality keeps its own model head; one round scheduler
trains and aggregates both per round. There is NO shared latent space — a
cross-modal representation would need synchronized cross-source data that does
not exist (all sources are temporally disjoint; see data_analysis_report.md).

Usage:
  python3 multimodal_federation_demo.py --max-events 50000 --rounds 10 \
      --plots-dir plots_multimodal
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from edge_ai_experiment import (
    best_threshold,
    binary_metrics,
    build_rows,
    dataset_from_rows,
    fetch_events,
    fill_missing,
    standardize,
    summarize_rows,
)
from fl_simulation import (
    average_states,
    build_mlp,
    evaluate as evaluate_event,
    param_count,
    partition_indices,
    train_local as train_event_client,
)


# ---------------- image modality (reuses the FL hardware benchmark model) ----------------

def build_autoencoder():
    return torch.nn.Sequential(
        torch.nn.Linear(400, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 8), torch.nn.ReLU(),
        torch.nn.Linear(8, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 400), torch.nn.Sigmoid(),
    )


def weighted_loss(prediction, target):
    return (((prediction - target) ** 2) * (1.0 + 4.0 * target)).mean()


def train_image_client(global_state, images, epochs, batch_size, lr, seed):
    torch.manual_seed(seed)
    net = build_autoencoder()
    if global_state is not None:
        net.load_state_dict(global_state)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    data = torch.from_numpy(images)
    for _epoch in range(epochs):
        order = torch.randperm(len(data))
        for start in range(0, len(data), batch_size):
            sample = data[order[start:start + batch_size]]
            loss = weighted_loss(net(sample), sample)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {key: value.detach().clone() for key, value in net.state_dict().items()}


def evaluate_image(state, tests):
    net = build_autoencoder(); net.load_state_dict(state); net.eval()
    per_device = {}
    with torch.no_grad():
        for device, values in tests.items():
            data = torch.from_numpy(values)
            per_device[device] = float(weighted_loss(net(data), data))
    weights = [len(tests[d]) for d in per_device]
    return {
        "weighted_mse": round(float(np.average(list(per_device.values()), weights=weights)), 7),
        "worst_device_weighted_mse": round(max(per_device.values()), 7),
    }


def load_image_clients(npz_path):
    archive = np.load(npz_path, allow_pickle=True)
    devices = [str(d) for d in archive["device_ids"]]
    trains, tests = {}, {}
    for index, device in enumerate(devices):
        trains[device] = archive[f"train_{index}"].reshape(-1, 400).astype(np.float32) / 255.0
        tests[device] = archive[f"test_{index}"].reshape(-1, 400).astype(np.float32) / 255.0
    return devices, trains, tests


# ---------------- communication accounting ----------------

def modality_communication(params, clients, rounds, raw_bytes):
    update = params * 4
    total = update * clients * rounds * 2  # upload + broadcast
    return {
        "model_parameters": int(params),
        "bytes_per_update_float32": int(update),
        "federated_total_bytes_with_broadcast": int(total),
        "raw_data_bytes_if_centralized": int(raw_bytes),
        "federated_vs_raw_ratio": round(total / max(1, raw_bytes), 4),
    }


def write_plots(results, plots_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(plots_dir); out.mkdir(parents=True, exist_ok=True)
    history = results["rounds_history"]
    rounds = [h["round"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(rounds, [h["event_f1"] for h in history], marker="o", color="#3b7dd8")
    axes[0].axhline(results["event_modality"]["centralized"]["f1"], ls="--", color="0.4",
                    label=f"centralized F1={results['event_modality']['centralized']['f1']}")
    axes[0].set_xlabel("federated round"); axes[0].set_ylabel("test F1")
    axes[0].set_title("Event clients (tiny MLP, synthetic partition)")
    axes[0].legend()
    axes[1].plot(rounds, [h["image_weighted_mse"] for h in history], marker="o", color="#d8703b")
    axes[1].axhline(results["image_modality"]["centralized"]["weighted_mse"], ls="--", color="0.4",
                    label=f"centralized MSE={results['image_modality']['centralized']['weighted_mse']}")
    axes[1].set_xlabel("federated round"); axes[1].set_ylabel("test weighted MSE")
    axes[1].set_title("Image clients (autoencoder, 8 real devices)")
    axes[1].legend()
    fig.suptitle("One federation, two modalities — per-modality heads, one round scheduler")
    fig.tight_layout()
    fig.savefig(out / "multimodal_convergence.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    event_comm = results["communication"]["event_modality"]
    image_comm = results["communication"]["image_modality"]
    labels = ["event\nupdates", "event\nraw", "image\nupdates", "image\nraw"]
    values = [event_comm["federated_total_bytes_with_broadcast"], event_comm["raw_data_bytes_if_centralized"],
              image_comm["federated_total_bytes_with_broadcast"], image_comm["raw_data_bytes_if_centralized"]]
    bars = ax.bar(labels, values, color=["#3b7dd8", "#9db8dc", "#d8703b", "#e8b394"])
    ax.set_yscale("log"); ax.set_ylabel("bytes (log)")
    ax.set_title("The communication win is modality-dependent\n"
                 f"events: {event_comm['federated_vs_raw_ratio']}x raw   "
                 f"images: {image_comm['federated_vs_raw_ratio']}x raw")
    fig.tight_layout()
    fig.savefig(out / "multimodal_communication.png", dpi=160)
    plt.close(fig)
    return [str(out / "multimodal_convergence.png"), str(out / "multimodal_communication.png")]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-01-23T00:00:00Z")
    parser.add_argument("--end", default="2026-01-25T00:00:00Z")
    parser.add_argument("--max-events", type=int, default=50_000)
    parser.add_argument("--page-size", type=int, default=5_000)
    parser.add_argument("--scroll", default="2m")
    parser.add_argument("--event-clients", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--event-batch-size", type=int, default=512)
    parser.add_argument("--event-lr", type=float, default=0.003)
    parser.add_argument("--image-data", default="legacy_fl_hardware.npz")
    parser.add_argument("--image-batch-size", type=int, default=256)
    parser.add_argument("--image-lr", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="multimodal_federation_results.json")
    parser.add_argument("--plots-dir", default="plots_multimodal")
    return parser.parse_args()


def main():
    args = parse_args(); started = time.time()
    max_events = args.max_events if args.max_events > 0 else None

    # ---- event modality data (real features, synthetic client split) ----
    print(f"Pulling events {args.start} -> {args.end} (max {max_events})")
    events = fetch_events(args.start, args.end, max_events=max_events,
                          page_size=args.page_size, scroll_keepalive=args.scroll)
    rows = build_rows(events)
    raw_features, labels = dataset_from_rows(rows)
    split = max(1, int(0.8 * len(rows)))
    train_raw, test_raw = raw_features[:split], raw_features[split:]
    train_labels, test_labels = labels[:split], labels[split:]
    train_filled, test_filled, _ = fill_missing(train_raw, test_raw)
    train_x, test_x, _, _ = standardize(train_filled, test_filled)
    input_dim = train_x.shape[1]
    event_indices = partition_indices(train_x, train_labels, args.event_clients, "non_iid", args.seed)

    # ---- image modality data (8 real devices) ----
    image_devices, image_trains, image_tests = load_image_clients(args.image_data)
    print(f"Event clients: {args.event_clients} (synthetic non-IID)   "
          f"Image clients: {len(image_devices)} real devices")

    # ---- one round scheduler, two model registries ----
    torch.manual_seed(args.seed)
    event_global = build_mlp(input_dim).state_dict()
    image_global = build_autoencoder().state_dict()
    history = []
    for round_index in range(args.rounds):
        event_states, event_weights = [], []
        for client_id, idx in enumerate(event_indices):
            if len(idx) == 0:
                continue
            state = train_event_client(event_global, train_x[idx], train_labels[idx],
                                       epochs=1, batch_size=args.event_batch_size,
                                       lr=args.event_lr, input_dim=input_dim,
                                       seed=args.seed + 1000 * round_index + client_id)
            event_states.append(state); event_weights.append(len(idx))
        event_global = average_states(event_states, event_weights)

        image_states, image_weights = [], []
        for client_id, device in enumerate(image_devices):
            state = train_image_client(image_global, image_trains[device], epochs=1,
                                       batch_size=args.image_batch_size, lr=args.image_lr,
                                       seed=args.seed + 1000 * round_index + 100 + client_id)
            image_states.append(state); image_weights.append(len(image_trains[device]))
        image_global = average_states(image_states, image_weights)

        event_metric = evaluate_event(event_global, train_x, train_labels, test_x, test_labels, input_dim)
        image_metric = evaluate_image(image_global, image_tests)
        history.append({"round": round_index + 1, "event_f1": event_metric["f1"],
                        "event_auc": event_metric["auc"],
                        "image_weighted_mse": image_metric["weighted_mse"]})
        print(f"round {round_index + 1:2d}: event F1={event_metric['f1']:.4f} "
              f"image weighted MSE={image_metric['weighted_mse']:.5f}")

    # ---- centralized references (same total gradient steps) ----
    event_central_state = train_event_client(None, train_x, train_labels, epochs=args.rounds,
                                             batch_size=args.event_batch_size, lr=args.event_lr,
                                             input_dim=input_dim, seed=args.seed)
    event_central = evaluate_event(event_central_state, train_x, train_labels, test_x, test_labels, input_dim)
    pooled_images = np.concatenate([image_trains[d] for d in image_devices])
    image_central_state = train_image_client(None, pooled_images, epochs=args.rounds,
                                             batch_size=args.image_batch_size, lr=args.image_lr,
                                             seed=args.seed)
    image_central = evaluate_image(image_central_state, image_tests)

    event_final = evaluate_event(event_global, train_x, train_labels, test_x, test_labels, input_dim)
    image_final = evaluate_image(image_global, image_tests)

    results = {
        "framing": {
            "claim": "One FedAvg orchestration layer serves two sensor modalities with separate "
                     "model heads; the federation is the shared layer, not the representation.",
            "real": "event features/labels are real ES data; image clients are 8 real devices",
            "simulated": "the event-side client split is synthetic (one physical detector)",
            "not_done_and_why": "no shared cross-modal latent space: all sources are temporally "
                                "disjoint (zero cross-source coincidence), so per-sample fusion has "
                                "no data to validate against",
        },
        "data": {"events": summarize_rows(rows),
                 "image_clients": {d: int(len(image_trains[d])) for d in image_devices}},
        "config": {"rounds": args.rounds, "event_clients": args.event_clients,
                   "image_clients": len(image_devices)},
        "event_modality": {"federated": event_final, "centralized": event_central},
        "image_modality": {"federated": image_final, "centralized": image_central},
        "rounds_history": history,
        "communication": {
            "event_modality": modality_communication(
                param_count(build_mlp(input_dim)), args.event_clients, args.rounds,
                raw_bytes=len(train_labels) * input_dim * 4),
            "image_modality": modality_communication(
                sum(p.numel() for p in build_autoencoder().parameters()), len(image_devices),
                args.rounds, raw_bytes=int(sum(len(v) for v in image_trains.values()) * 400)),
            "note": "the communication win is modality-dependent: tiny event model vs large raw "
                    "stream wins; 26.6k-param image model vs 400-byte images does not",
        },
        "caveats": [
            "event federation is SIMULATED (synthetic partition of one physical detector)",
            "image federation uses real device partitions but runs on one machine here; "
            "per-client on-device cost is measured separately (fl_hardware_benchmark_*.json)",
            "coincident is a weak hardware label; the event F1 ceiling ~0.40 is label noise, "
            "not model capacity",
        ],
    }
    if args.plots_dir:
        results["plots"] = write_plots(results, args.plots_dir)
    results["runtime_seconds"] = round(time.time() - started, 2)
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=2)

    print(f"\nEvent modality:  federated F1={event_final['f1']}  centralized F1={event_central['f1']}")
    print(f"Image modality:  federated weighted MSE={image_final['weighted_mse']}  "
          f"centralized={image_central['weighted_mse']}")
    for name in ("event_modality", "image_modality"):
        comm = results["communication"][name]
        print(f"{name}: updates {comm['federated_total_bytes_with_broadcast']:,} B "
              f"vs raw {comm['raw_data_bytes_if_centralized']:,} B "
              f"(ratio {comm['federated_vs_raw_ratio']})")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
