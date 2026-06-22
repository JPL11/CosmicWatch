#!/usr/bin/env python3
"""
Simulation-only federated-learning prototype on REAL CosmicWatch edge features.

The federation is SIMULATED by partitioning the single-node Jan 23-24 CosmicWatch
stream into synthetic clients. This is necessary because the current ES index has
one timestamped device_id (cosmicwatch-001), so a real cross-device federation does
not exist in the data (see HANDOFF.md / SNN_GNN_FL_design.md).

What is real here:
  - The per-event features and the `coincident` weak label come from real ES data.
What is simulated here:
  - The split into "clients". Real FL needs multiple physical detectors/sites/phones.

The script trains the same tiny MLP three ways and reports BOTH accuracy and the
communication cost, which is the honest headline FL number:

  1. centralized   - one model on the pooled data (accuracy ceiling, ships raw data)
  2. federated     - FedAvg across clients, sharing only model updates
  3. local_only    - each client trains alone and never shares (no collaboration)

Two partitionings are run:
  - iid     : rows shuffled, then split evenly (easy case)
  - non_iid : rows sorted by adc_value then split, so clients get skewed feature/label
              distributions (realistic: detectors in different environments)

Usage:
  python3 fl_simulation.py --max-events 0 --clients 8 --rounds 15 --plots-dir plots_fl
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None

# Reuse the real-data pull, feature engineering and metrics from the edge experiment.
from edge_ai_experiment import (
    FEATURE_NAMES,
    SOURCE,
    best_threshold,
    binary_metrics,
    build_rows,
    dataset_from_rows,
    fetch_events,
    fill_missing,
    standardize,
    summarize_rows,
)


def build_mlp(input_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1),
    )


def param_count(model):
    return int(sum(p.numel() for p in model.parameters()))


def pos_weight_for(labels):
    positive = float(np.sum(labels))
    negative = float(len(labels) - positive)
    return torch.tensor([negative / max(1.0, positive)])


def train_local(global_state, features, labels, epochs, batch_size, lr, input_dim, seed):
    """Train one local model starting from the global weights; return its state_dict."""
    torch.manual_seed(seed)
    model = build_mlp(input_dim)
    if global_state is not None:
        model.load_state_dict(global_state)

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_for(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    feature_tensor = torch.tensor(features, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.float32)

    for _epoch in range(epochs):
        permutation = torch.randperm(len(feature_tensor))
        for start in range(0, len(feature_tensor), batch_size):
            idx = permutation[start : start + batch_size]
            logits = model(feature_tensor[idx]).squeeze(-1)
            loss = loss_fn(logits, label_tensor[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model.state_dict()


def average_states(states, weights):
    """FedAvg: sample-count-weighted average of client state_dicts."""
    total = float(sum(weights))
    averaged = {}
    for key in states[0]:
        averaged[key] = sum(
            state[key].float() * (weight / total)
            for state, weight in zip(states, weights)
        )
    return averaged


def evaluate(state, train_features, train_labels, test_features, test_labels, input_dim):
    """Pick the F1-optimal threshold on train scores, then score the test split."""
    model = build_mlp(input_dim)
    model.load_state_dict(state)
    with torch.no_grad():
        train_scores = torch.sigmoid(model(torch.tensor(train_features, dtype=torch.float32)).squeeze(-1)).numpy()
        test_scores = torch.sigmoid(model(torch.tensor(test_features, dtype=torch.float32)).squeeze(-1)).numpy()
    threshold = best_threshold(train_labels, train_scores)
    return binary_metrics(test_labels, test_scores, threshold)


def partition_indices(train_features, train_labels, clients, scheme, seed):
    """Return a list of index arrays, one per simulated client."""
    n = len(train_labels)
    if scheme == "iid":
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
    elif scheme == "non_iid":
        # Sort by adc_value (feature 0), then hand each client a contiguous band so
        # they receive skewed feature/label mixes (distinct positive rates).
        order = np.argsort(train_features[:, 0])
    else:
        raise ValueError(f"unknown scheme {scheme}")
    return [np.asarray(block) for block in np.array_split(order, clients)]


def client_profiles(indices, train_labels):
    profiles = []
    for client_id, idx in enumerate(indices):
        labels = train_labels[idx]
        profiles.append(
            {
                "client": client_id,
                "rows": int(len(idx)),
                "positive_rate": round(float(np.mean(labels)) if len(idx) else 0.0, 4),
            }
        )
    return profiles


def run_federated(train_features, train_labels, test_features, test_labels,
                  indices, rounds, local_epochs, batch_size, lr, input_dim, seed):
    """FedAvg over the simulated clients; track global test metrics per round."""
    global_state = build_mlp(input_dim).state_dict()
    history = []
    for round_index in range(rounds):
        client_states, client_weights = [], []
        for client_id, idx in enumerate(indices):
            if len(idx) == 0:
                continue
            state = train_local(
                global_state,
                train_features[idx],
                train_labels[idx],
                epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                input_dim=input_dim,
                seed=seed + 1000 * round_index + client_id,
            )
            client_states.append(state)
            client_weights.append(len(idx))
        global_state = average_states(client_states, client_weights)
        metric = evaluate(global_state, train_features, train_labels, test_features, test_labels, input_dim)
        history.append({"round": round_index + 1, "f1": metric["f1"], "auc": metric["auc"]})
    final = evaluate(global_state, train_features, train_labels, test_features, test_labels, input_dim)
    return final, history


def run_local_only(train_features, train_labels, test_features, test_labels,
                   indices, total_epochs, batch_size, lr, input_dim, seed):
    """Each client trains alone (no sharing); report the mean test metric."""
    per_client = []
    for client_id, idx in enumerate(indices):
        if len(idx) == 0:
            continue
        state = train_local(
            None,
            train_features[idx],
            train_labels[idx],
            epochs=total_epochs,
            batch_size=batch_size,
            lr=lr,
            input_dim=input_dim,
            seed=seed + client_id,
        )
        metric = evaluate(state, train_features, train_labels, test_features, test_labels, input_dim)
        per_client.append(metric["f1"])
    return {
        "mean_f1": round(float(np.mean(per_client)), 4) if per_client else None,
        "min_f1": round(float(np.min(per_client)), 4) if per_client else None,
        "max_f1": round(float(np.max(per_client)), 4) if per_client else None,
    }


def communication_report(model_params, clients, rounds, train_rows, num_features):
    """The honest FL headline: model updates shipped vs raw data shipped."""
    update_bytes = model_params * 4  # float32 weights
    fl_upload = update_bytes * clients * rounds
    fl_total = fl_upload * 2  # client upload + server broadcast
    raw_data_bytes = train_rows * num_features * 4  # if you instead centralized the features
    return {
        "model_parameters": int(model_params),
        "bytes_per_update_float32": int(update_bytes),
        "federated_upload_bytes": int(fl_upload),
        "federated_total_bytes_with_broadcast": int(fl_total),
        "centralized_raw_feature_bytes": int(raw_data_bytes),
        "federated_vs_centralized_ratio": round(fl_total / max(1, raw_data_bytes), 6),
        "interpretation": (
            "Federated training moves a fixed, tiny model footprint regardless of data size, "
            "while centralizing ships raw events that grow with the dataset. The federation also "
            "never moves raw data, which is the privacy argument. Accuracy parity vs centralized "
            "is the second axis; communication savings is the first."
        ),
    }


def write_plots(results, plots_dir):
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    out = Path(plots_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for scheme in ("iid", "non_iid"):
        history = results[scheme]["federated"]["history"]
        rounds = [h["round"] for h in history]
        f1 = [h["f1"] for h in history]
        plt.plot(rounds, f1, marker="o", label=f"FedAvg {scheme} (test F1)")
    plt.axhline(results["centralized"]["test"]["f1"], linestyle="--",
                label=f"centralized F1={results['centralized']['test']['f1']}")
    plt.xlabel("federated round")
    plt.ylabel("test F1")
    plt.title("FedAvg convergence vs centralized (real CosmicWatch features)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "fl_convergence.png", dpi=160)
    plt.close()

    comms = results["communication"]
    plt.figure(figsize=(7, 5))
    labels = ["federated\n(model updates)", "centralized\n(raw features)"]
    values = [comms["federated_total_bytes_with_broadcast"], comms["centralized_raw_feature_bytes"]]
    plt.bar(labels, values, color=["#3b7dd8", "#d8703b"])
    plt.yscale("log")
    plt.ylabel("bytes transferred (log scale)")
    plt.title("Communication cost: federated updates vs centralizing raw data")
    plt.tight_layout()
    plt.savefig(out / "fl_communication.png", dpi=160)
    plt.close()

    return [str(out / "fl_convergence.png"), str(out / "fl_communication.png")]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-01-23T00:00:00Z")
    parser.add_argument("--end", default="2026-01-25T00:00:00Z")
    parser.add_argument("--max-events", type=int, default=50_000,
                        help="0 means pull the full date window")
    parser.add_argument("--page-size", type=int, default=5_000)
    parser.add_argument("--scroll", default="2m")
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="fl_simulation_results.json")
    parser.add_argument("--plots-dir", default=None)
    return parser.parse_args()


def main():
    if torch is None:
        raise SystemExit("PyTorch is required for fl_simulation.py")

    args = parse_args()
    started_at = time.time()
    max_events = args.max_events if args.max_events > 0 else None

    print(f"Pulling {SOURCE} events: {args.start} -> {args.end}")
    events = fetch_events(args.start, args.end, max_events=max_events,
                          page_size=args.page_size, scroll_keepalive=args.scroll)
    rows = build_rows(events)
    if len(rows) < 200:
        raise SystemExit(f"Only {len(rows)} usable rows; not enough to simulate FL.")

    raw_features, labels = dataset_from_rows(rows)
    split = max(1, int(0.8 * len(rows)))
    train_raw, test_raw = raw_features[:split], raw_features[split:]
    train_labels, test_labels = labels[:split], labels[split:]
    train_filled, test_filled, _ = fill_missing(train_raw, test_raw)
    train_x, test_x, _, _ = standardize(train_filled, test_filled)
    input_dim = train_x.shape[1]

    # Centralized ceiling: total gradient steps ~ rounds * local_epochs.
    centralized_state = train_local(
        None, train_x, train_labels,
        epochs=args.rounds * args.local_epochs,
        batch_size=args.batch_size, lr=args.learning_rate,
        input_dim=input_dim, seed=args.seed,
    )
    centralized_metric = evaluate(centralized_state, train_x, train_labels, test_x, test_labels, input_dim)

    results = {
        "framing": {
            "real": "per-event features and the coincident weak label are real ES data",
            "simulated": "the split into clients is synthetic; real FL needs multiple physical detectors/sites/phones",
            "label_note": "coincident is a hardware-derived weak label, not independent hand truth",
        },
        "data": summarize_rows(rows),
        "config": {
            "clients": args.clients, "rounds": args.rounds,
            "local_epochs": args.local_epochs, "feature_names": FEATURE_NAMES,
        },
        "centralized": {"description": "one model on pooled data (ships raw data)", "test": centralized_metric},
    }

    for scheme in ("iid", "non_iid"):
        indices = partition_indices(train_x, train_labels, args.clients, scheme, args.seed)
        federated_metric, history = run_federated(
            train_x, train_labels, test_x, test_labels, indices,
            rounds=args.rounds, local_epochs=args.local_epochs,
            batch_size=args.batch_size, lr=args.learning_rate,
            input_dim=input_dim, seed=args.seed,
        )
        local_only = run_local_only(
            train_x, train_labels, test_x, test_labels, indices,
            total_epochs=args.rounds * args.local_epochs,
            batch_size=args.batch_size, lr=args.learning_rate,
            input_dim=input_dim, seed=args.seed,
        )
        results[scheme] = {
            "client_profiles": client_profiles(indices, train_labels),
            "federated": {"test": federated_metric, "history": history},
            "local_only": local_only,
        }

    model_params = param_count(build_mlp(input_dim))
    results["communication"] = communication_report(
        model_params, args.clients, args.rounds, len(train_labels), input_dim
    )
    results["caveats"] = [
        "SIMULATION ONLY: clients are synthetic partitions of one physical detector",
        "high accuracy is not a real multi-node detector result",
        "the defensible headline is communication/privacy, not a record F1",
        "real federation requires CREDO multi-node / multi-site data (see SNN_GNN_FL_design.md)",
    ]

    if args.plots_dir:
        results["plots"] = write_plots(results, args.plots_dir)

    results["runtime_seconds"] = round(time.time() - started_at, 2)
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=2)

    print(f"Rows: {results['data']['rows']:,}  clients={args.clients}  rounds={args.rounds}")
    print(f"Centralized test F1:        {centralized_metric['f1']}  AUC={centralized_metric['auc']}")
    for scheme in ("iid", "non_iid"):
        fed = results[scheme]["federated"]["test"]
        loc = results[scheme]["local_only"]["mean_f1"]
        print(f"Federated ({scheme:7s}) F1:   {fed['f1']}  AUC={fed['auc']}   local-only mean F1={loc}")
    comms = results["communication"]
    print(f"Comm cost: federated {comms['federated_total_bytes_with_broadcast']:,} B "
          f"vs centralizing raw {comms['centralized_raw_feature_bytes']:,} B "
          f"(ratio {comms['federated_vs_centralized_ratio']})")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
