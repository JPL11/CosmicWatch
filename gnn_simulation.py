#!/usr/bin/env python3
"""
Simulated sensor-network GNN for Cosmic-Ray Ensemble detection.

Important: this is simulation-only. The current Elasticsearch index does not
contain synchronized multi-node CosmicWatch/CREDO data suitable for a real GNN.

The simulation anchors background rates to the observed CosmicWatch prototype
window, then injects synthetic network-level events into a detector array. The
goal is to prototype the graph pipeline and make the data dependency explicit.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None

FEATURE_NAMES = [
    "log1p_event_count",
    "coincident_fraction",
    "mean_adc_scaled",
    "max_adc_scaled",
    "relative_time_scaled",
    "burst_score",
    "x_position_scaled",
    "y_position_scaled",
]

REAL_DATA_ANCHORS = {
    "cosmicwatch_full_window": "2026-01-23T00:00:00Z to 2026-01-24T23:59:58Z",
    "observed_event_rate_hz": 1.3757,
    "observed_coincident_rate": 0.1225,
    "cosmicwatch_nodes_in_es": 1,
    "multi_source_overlap_days_in_es": 0,
    "adc_context": {
        "presentation_noncoincident_mean": 249.4,
        "presentation_coincident_mean": 394.7,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=int, default=3000)
    parser.add_argument("--nodes", type=int, default=32)
    parser.add_argument("--area-km", type=float, default=500.0)
    parser.add_argument("--window-s", type=float, default=10.0)
    parser.add_argument("--positive-rate", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--neighbors", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--timing-jitter-ms", type=float, default=12.0)
    parser.add_argument("--core-sigma-km", type=float, default=150.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", default="gnn_simulation_results.json")
    parser.add_argument("--plots-dir", default="plots_gnn")
    return parser.parse_args()


def generate_layout(args, rng):
    positions = rng.uniform(0.0, args.area_km, size=(args.nodes, 2)).astype(np.float32)
    sigma = 0.25
    mean_log_rate = math.log(REAL_DATA_ANCHORS["observed_event_rate_hz"]) - sigma**2 / 2
    rates = rng.lognormal(mean=mean_log_rate, sigma=sigma, size=args.nodes).astype(np.float32)
    return positions, rates


def normalized_adjacency(positions, neighbors):
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    scale = np.median(distances[distances > 0])
    adjacency = np.zeros_like(distances, dtype=np.float32)
    for node_index in range(len(positions)):
        nearest = np.argsort(distances[node_index])[1 : neighbors + 1]
        adjacency[node_index, nearest] = np.exp(-distances[node_index, nearest] / scale)
    adjacency = np.maximum(adjacency, adjacency.T)
    adjacency += np.eye(len(positions), dtype=np.float32)
    degree = np.sum(adjacency, axis=1)
    degree_inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-6))
    return (degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]).astype(np.float32)


def poisson_background(node_rates, window_s, rng):
    counts = rng.poisson(np.maximum(node_rates * window_s, 0.01)).astype(np.float32)
    coincident = rng.binomial(
        np.maximum(counts.astype(int), 0),
        REAL_DATA_ANCHORS["observed_coincident_rate"],
    ).astype(np.float32)
    return counts, coincident


def simulate_dataset(args, positions, node_rates, rng):
    features = np.zeros((args.graphs, args.nodes, len(FEATURE_NAMES)), dtype=np.float32)
    graph_labels = np.zeros(args.graphs, dtype=np.float32)
    node_labels = np.zeros((args.graphs, args.nodes), dtype=np.float32)
    event_metadata = []
    window_ms = args.window_s * 1000.0
    c_km_per_ms = 299.792458
    position_center = args.area_km / 2.0

    for graph_index in range(args.graphs):
        is_positive = rng.random() < args.positive_rate
        graph_labels[graph_index] = 1.0 if is_positive else 0.0

        counts, coincident = poisson_background(node_rates, args.window_s, rng)
        mean_adc = (
            249.4
            + 145.0 * np.divide(coincident, np.maximum(counts, 1.0))
            + rng.normal(0.0, 25.0, size=args.nodes)
        )
        max_adc = mean_adc + rng.gamma(shape=2.0, scale=65.0, size=args.nodes)
        relative_time_ms = np.where(
            counts > 0,
            rng.uniform(0.0, window_ms, size=args.nodes),
            window_ms,
        )
        burst_score = rng.gamma(shape=1.0, scale=0.08, size=args.nodes)

        injected_nodes = np.zeros(args.nodes, dtype=bool)
        event_time_ms = None
        direction = None
        core = None

        if is_positive:
            core = rng.uniform(0.0, args.area_km, size=2)
            core_distance = np.linalg.norm(positions - core[None, :], axis=1)
            hit_probability = 0.03 + 0.55 * np.exp(
                -(core_distance**2) / (2.0 * args.core_sigma_km**2)
            )
            injected_nodes = rng.random(args.nodes) < hit_probability
            if injected_nodes.sum() < 4:
                injected_nodes[np.argsort(hit_probability)[-4:]] = True

            angle = rng.uniform(0.0, 2.0 * math.pi)
            direction = np.array([math.cos(angle), math.sin(angle)])
            projected_delay_ms = (positions @ direction) / c_km_per_ms
            event_time_ms = rng.uniform(0.2 * window_ms, 0.8 * window_ms)
            arrival_ms = (
                event_time_ms
                + projected_delay_ms
                + rng.normal(0.0, args.timing_jitter_ms, size=args.nodes)
            )
            first_arrival = float(np.min(arrival_ms[injected_nodes]))

            extra_counts = rng.poisson(0.5, size=args.nodes).astype(np.float32) + 1.0
            counts[injected_nodes] += extra_counts[injected_nodes]
            coincident[injected_nodes] += 1.0
            max_adc[injected_nodes] = np.maximum(
                max_adc[injected_nodes],
                rng.normal(610.0, 170.0, size=int(injected_nodes.sum())),
            )
            mean_adc[injected_nodes] += rng.normal(
                28.0,
                18.0,
                size=int(injected_nodes.sum()),
            )
            relative_time_ms[injected_nodes] = np.maximum(
                0.0,
                arrival_ms[injected_nodes] - first_arrival,
            )
            relative_time_ms[~injected_nodes] = np.minimum(
                relative_time_ms[~injected_nodes],
                window_ms,
            )
            burst_score[injected_nodes] += np.divide(
                extra_counts[injected_nodes],
                np.sqrt(np.maximum(counts[injected_nodes], 1.0)),
            )
            node_labels[graph_index, injected_nodes] = 1.0
        else:
            noisy_burst_count = int(rng.integers(0, 4))
            if noisy_burst_count:
                noisy_nodes = rng.choice(args.nodes, size=noisy_burst_count, replace=False)
                burst_score[noisy_nodes] += rng.uniform(0.4, 1.2, size=noisy_burst_count)
                max_adc[noisy_nodes] += rng.normal(180.0, 75.0, size=noisy_burst_count)
            if rng.random() < 0.35:
                false_core = rng.uniform(0.0, args.area_km, size=2)
                false_distance = np.linalg.norm(positions - false_core[None, :], axis=1)
                false_count = int(rng.integers(3, 8))
                false_nodes = np.argsort(false_distance)[:false_count]
                counts[false_nodes] += rng.poisson(0.5, size=false_count).astype(np.float32) + 1.0
                coincident[false_nodes] += rng.binomial(1, 0.55, size=false_count)
                relative_time_ms[false_nodes] = np.abs(
                    rng.normal(0.0, args.timing_jitter_ms * 2.5, size=false_count)
                )
                burst_score[false_nodes] += rng.uniform(0.25, 0.9, size=false_count)
                max_adc[false_nodes] += rng.normal(130.0, 80.0, size=false_count)

        coincident_fraction = np.divide(coincident, np.maximum(counts, 1.0))
        clipped_mean_adc = np.clip(mean_adc, 0.0, 4095.0)
        clipped_max_adc = np.clip(max_adc, 0.0, 4095.0)
        features[graph_index, :, 0] = np.log1p(counts)
        features[graph_index, :, 1] = coincident_fraction
        features[graph_index, :, 2] = clipped_mean_adc / 4095.0
        features[graph_index, :, 3] = clipped_max_adc / 4095.0
        features[graph_index, :, 4] = np.clip(relative_time_ms / window_ms, 0.0, 1.0)
        features[graph_index, :, 5] = np.clip(burst_score, 0.0, 5.0)
        features[graph_index, :, 6] = (positions[:, 0] - position_center) / args.area_km
        features[graph_index, :, 7] = (positions[:, 1] - position_center) / args.area_km

        event_metadata.append(
            {
                "positive": bool(is_positive),
                "injected_node_count": int(injected_nodes.sum()),
                "event_time_ms": None if event_time_ms is None else float(event_time_ms),
                "core_km": None if core is None else [float(core[0]), float(core[1])],
                "direction": None
                if direction is None
                else [float(direction[0]), float(direction[1])],
            }
        )

    return features, graph_labels, node_labels, event_metadata


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
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


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
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1_score), 4),
        "auc": round(float(roc_auc(labels, scores)), 4),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def best_threshold(labels, scores):
    candidates = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 97)))
    best_candidate = 0.5
    best_f1_score = -1.0
    for candidate in candidates:
        f1_score = binary_metrics(labels, scores, candidate)["f1"]
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_candidate = float(candidate)
    return best_candidate


if torch is not None:

    class DenseMessagePassingGNN(torch.nn.Module):
        def __init__(self, feature_count, hidden_count, layer_count):
            super().__init__()
            self.input_layer = torch.nn.Linear(feature_count, hidden_count)
            self.message_layers = torch.nn.ModuleList(
                torch.nn.Linear(hidden_count, hidden_count) for _ in range(layer_count)
            )
            self.readout = torch.nn.Sequential(
                torch.nn.Linear(hidden_count * 2, hidden_count),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_count, 1),
            )

        def forward(self, node_features, adjacency):
            hidden = torch.relu(self.input_layer(node_features))
            for layer in self.message_layers:
                messages = torch.einsum("ij,bjh->bih", adjacency, hidden)
                hidden = torch.relu(layer(messages) + hidden)
            pooled = torch.cat([hidden.mean(dim=1), hidden.max(dim=1).values], dim=-1)
            return self.readout(pooled).squeeze(-1)


def standardize(train_features, test_features):
    means = train_features.reshape(-1, train_features.shape[-1]).mean(axis=0)
    stds = train_features.reshape(-1, train_features.shape[-1]).std(axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)
    return (
        ((train_features - means) / stds).astype(np.float32),
        ((test_features - means) / stds).astype(np.float32),
        means.astype(np.float32),
        stds.astype(np.float32),
    )


def train_model(args, train_features, train_labels, test_features, test_labels, adjacency):
    if torch is None:
        raise SystemExit("PyTorch is required for gnn_simulation.py")

    torch.manual_seed(args.seed)
    model = DenseMessagePassingGNN(
        feature_count=train_features.shape[-1],
        hidden_count=args.hidden,
        layer_count=args.layers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_x = torch.tensor(train_features, dtype=torch.float32)
    train_y = torch.tensor(train_labels, dtype=torch.float32)
    test_x = torch.tensor(test_features, dtype=torch.float32)
    test_y = torch.tensor(test_labels, dtype=torch.float32)
    adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32)
    history = []

    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(len(train_x))
        model.train()
        losses = []
        for start in range(0, len(train_x), args.batch_size):
            batch_indices = permutation[start : start + args.batch_size]
            logits = model(train_x[batch_indices], adjacency_tensor)
            loss = loss_fn(logits, train_y[batch_indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))

        model.eval()
        with torch.no_grad():
            test_logits = model(test_x, adjacency_tensor)
            test_loss = float(loss_fn(test_logits, test_y).item())
            test_scores = torch.sigmoid(test_logits).numpy()
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(float(np.mean(losses)), 5),
                "test_loss": round(test_loss, 5),
                "test_auc": round(float(roc_auc(test_labels, test_scores)), 5),
            }
        )

    model.eval()
    with torch.no_grad():
        train_scores = torch.sigmoid(model(train_x, adjacency_tensor)).numpy()
        test_scores = torch.sigmoid(model(test_x, adjacency_tensor)).numpy()

    threshold = best_threshold(train_labels, train_scores)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "model": model,
        "train_scores": train_scores,
        "test_scores": test_scores,
        "threshold": threshold,
        "metrics": {
            "train": binary_metrics(train_labels, train_scores, threshold),
            "test": binary_metrics(test_labels, test_scores, threshold),
        },
        "history": history,
        "size": {
            "parameters": int(parameter_count),
            "float32_bytes": int(parameter_count * 4),
            "int8_bytes": int(parameter_count),
        },
    }


def write_plots(args, positions, adjacency, features, labels, node_labels, test_scores, history, test_indices):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(args.plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = []

    plt.figure(figsize=(8, 5))
    plt.plot([item["epoch"] for item in history], [item["test_auc"] for item in history])
    plt.xlabel("epoch")
    plt.ylabel("test AUC")
    plt.title("Simulated GNN training curve")
    plt.tight_layout()
    path = output_dir / "training_curve.png"
    plt.savefig(path, dpi=160)
    plt.close()
    plots.append(str(path))

    positive_scores = test_scores[labels[test_indices].astype(bool)]
    negative_scores = test_scores[~labels[test_indices].astype(bool)]
    plt.figure(figsize=(8, 5))
    plt.hist(negative_scores, bins=40, alpha=0.7, label="simulated background")
    plt.hist(positive_scores, bins=40, alpha=0.7, label="injected ensemble")
    plt.xlabel("GNN score")
    plt.ylabel("graphs")
    plt.title("Simulated graph scores")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "score_histogram.png"
    plt.savefig(path, dpi=160)
    plt.close()
    plots.append(str(path))

    positive_test_indices = [idx for idx in test_indices if labels[idx] == 1]
    if positive_test_indices:
        graph_index = positive_test_indices[0]
        plt.figure(figsize=(6, 6))
        for src in range(len(positions)):
            for dst in range(src + 1, len(positions)):
                if adjacency[src, dst] > 0.02:
                    plt.plot(
                        [positions[src, 0], positions[dst, 0]],
                        [positions[src, 1], positions[dst, 1]],
                        color="lightgray",
                        linewidth=0.5,
                        zorder=0,
                    )
        hit_mask = node_labels[graph_index].astype(bool)
        burst = features[graph_index, :, 5]
        plt.scatter(
            positions[~hit_mask, 0],
            positions[~hit_mask, 1],
            s=35 + 30 * burst[~hit_mask],
            color="#4c78a8",
            label="background node",
        )
        plt.scatter(
            positions[hit_mask, 0],
            positions[hit_mask, 1],
            s=70 + 45 * burst[hit_mask],
            color="#e45756",
            label="injected hit node",
        )
        plt.xlabel("x position (km)")
        plt.ylabel("y position (km)")
        plt.title("Simulated positive graph")
        plt.legend(loc="best")
        plt.tight_layout()
        path = output_dir / "simulated_positive_graph.png"
        plt.savefig(path, dpi=160)
        plt.close()
        plots.append(str(path))

    return plots


def main():
    started_at = time.time()
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    positions, node_rates = generate_layout(args, rng)
    adjacency = normalized_adjacency(positions, args.neighbors)
    features, labels, node_labels, event_metadata = simulate_dataset(
        args,
        positions,
        node_rates,
        rng,
    )

    indices = rng.permutation(args.graphs)
    split = int(0.8 * args.graphs)
    train_indices = indices[:split]
    test_indices = indices[split:]
    train_features_raw = features[train_indices]
    test_features_raw = features[test_indices]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    train_features, test_features, feature_means, feature_stds = standardize(
        train_features_raw,
        test_features_raw,
    )

    result = train_model(
        args,
        train_features,
        train_labels,
        test_features,
        test_labels,
        adjacency,
    )
    plots = write_plots(
        args,
        positions,
        adjacency,
        features,
        labels,
        node_labels,
        result["test_scores"],
        result["history"],
        test_indices,
    )

    output = {
        "simulation_only": True,
        "why_simulation": (
            "The ES index has no synchronized multi-source days and only one "
            "timestamped CosmicWatch node, so this is not a claim about real "
            "network-level CREDO performance."
        ),
        "real_data_anchors": REAL_DATA_ANCHORS,
        "config": vars(args),
        "feature_names": FEATURE_NAMES,
        "detector_layout": {
            "node_count": args.nodes,
            "area_km": args.area_km,
            "mean_background_rate_hz": round(float(np.mean(node_rates)), 4),
            "min_background_rate_hz": round(float(np.min(node_rates)), 4),
            "max_background_rate_hz": round(float(np.max(node_rates)), 4),
        },
        "split": {
            "train_graphs": int(len(train_indices)),
            "test_graphs": int(len(test_indices)),
            "train_positive_rate": round(float(np.mean(train_labels)), 4),
            "test_positive_rate": round(float(np.mean(test_labels)), 4),
        },
        "feature_standardization": {
            "mean": {name: float(value) for name, value in zip(FEATURE_NAMES, feature_means)},
            "std": {name: float(value) for name, value in zip(FEATURE_NAMES, feature_stds)},
        },
        "model": {
            "type": "plain PyTorch dense message-passing GNN",
            "hidden": args.hidden,
            "layers": args.layers,
            "size": result["size"],
        },
        "metrics": result["metrics"],
        "training_history": result["history"],
        "plots": plots,
        "example_positive_event": next(
            (item for item in event_metadata if item["positive"]),
            None,
        ),
        "caveats": [
            "Synthetic labels are injected by construction.",
            "Detector layout, timing jitter, and ensemble hit probability are assumptions.",
            "This prototype validates software mechanics, not astrophysical discovery power.",
            "Publication-grade GNN work requires real synchronized multi-node data or a physics-backed simulator with collaborator agreement.",
        ],
        "runtime_seconds": round(time.time() - started_at, 2),
    }
    with open(args.out, "w") as output_file:
        json.dump(output, output_file, indent=2)

    print("Simulation-only GNN complete.")
    print(f"Test F1: {output['metrics']['test']['f1']}")
    print(f"Test AUC: {output['metrics']['test']['auc']}")
    print(f"Wrote {args.out}")
    print(f"Plots: {', '.join(plots)}")


if __name__ == "__main__":
    main()
