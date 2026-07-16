#!/usr/bin/env python3
"""Real-device federated autoencoder experiment on deduplicated legacy hit crops."""
import argparse
import copy
import json
import time
from collections import Counter

import numpy as np
import torch

from legacy_common import load_images


def model():
    return torch.nn.Sequential(
        torch.nn.Linear(400, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 8), torch.nn.ReLU(),
        torch.nn.Linear(8, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 400), torch.nn.Sigmoid(),
    )


def weighted_loss(pred, target):
    return (((pred - target) ** 2) * (1.0 + 4.0 * target)).mean()


def train(state, x, epochs, batch, lr, seed):
    torch.manual_seed(seed)
    net = model()
    net.load_state_dict(state)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    data = torch.from_numpy(x)
    for _ in range(epochs):
        order = torch.randperm(len(data))
        for start in range(0, len(data), batch):
            sample = data[order[start:start + batch]]
            loss = weighted_loss(net(sample), sample)
            opt.zero_grad(); loss.backward(); opt.step()
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def average(states, weights):
    total = float(sum(weights))
    return {key: sum(state[key] * (weight / total) for state, weight in zip(states, weights))
            for key in states[0]}


def evaluate(state, tests):
    net = model(); net.load_state_dict(state); net.eval()
    per_device = {}
    with torch.no_grad():
        for device, values in tests.items():
            data = torch.from_numpy(values)
            per_device[device] = float(weighted_loss(net(data), data))
    values = list(per_device.values())
    return {
        "weighted_mse": round(float(np.average(values, weights=[len(tests[d]) for d in per_device])), 7),
        "per_device_weighted_mse": {k: round(v, 7) for k, v in per_device.items()},
        "worst_device_weighted_mse": round(max(values), 7),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="credo_useful.csv")
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--max-per-client", type=int, default=3000)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="federated_legacy.json")
    ap.add_argument("--report", default="federated_legacy_report.md")
    args = ap.parse_args(); started = time.time()

    loaded = load_images(args.csv)
    counts = Counter(loaded["devices"])
    clients = [device for device, _ in counts.most_common(args.clients)]
    trains, tests = {}, {}
    rng = np.random.default_rng(args.seed)
    flat = loaded["images"].reshape(-1, 400).astype(np.float32) / 255.0
    for device in clients:
        idx = np.where(loaded["devices"] == device)[0]
        idx = idx[np.argsort(loaded["times"][idx])]
        if len(idx) > args.max_per_client:
            idx = np.sort(rng.choice(idx, args.max_per_client, replace=False))
            idx = idx[np.argsort(loaded["times"][idx])]
        split = max(1, int(0.8 * len(idx)))
        trains[device], tests[device] = flat[idx[:split]], flat[idx[split:]]

    torch.manual_seed(args.seed)
    initial = copy.deepcopy(model().state_dict())
    global_state = initial
    history = []
    for round_index in range(args.rounds):
        states = [train(global_state, trains[d], 1, args.batch_size, args.learning_rate,
                        args.seed + round_index * 100 + i) for i, d in enumerate(clients)]
        global_state = average(states, [len(trains[d]) for d in clients])
        history.append({"round": round_index + 1, **evaluate(global_state, tests)})

    pooled = np.concatenate([trains[d] for d in clients])
    centralized = train(initial, pooled, args.rounds, args.batch_size, args.learning_rate, args.seed)
    local_metrics = {}
    for i, device in enumerate(clients):
        state = train(initial, trains[device], args.rounds, args.batch_size, args.learning_rate, args.seed + i)
        local_metrics[device] = evaluate(state, {device: tests[device]})["weighted_mse"]

    parameters = sum(p.numel() for p in model().parameters())
    train_rows = sum(len(v) for v in trains.values())
    raw_bytes = train_rows * 400
    update_bytes = parameters * 4 * len(clients) * args.rounds * 2
    result = {
        "framing": "Real device IDs and real images; self-supervised reconstruction, not physical class labels.",
        "data": {"decoded_unique_images": int(len(flat)), "duplicates_removed": loaded["duplicates_removed"],
                 "selected_clients": clients, "client_rows": {d: int(len(trains[d]) + len(tests[d])) for d in clients}},
        "split": "chronological 80/20 within each real device after deterministic per-device capping",
        "federated": {"rounds": args.rounds, "final": evaluate(global_state, tests), "history": history},
        "centralized": evaluate(centralized, tests),
        "local_only": {"per_device_weighted_mse": local_metrics,
                       "mean_weighted_mse": round(float(np.mean(list(local_metrics.values()))), 7)},
        "communication": {"model_parameters": parameters, "estimated_fedavg_bytes": update_bytes,
                          "raw_grayscale_training_bytes": raw_bytes,
                          "fedavg_to_raw_ratio": round(update_bytes / raw_bytes, 4)},
        "limitations": ["No expert morphology labels are present.",
                        "Reconstruction error measures representation fidelity, not cosmic-ray classification.",
                        "Device clocks and hardware models are not independently validated."],
        "runtime_seconds": round(time.time() - started, 2),
    }
    with open(args.out, "w") as fh: json.dump(result, fh, indent=2)
    with open(args.report, "w") as fh:
        fh.write("# Real-Device Federated Legacy Images\n\n")
        fh.write(f"- {len(clients)} real devices; {train_rows:,} training images; {loaded['duplicates_removed']:,} duplicates removed.\n")
        fh.write(f"- FedAvg weighted MSE: {result['federated']['final']['weighted_mse']}.\n")
        fh.write(f"- Centralized weighted MSE: {result['centralized']['weighted_mse']}.\n")
        fh.write(f"- Mean local-only weighted MSE: {result['local_only']['mean_weighted_mse']}.\n")
        fh.write(f"- Estimated FedAvg/raw byte ratio: {result['communication']['fedavg_to_raw_ratio']}.\n\n")
        fh.write("This is a real-client self-supervised result, not a labeled particle classifier.\n")
    print(f"Wrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()
