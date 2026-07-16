#!/usr/bin/env python3
"""Create a compact, transferable real-device image dataset for FL benchmarks."""
import argparse
from collections import Counter

import numpy as np

from legacy_common import load_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="credo_useful.csv")
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--max-per-client", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="legacy_fl_hardware.npz")
    args = ap.parse_args()
    loaded = load_images(args.csv); rng = np.random.default_rng(args.seed)
    clients = [device for device, _ in Counter(loaded["devices"]).most_common(args.clients)]
    payload = {"device_ids": np.asarray(clients),
               "duplicates_removed": np.asarray([loaded["duplicates_removed"]], np.int64)}
    for client_index, device in enumerate(clients):
        idx = np.where(loaded["devices"] == device)[0]
        if len(idx) > args.max_per_client:
            idx = rng.choice(idx, args.max_per_client, replace=False)
        idx = idx[np.argsort(loaded["times"][idx])]
        split = max(1, int(0.8 * len(idx)))
        payload[f"train_{client_index}"] = loaded["images"][idx[:split]]
        payload[f"test_{client_index}"] = loaded["images"][idx[split:]]
    np.savez_compressed(args.out, **payload)
    print(f"Wrote {args.out} with {len(clients)} real device clients")


if __name__ == "__main__":
    main()
