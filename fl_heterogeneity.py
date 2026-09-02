"""Quantify client heterogeneity in the real 8-device CREDO federation
using the 1-Wasserstein distance, following the methodology of Zhang et
al., "Federated Scientific Machine Learning..." (IEEE TNNLS 2025,
doi:10.1109/TNNLS.2025.3580409), which relates non-i.i.d. degree measured
by W1 to federated-vs-centralized performance gaps.

Two views per client pair, both exact 1-D W1 via sorted samples:
  pixel  - W1 between pixel-intensity distributions (image content)
  bright - W1 between per-image total-brightness distributions (morphology
           proxy: point hits vs streaks/clusters)
Output: fl_heterogeneity.json + printed matrix. Run: python3 fl_heterogeneity.py
"""
import json

import numpy as np

z = np.load("legacy_fl_hardware.npz", allow_pickle=True)
dev = [str(d) for d in z["device_ids"]]
K = len(dev)
imgs = []
for i in range(K):
    key = f"train_{i}" if f"train_{i}" in z else f"test_{i}"
    a = z[key].astype(np.float32)
    imgs.append(a.reshape(len(a), -1))
print("clients:", dev, "| sizes:", [len(a) for a in imgs])

def w1(a, b, n=20000):
    rng = np.random.default_rng(0)
    a = a.ravel(); b = b.ravel()
    if len(a) > n: a = rng.choice(a, n, replace=False)
    if len(b) > n: b = rng.choice(b, n, replace=False)
    m = min(len(a), len(b))
    qa = np.quantile(np.sort(a), np.linspace(0, 1, m))
    qb = np.quantile(np.sort(b), np.linspace(0, 1, m))
    return float(np.abs(qa - qb).mean())

Wp = np.zeros((K, K)); Wb = np.zeros((K, K))
for i in range(K):
    for j in range(i + 1, K):
        Wp[i, j] = Wp[j, i] = w1(imgs[i], imgs[j])
        Wb[i, j] = Wb[j, i] = w1(imgs[i].sum(1), imgs[j].sum(1))

def show(name, W):
    print(f"\n{name} (1-Wasserstein):")
    print("        " + " ".join(f"d{d:>6s}" for d in dev))
    for i in range(K):
        print(f"d{dev[i]:>6s} " + " ".join(f"{W[i,j]:7.3f}" for j in range(K)))
    off = W[np.triu_indices(K, 1)]
    print(f"mean {off.mean():.3f}  max {off.max():.3f} "
          f"(pair d{dev[np.unravel_index(W.argmax(), W.shape)[0]]}"
          f"-d{dev[np.unravel_index(W.argmax(), W.shape)[1]]})")

show("pixel-intensity", Wp)
show("per-image brightness", Wb / imgs[0].shape[1])  # normalize by n_pixels
json.dump({"device_ids": dev,
           "w1_pixel": Wp.tolist(),
           "w1_brightness_per_pixel": (Wb / imgs[0].shape[1]).tolist(),
           "method": "exact 1-D W1 on sorted samples; TNNLS 2025 "
                     "doi:10.1109/TNNLS.2025.3580409 methodology"},
          open("fl_heterogeneity.json", "w"), indent=1)
print("\nWROTE fl_heterogeneity.json")
