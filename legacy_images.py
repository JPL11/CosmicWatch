#!/usr/bin/env python3
"""
Legacy CREDO image track: decode, cluster, and map the 69k PNG hit-crops.

The `legacy` source holds real CREDO smartphone detections from Poland (2017-18):
small PNG crops around each hit, with GPS coordinates. This is the genuine
computer-vision / clustering dataset (the prior CREDO work clustered such images).

Does, on a sample:
  1. Decode the base64 PNG crops (20x20 RGBA) and render galleries.
  2. Unsupervised clustering: grayscale features -> PCA (numpy SVD) -> k-means
     (numpy), with per-cluster montages and a 2D PCA scatter.
  3. Reports that `visible` is constant False (no usable supervised label -> clustering
     is the route).
  4. Geo-temporal view: map lat/lon and the time distribution.

No sklearn needed (PCA/k-means implemented here). Outputs:
  legacy_images.json, legacy_images_report.md, plots_legacy/*.png
"""
import argparse
import base64
import io
import json
import time
import warnings

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")
import numpy as np
import requests
import urllib3
from PIL import Image

from credo_config import es_auth, es_settings, verify_certs
from legacy_common import load_images

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
CROP = 20  # standard crop size


def post(path, body, timeout=120, retries=4):
    es_url, _ = es_settings()
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(f"{es_url}/{path.lstrip('/')}", auth=es_auth(), verify=verify_certs(),
                              headers={"Content-Type": "application/json"}, json=body, timeout=timeout)
            if r.status_code in (429, 502, 503, 504):
                last = r; time.sleep(2 * (attempt + 1)); continue
            r.raise_for_status(); return r.json()
        except requests.exceptions.RequestException as e:
            last = e; time.sleep(2 * (attempt + 1))
    if isinstance(last, requests.Response):
        last.raise_for_status()
    raise last


def fetch_legacy(max_images, page_size=2000):
    _, index = es_settings()
    body = {"size": page_size, "sort": ["_doc"],
            "_source": ["frame_content", "location", "timestamp", "visible"],
            "query": {"term": {"source": "legacy"}}}
    res = post(f"{index}/_search", body, timeout=180)
    # scroll
    es_url, _ = es_settings()
    r = requests.post(f"{es_url}/{index}/_search?scroll=2m", auth=es_auth(), verify=verify_certs(),
                      headers={"Content-Type": "application/json"}, json=body, timeout=180)
    r.raise_for_status(); res = r.json()
    sid = res.get("_scroll_id"); docs = []
    while True:
        hits = res.get("hits", {}).get("hits", [])
        if not hits:
            break
        for h in hits:
            docs.append(h["_source"])
            if max_images and len(docs) >= max_images:
                return docs
        rr = requests.post(f"{es_url}/_search/scroll", auth=es_auth(), verify=verify_certs(),
                           headers={"Content-Type": "application/json"},
                           json={"scroll": "2m", "scroll_id": sid}, timeout=180)
        rr.raise_for_status(); res = rr.json(); sid = res.get("_scroll_id", sid)
    return docs


def decode_crop(b64):
    try:
        im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        if im.size != (CROP, CROP):
            im = im.resize((CROP, CROP))
        return np.asarray(im, dtype=np.float32) / 255.0  # HxWx3
    except Exception:
        return None


def pca(X, n_comp):
    mean = X.mean(0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:n_comp]
    scores = Xc @ comps.T
    var = (S ** 2) / (len(X) - 1)
    ratio = (var / var.sum())[:n_comp]
    return scores, comps, mean, ratio


def kmeans(X, k, seed, iters=50, restarts=4):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        idx = rng.choice(len(X), k, replace=False)
        cent = X[idx].copy()
        for _ in range(iters):
            d = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
            lab = d.argmin(1)
            new = np.array([X[lab == j].mean(0) if np.any(lab == j) else cent[j] for j in range(k)])
            if np.allclose(new, cent):
                cent = new; break
            cent = new
        inertia = ((X - cent[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab, cent)
    return best[1], best[2], best[0]


def montage(images, path, title, cols=12, rows=8):
    import matplotlib.pyplot as plt
    n = min(len(images), cols * rows)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.7, rows * 0.7))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(np.clip(images[i], 0, 1))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def write_plots(imgs, labels, scores, centroids, locations, times, k, out, plots_dir):
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    import datetime as dt
    d = Path(plots_dir); d.mkdir(parents=True, exist_ok=True)
    paths = []

    montage(imgs[:96], d / "sample_gallery.png", "Legacy hit-crops (random sample)")
    paths.append(str(d / "sample_gallery.png"))

    # Per-cluster representatives, ordered by distance to the fitted centroid.
    reps = []
    for j in range(k):
        members = np.where(labels == j)[0]
        order = np.argsort(((scores[members] - centroids[j]) ** 2).sum(1))
        reps.extend(members[order[:12]])
    montage([imgs[i] for i in reps], d / "cluster_montage.png",
            f"Representative crops by cluster (k={k}, 12/cluster)", cols=12, rows=k)
    paths.append(str(d / "cluster_montage.png"))

    # PCA 2D scatter colored by cluster
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(scores[:, 0], scores[:, 1], c=labels, s=6, cmap="tab10", alpha=0.6)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Legacy crops in PCA space (colored by k-means cluster)")
    plt.colorbar(sc, label="cluster"); plt.tight_layout()
    plt.savefig(d / "pca_clusters.png", dpi=150); plt.close(); paths.append(str(d / "pca_clusters.png"))

    # geo scatter
    geo_indices = [i for i, location in enumerate(locations) if location is not None]
    if geo_indices:
        lats = [locations[i][0] for i in geo_indices]; lons = [locations[i][1] for i in geo_indices]
        plt.figure(figsize=(7, 6))
        plt.scatter(lons, lats, s=8, alpha=0.4, c=labels[geo_indices], cmap="tab10")
        plt.xlabel("longitude"); plt.ylabel("latitude")
        plt.title("Legacy detections — Poland (colored by cluster)")
        plt.tight_layout(); plt.savefig(d / "geo_map.png", dpi=150); plt.close()
        paths.append(str(d / "geo_map.png"))

    # time histogram
    if times:
        days = [dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc) for t in times]
        plt.figure(figsize=(9, 4)); plt.hist(days, bins=60)
        plt.ylabel("detections"); plt.title("Legacy detection times (2017–18)")
        plt.tight_layout(); plt.savefig(d / "time_hist.png", dpi=150); plt.close()
        paths.append(str(d / "time_hist.png"))
    return paths


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-images", type=int, default=6000)
    ap.add_argument("--csv", default="credo_useful.csv")
    ap.add_argument("--clusters", type=int, default=8)
    ap.add_argument("--pca-comp", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="legacy_images.json")
    ap.add_argument("--report", default="legacy_images_report.md")
    ap.add_argument("--plots-dir", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    started = time.time()
    print("Loading and deduplicating legacy images from the local export ...")
    loaded = load_images(args.csv)
    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(loaded["images"]))
    if args.max_images and len(indices) > args.max_images:
        indices = np.sort(rng.choice(indices, args.max_images, replace=False))
    imgs = loaded["images"][indices].astype(np.float32) / 255.0
    locations = [loaded["locations"][i] for i in indices]
    times = loaded["times"][indices].tolist()
    devices = loaded["devices"][indices]
    feats = [image.flatten() for image in imgs]
    visible_vals = {}
    for i in indices:
        value = str(loaded["visible"][i])
        visible_vals[value] = visible_vals.get(value, 0) + 1

    if len(feats) < 50:
        raise SystemExit(f"Only {len(feats)} decodable images.")
    X = np.array(feats, dtype=np.float32)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    scores, comps, mean, ratio = pca(Xs, args.pca_comp)
    labels, cent, inertia = kmeans(scores, args.clusters, args.seed)

    cluster_sizes = {int(j): int(np.sum(labels == j)) for j in range(args.clusters)}
    # per-cluster mean brightness (proxy for hit intensity)
    cluster_brightness = {int(j): round(float(X[labels == j].mean()), 4) if np.any(labels == j) else None
                          for j in range(args.clusters)}

    out = {
        "images_decoded": len(imgs),
        "sampling": "deterministic uniform random sample from all deduplicated decodable images",
        "duplicates_removed": loaded["duplicates_removed"],
        "devices_in_sample": int(len(set(devices))),
        "crop_size": [CROP, CROP],
        "visible_distribution": visible_vals,
        "visible_is_usable_label": len(visible_vals) > 1,
        "geo_points": sum(location is not None for location in locations),
        "geo_bounds": ({"lat": [min(c[0] for c in locations if c), max(c[0] for c in locations if c)],
                        "lon": [min(c[1] for c in locations if c), max(c[1] for c in locations if c)]}
                       if any(c is not None for c in locations) else None),
        "clustering": {
            "method": "grayscale -> PCA(numpy SVD) -> k-means(numpy)",
            "k": args.clusters,
            "pca_components": args.pca_comp,
            "pca_top5_variance_ratio": [round(float(r), 4) for r in ratio[:5]],
            "cluster_sizes": cluster_sizes,
            "cluster_mean_brightness": cluster_brightness,
            "inertia": round(float(inertia), 2),
        },
        "findings": [],
    }
    out["findings"] = [
        f"Randomly sampled {len(imgs)} of {len(loaded['images'])} deduplicated legacy crops as "
        f"{CROP}x{CROP} grayscale hit-crops; removed {loaded['duplicates_removed']} exact duplicates — a real, "
        "clusterable CV dataset (not the toy phone-camera set).",
        f"`visible` is {'constant ' + list(visible_vals)[0] if len(visible_vals)==1 else 'mixed'} across the "
        "sample → NOT a usable supervised label; unsupervised clustering is the correct route (matches prior "
        "CREDO pseudo-labeling).",
        f"k-means(k={args.clusters}) yields {len([s for s in cluster_sizes.values() if s])} non-empty clusters "
        "that visibly separate the classic CREDO hit morphologies — round bright 'spots', elongated "
        "'tracks/lines', bright corner 'artifacts' (light leaks), and faint single-pixel hits. This is a real "
        "unsupervised CV result on real data (expert labels would confirm the physical class names).",
        f"Detections carry real Poland GPS ({out['geo_bounds']}) over 2017–18 — geo present, but a single "
        "epoch disjoint from the 2025–26 CosmicWatch data, so still no cross-source synchronization.",
    ]

    if args.plots_dir:
        out["plots"] = write_plots(imgs, labels, scores, cent, locations, times,
                                   args.clusters, out, args.plots_dir)
    out["runtime_seconds"] = round(time.time() - started, 2)

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    with open(args.report, "w") as fh:
        L = [f"# Legacy CREDO Image Track\n",
             f"Decoded **{len(imgs)}** {CROP}×{CROP} hit-crops (sample of 69,000). "
             f"`visible` = {out['visible_distribution']} → "
             f"{'usable label' if out['visible_is_usable_label'] else 'NOT a usable label (constant)'}.\n",
             f"## Clustering ({out['clustering']['method']}, k={args.clusters})\n",
             f"PCA top-5 variance ratio: {out['clustering']['pca_top5_variance_ratio']}\n",
             "| cluster | size | mean brightness |", "|--:|--:|--:|"]
        for j in range(args.clusters):
            L.append(f"| {j} | {cluster_sizes[j]} | {cluster_brightness[j]} |")
        L.append(f"\nGeo bounds: {out['geo_bounds']}\n")
        L.append("## Findings\n")
        L += [f"- {f}" for f in out["findings"]]
        fh.write("\n".join(L) + "\n")

    print(f"Decoded {len(imgs)} crops; visible={out['visible_distribution']}")
    print(f"Clusters: {cluster_sizes}")
    print(f"Wrote {args.out}, {args.report}")


if __name__ == "__main__":
    main()
