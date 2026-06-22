#!/usr/bin/env python3
"""Compose a one-page presentation PDF from the verified results (numbers pulled live)."""
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

edge = json.load(open("edge_ai_experiment_full.json"))
fl = json.load(open("fl_simulation_results.json"))

adc = edge["baselines"]["adc_threshold"]["test"]["f1"]
mlp = edge["models"]["mlp"]["test"]["f1"]
snn = edge["models"]["snn"]["test"]["f1"]
snn_bytes = edge["models"]["snn"]["size"]["int8_bytes"]
rows = edge["data"]["rows"]
rate = edge["data"]["coincident_rate"]

cen = fl["centralized"]["test"]["f1"]
iid = fl["iid"]["federated"]["test"]["f1"]
non = fl["non_iid"]["federated"]["test"]["f1"]
loc = fl["non_iid"]["local_only"]["mean_f1"]
comm = fl["communication"]
ratio = comm["federated_vs_centralized_ratio"]
fold = 1.0 / ratio

INK = "#1a1a1a"
BLUE = "#23508c"
GREY = "#555555"

fig = plt.figure(figsize=(8.5, 11))
fig.patch.set_facecolor("white")
gs = GridSpec(6, 2, figure=fig, height_ratios=[0.9, 1.25, 1.25, 1.15, 1.7, 0.45],
              hspace=0.55, wspace=0.12, left=0.07, right=0.93, top=0.96, bottom=0.04)


def panel(ax):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax


# ---- Title ----
ax = panel(fig.add_subplot(gs[0, :]))
ax.text(0, 0.80, "Edge + Network Learning for Low-Cost Cosmic-Ray Sensor Arrays",
        fontsize=13.5, fontweight="bold", color=INK, va="top")
ax.text(0, 0.30, "CosmicWatch / CREDO prototype — verified against the live Elasticsearch index "
        "(credo-detections, 3,437,063 docs)", fontsize=9.5, color=GREY, va="top")
ax.plot([0, 1], [0.02, 0.02], color=BLUE, lw=2)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# ---- Data reality ----
ax = panel(fig.add_subplot(gs[1, 0]))
ax.text(0, 1.0, "Data reality (verified live)", fontsize=11.5, fontweight="bold", color=BLUE, va="top")
ax.text(0, 0.74,
        "•  1 timestamped device  (cosmicwatch-001)\n"
        "•  0 lat/lon on CosmicWatch rows\n"
        "•  0 days with multi-source overlap\n"
        "•  coincident = hardware weak label (intra-unit)",
        fontsize=9.5, color=INK, va="top", linespacing=1.6)
ax.text(0, 0.02, "⇒ supports real edge inference; NOT a real\n   synchronized multi-node network yet",
        fontsize=8.8, color=GREY, va="top", style="italic", linespacing=1.4)

# ---- What's real vs simulated ----
ax = panel(fig.add_subplot(gs[1, 1]))
ax.text(0, 1.0, "Real  vs  Simulation", fontsize=11.5, fontweight="bold", color=BLUE, va="top")
ax.text(0, 0.78,
        "REAL now:\n"
        "  • SNN/MLP edge inference on real events\n"
        "SIMULATION only (mechanics, synthetic labels):\n"
        "  • GNN network correlation\n"
        "  • Federated learning across clients",
        fontsize=9.1, color=INK, va="top", linespacing=1.5)
ax.text(0, -0.06, "Blocker is DATA, not modeling.", fontsize=8.6, color=GREY, va="top", style="italic")

# ---- Edge result ----
ax = panel(fig.add_subplot(gs[2, 0]))
ax.text(0, 1.0, "Real-data edge result", fontsize=11.5, fontweight="bold", color=BLUE, va="top")
ax.text(0, 0.74,
        f"Jan 23–24  ·  {rows:,} events  ·  {rate*100:.1f}% coincident",
        fontsize=8.8, color=INK, va="top")
ax.text(0, 0.52,
        f"  ADC threshold   F1 = {adc}\n"
        f"  Tiny MLP        F1 = {mlp}\n"
        f"  Tiny SNN        F1 = {snn}  ({snn_bytes} B)",
        fontsize=9.2, color=INK, va="top", linespacing=1.5, family="monospace")
ax.text(0, -0.04, "Headline: SNN footprint, not a record metric.", fontsize=8.4, color=GREY,
        va="top", style="italic")

# ---- FL result ----
ax = panel(fig.add_subplot(gs[2, 1]))
ax.text(0, 1.0, "Federated learning (simulation)", fontsize=11.5, fontweight="bold", color=BLUE, va="top")
ax.text(0, 0.72,
        f"  Centralized        F1 = {cen}\n"
        f"  Federated (IID)    F1 = {iid}   ← matches\n"
        f"  Federated (non-IID)F1 = {non}\n"
        f"  Local-only (n-IID) F1 = {loc}   ← collapses",
        fontsize=9.4, color=INK, va="top", linespacing=1.5, family="monospace")
ax.text(0, 0.0, f"~{fold:.0f}× less data moved; no raw data shipped.", fontsize=8.6,
        color=GREY, va="top", style="italic")

# ---- Three claims band ----
ax = panel(fig.add_subplot(gs[3, :]))
ax.text(0, 1.0, "Three defensible claims", fontsize=11.5, fontweight="bold", color=BLUE, va="top")
ax.text(0, 0.66,
        "1.  Federated training recovers centralized accuracy on IID data.\n"
        "2.  Federated beats train-alone under non-IID skew  —  the reason to collaborate.\n"
        f"3.  Federation moves ~{fold:.0f}× less data and never ships raw events  —  bandwidth + privacy.",
        fontsize=9.6, color=INK, va="top", linespacing=1.7)

# ---- Plots ----
for col, name, title in [(0, "plots_fl/fl_convergence.png", "FedAvg vs centralized"),
                         (1, "plots_fl/fl_communication.png", "Communication cost")]:
    ax = fig.add_subplot(gs[4, col])
    ax.imshow(mpimg.imread(name))
    ax.axis("off")
    ax.set_title(title, fontsize=9, color=GREY)

# ---- Footer ----
ax = panel(fig.add_subplot(gs[5, :]))
ax.text(0, 0.9, "Path to publishing", fontsize=10.5, fontweight="bold", color=BLUE, va="top")
ax.text(0, 0.45,
        "Tier A (now): data-readiness + real edge prototype + honest FL/GNN simulations → workshop / demo.    "
        "Tier B (needs data): real synchronized multi-node data OR full CREDO image set → conference / journal.",
        fontsize=8.4, color=INK, va="top", linespacing=1.4, wrap=True)

fig.savefig("CosmicWatch_OnePager.pdf", dpi=200)
print("Wrote CosmicWatch_OnePager.pdf")
