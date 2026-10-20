"""Figures for the October 2026 real-WAN campaigns, in the house style of
make_paper_figs.py. Reads results/*.json(.gz) via analyze_2026-10.load and
results/net_probe_2026-10-20.jsonl; writes PDFs into every paper directory
given (default: both WAN drafts).

  python3 make_wan_figs.py [figdir ...]
"""
import importlib.util
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "an", os.path.join(HERE, "analyze_2026-10.py"))
an = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(an)
load = an.load

FIGDIRS = sys.argv[1:] or [
    os.path.expanduser("~/Project/Expanse/paper/fedcw-wan-simreal/figures/"),
    os.path.expanduser("~/Project/Expanse/paper/fedcw-wan-design/figures/")]
BLUE, ORANGE, GREEN, GREY = "#2470a8", "#d1611e", "#2c8a5a", "#6b6b6b"
RED, INK = "#b03a3a", "#1b1b1b"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.2, "mathtext.fontset": "stix", "pdf.fonttype": 42,
    "axes.linewidth": 0.6})
SEEDS = [1, 2, 3, 4, 5]
HOST_ORDER = ["desktop", "pi5", "jetson", "pi5b", "pi4", "pi4b", "pi400",
              "pi3", "pi02w", "nano"]


def style(ax):
    ax.tick_params(labelsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, name):
    for d in FIGDIRS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------
# W1: sync vs async over the WAN (campaign A, clean-power hosts)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
ax = axes[0]
for s in SEEDS:
    a, y = load("A_async_s%d" % s), load("A_sync_s%d" % s)
    lw, al = (1.0, 1.0) if s == 1 else (0.5, 0.35)
    u = a["updates"][::4]
    ax.plot([x["t_s"] for x in u], [x["global_test_weighted_mse"] for x in u],
            lw=lw, alpha=al, color=RED, label="async (FedAsync)" if s == 1 else None)
    t, cum = [], 0.0
    for r in y["rounds"]:
        cum += r["wall_s"]
        t.append((cum, r["global_test_weighted_mse"]))
    ax.plot([x for x, _ in t], [v for _, v in t], lw=lw, alpha=al, color=GREY,
            label="synchronous FedAvg" if s == 1 else None)
ax.set_xlabel("wall-clock seconds"); ax.set_ylabel("global test loss")
ax.set_yscale("log"); ax.set_xlim(0, 120); ax.legend(frameon=False)
ax.set_title("(a) Asynchrony wins on every seed over the WAN", fontsize=7.6)
style(ax)
ax = axes[1]
hosts = ["desktop", "pi5", "jetson", "pi02w", "nano"]
train = {h: [] for h in hosts}
rounds = []
for s in SEEDS:
    for r in load("A_sync_s%d" % s)["rounds"][1:]:
        rounds.append(r["wall_s"])
        for h in hosts:
            train[h].append(r["client_timings"][h]["train_s"])
rmed = st.median(rounds)
xs = range(len(hosts))
ax.bar(xs, [st.median(train[h]) for h in hosts], color=BLUE, width=0.6,
       label="local training (median)")
ax.axhline(rmed, color=RED, lw=1, ls="--", label="sync round (median %.2f s)" % rmed)
ax.set_xticks(list(xs)); ax.set_xticklabels(hosts)
ax.set_ylabel("seconds per round")
ax.legend(frameon=False, loc="upper left")
ax.set_title("(b) A round is priced by the slowest host", fontsize=7.6)
style(ax)
fig.tight_layout()
save(fig, "fig_wan_sync_async.pdf")

# ----------------------------------------------------------------------
# W2: wire ladder over the WAN (campaign B, all 10 hosts)
# ----------------------------------------------------------------------
WIRES = [("fp32", "fp32", GREY), ("int8", "int8 (naive)", RED),
         ("int8_delta2_ef", "delta+EF (both)", BLUE), ("topk_ef", "top-$k$+EF", GREEN)]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
ax = axes[0]
width = 0.2
for i, (w, lab, c) in enumerate(WIRES):
    per = {h: [] for h in HOST_ORDER}
    for s in SEEDS:
        uph = load("B_%s_s%d" % (w, s))["updates_per_host"]
        for h in HOST_ORDER:
            per[h].append(uph.get(h, 0))
    ax.bar([j + (i - 1.5) * width for j in range(len(HOST_ORDER))],
           [st.mean(per[h]) for h in HOST_ORDER], width=width, color=c, label=lab)
ax.set_xticks(range(len(HOST_ORDER))); ax.set_xticklabels(HOST_ORDER, rotation=40)
ax.set_ylabel("updates in 120 s (mean of 5 seeds)")
ax.legend(frameon=False, ncol=2, fontsize=6.3)
ax.set_title("(a) Compression buys participation on every host", fontsize=7.6)
style(ax)
ax = axes[1]
for w, lab, c in WIRES:
    tails = [an.tail_async(load("B_%s_s%d" % (w, s))) for s in SEEDS]
    mb = [load("B_%s_s%d" % (w, s))["bytes_total"] / 1e6 for s in SEEDS]
    ax.errorbar(st.mean(mb), st.mean(tails), yerr=st.stdev(tails), xerr=st.stdev(mb),
                fmt="o", ms=4, color=c, capsize=2, lw=0.8, label=lab)
ax.set_xlabel("MB moved in 120 s (both directions)")
ax.set_ylabel("tail test loss (mean ± seed std)")
ax.legend(frameon=False, fontsize=6.3)
ax.set_title("(b) EF wires match fp32 at a third of the bytes", fontsize=7.6)
style(ax)
fig.tight_layout()
save(fig, "fig_wan_ladder.pdf")

# ----------------------------------------------------------------------
# W3: the path — per-flow ceiling, shared uplink, and what BBR changes
# ----------------------------------------------------------------------
rows = [json.loads(l) for l in open(os.path.join(an.RES, "net_probe_2026-10-20.jsonl"))]
alone = {r["host"]: r["up_mbps"] for r in rows if r["phase"] == "alone"}
conc = {r["host"]: r["up_mbps"] for r in rows if r["phase"] == "concurrent"}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
ax = axes[0]
hs = [h for h in HOST_ORDER if h in alone]
xs = range(len(hs))
ax.bar([x - 0.2 for x in xs], [alone[h] for h in hs], width=0.4, color=BLUE,
       label="one flow, host alone")
ax.bar([x + 0.2 for x in xs], [conc.get(h, 0) for h in hs], width=0.4, color=ORANGE,
       label="all 10 hosts at once (sum %.1f)" % sum(conc.values()))
ax.set_xticks(list(xs)); ax.set_xticklabels(hs, rotation=40)
ax.set_ylabel("upload Mbps (CUBIC, over WireGuard)")
ax.legend(frameon=False, fontsize=6.3)
ax.set_title("(a) A loss-limited per-flow ceiling, not host speed", fontsize=7.6)
style(ax)
ax = axes[1]
labels = ["bulk upload\n(Mbps, pi5)", "FL updates\n(÷100, 10 hosts)",
          "small-Pi comm\n(ms/exchange)"]
cub = [st.mean([9.67, 13.27]), None, None]
bbr = [st.mean([33.81, 30.64]), None, None]
cu, bu, cc_, bc = [], [], [], []
for s in SEEDS:
    for cc, upd, com in (("cubic", cu, cc_), ("bbr", bu, bc)):
        r = load("BBR_%s_s%d" % (cc, s))
        upd.append(len(r["updates"]) / 100)
        com.append(1000 * st.median(u["comm_s"] for u in r["updates"]
                                    if u["host"] in ("pi3", "pi4", "pi4b", "pi400", "pi02w")))
cub[1], bbr[1], cub[2], bbr[2] = st.mean(cu), st.mean(bu), st.mean(cc_), st.mean(bc)
xs = range(3)
ax.bar([x - 0.2 for x in xs], cub, width=0.4, color=GREY, label="CUBIC")
ax.bar([x + 0.2 for x in xs], bbr, width=0.4, color=BLUE, label="BBR")
ax.set_xticks(list(xs)); ax.set_xticklabels(labels, fontsize=6.3)
ax.legend(frameon=False)
ax.set_title("(b) BBR triples bulk throughput, not participation", fontsize=7.6)
style(ax)
fig.tight_layout()
save(fig, "fig_wan_path.pdf")
