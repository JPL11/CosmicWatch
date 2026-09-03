"""Generate all figures for the ICAIIC federation paper in the Expanse
house style (data-rich single-row pipeline; Helvetica; stix math;
per-stage colored borders; real data in every panel).

Run from fed_real/ with legacy_fl_hardware.npz and the camp/run JSONs
present. Writes into the paper's figures/ directory.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

FIGDIR = os.path.expanduser("~/Project/Expanse/paper/fedcw-icaiic/figures/")
BLUE, ORANGE, GREEN, GREY = "#2470a8", "#d1611e", "#2c8a5a", "#6b6b6b"
RED, INK = "#b03a3a", "#1b1b1b"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.2, "mathtext.fontset": "stix", "pdf.fonttype": 42,
    "axes.linewidth": 0.6})


def load(f):
    d = json.load(open(f))
    return d, d["updates"]


# ======================================================================
# Fig 1: architecture pipeline (house single-row data-rich style)
# ======================================================================
FW, FH = 7.2, 2.85
fig = plt.figure(figsize=(FW, FH))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1)
bg.axis("off")
FLOW = 0.60
TY = 0.975


def stage_title(xc, text, color=INK, y=TY):
    bg.text(xc, y, text, fontsize=7.6, fontweight="bold", color=color,
            ha="center", va="top")
    bg.plot([xc - 0.028, xc + 0.028], [y - 0.055] * 2, lw=1.2,
            color="0.35", solid_capstyle="butt")


def sub(xc, ytop, text, color="0.4"):
    bg.text(xc, ytop, text, fontsize=6.0, ha="center", va="top",
            color=color, linespacing=1.25)


def arrow(x0, x1, y=FLOW, color="0.45", lw=1.2):
    bg.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                shrinkA=0, shrinkB=0))


def tag(xc, yc, text, fill, txt, w=0.052, h=0.030):
    bg.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h,
                 boxstyle="round,pad=0.002,rounding_size=0.010",
                 facecolor=fill, edgecolor=txt, lw=0.6, zorder=8))
    bg.text(xc, yc, text, ha="center", va="center", fontsize=5.7,
            color=txt, fontweight="bold", zorder=9)


def iso_block(x0, w, h, yc, label, edge, front, top, side, depth_lbl="",
              n_slabs=5, lw=1.1, ty=0.5, label_fs=6.9):
    AR = FW / FH
    dx = 0.014
    dy = dx * AR * 0.72
    fx0, fx1 = x0, x0 + w
    fy0, fy1 = yc - h / 2, yc + h / 2
    bg.add_patch(Polygon([(fx0, fy1), (fx1, fy1), (fx1 + dx, fy1 + dy),
                          (fx0 + dx, fy1 + dy)], closed=True, facecolor=top,
                 edgecolor=edge, lw=lw, zorder=2))
    bg.add_patch(Polygon([(fx1, fy0), (fx1 + dx, fy0 + dy),
                          (fx1 + dx, fy1 + dy), (fx1, fy1)], closed=True,
                 facecolor=side, edgecolor=edge, lw=lw, zorder=2))
    for i in range(1, n_slabs):
        ox, oy = dx * i / n_slabs, dy * i / n_slabs
        bg.plot([fx0 + ox, fx1 + ox], [fy1 + oy, fy1 + oy], color=edge,
                lw=0.4, alpha=0.45, zorder=3, solid_capstyle="butt")
        bg.plot([fx1 + ox, fx1 + ox], [fy0 + oy, fy1 + oy], color=edge,
                lw=0.4, alpha=0.45, zorder=3, solid_capstyle="butt")
    bg.add_patch(Polygon([(fx0, fy0), (fx1, fy0), (fx1, fy1), (fx0, fy1)],
                 closed=True, facecolor=front, edgecolor=edge, lw=lw,
                 zorder=4))
    bg.text((fx0 + fx1) / 2, fy0 + ty * h, label, ha="center", va="center",
            fontsize=label_fs, color=INK, zorder=5, linespacing=1.25)
    if depth_lbl:
        bg.text(fx1 + dx, fy1 + dy + 0.008, depth_lbl, ha="right",
                va="bottom", fontsize=5.7, color=edge, fontweight="bold",
                zorder=5)


# ---- S1: the fleet (real hosts, real train times) ---------------------
s1x, s1w = 0.016, 0.106
hosts = [("workstation x86", 0.010, "#eef3f7"),
         ("Raspberry Pi 5", 0.150, "#eef7ef"),
         ("Raspberry Pi 4", 0.271, "#eef7ef"),
         ("Jetson Orin", 0.160, "#eef7ef")]
hh, gap = 0.078, 0.018
y0 = FLOW + 1.5 * (hh + gap)
tmax = 0.30
for i, (name, tr, fc) in enumerate(hosts):
    yc = y0 - i * (hh + gap)
    bg.add_patch(FancyBboxPatch((s1x, yc - hh / 2), s1w, hh,
                 boxstyle="round,pad=0.002,rounding_size=0.008",
                 facecolor=fc, edgecolor="0.45", lw=0.8, zorder=4))
    bg.text(s1x + 0.005, yc + 0.012, name, fontsize=5.9, ha="left",
            va="center", color=INK, zorder=5)
    bg.add_patch(FancyBboxPatch((s1x + 0.005, yc - 0.024),
                 (s1w - 0.012) * tr / tmax, 0.014,
                 boxstyle="square,pad=0", facecolor=GREY, edgecolor="none",
                 zorder=5))
    bg.text(s1x + s1w - 0.004, yc - 0.017, f"{tr:.2f} s", fontsize=5.0,
            ha="right", va="center", color="0.35", zorder=6)
stage_title(s1x + s1w / 2, "Fleet")
sub(s1x + s1w / 2, y0 - 3 * (hh + gap) - hh / 2 - 0.012,
    "4 hosts, Wi-Fi LAN\nbars: local train s")

# ---- S2: real sensor crops (the federated data) -----------------------
s2x, s2w = 0.158, 0.088
z = np.load("legacy_fl_hardware.npz", allow_pickle=True)
crops = z["test_0"][:9].astype("float32") / 255.0
grid = np.ones((3 * 20 + 2, 3 * 20 + 2))
for i in range(3):
    for j in range(3):
        grid[i * 21:i * 21 + 20, j * 21:j * 21 + 20] = crops[3 * i + j]
h2 = s2w * FW / FH
ax2 = fig.add_axes([s2x, FLOW - h2 / 2, s2w, h2])
ax2.imshow(grid ** 0.45, cmap="magma", interpolation="nearest",
           aspect="auto", vmax=0.85)
ax2.set_xticks([]); ax2.set_yticks([])
for sp in ax2.spines.values():
    sp.set_linewidth(1.0); sp.set_color("0.55")
stage_title(s2x + s2w / 2, "Local data")
sub(s2x + s2w / 2, FLOW - h2 / 2 - 0.028,
    "20$\\times$20 detector crops\n8 client partitions")

# ---- S3: the model (iso block) ---------------------------------------
s3x, s3w = 0.288, 0.080
iso_block(s3x, s3w, 0.30, FLOW, "autoencoder\n400-32-8-32-400", BLUE,
          "#e8f1f8", "#bcd8ee", "#93c0e4", depth_lbl=r"$\times$8", ty=0.45)
tag(s3x + s3w - 0.006, FLOW + 0.15 - 0.004, "26.6k", "#dbe9f4", BLUE)
stage_title(s3x + s3w / 2, "Client update", color=BLUE)
sub(s3x + s3w / 2, FLOW - 0.15 - 0.030, "1 local epoch / exchange\nAdam, weighted MSE")

# ---- S4: wire packer ladder (real measured kB) ------------------------
s4x, s4w = 0.412, 0.104
wires = [("fp32", 106, GREY), ("fp16", 53, GREY),
         ("int8 delta+EF", 27, ORANGE), ("top-k+EF", 21, ORANGE)]
h4 = 0.30
ax4 = fig.add_axes([s4x, FLOW - h4 / 2, s4w, h4])
yy = np.arange(len(wires))[::-1]
ax4.barh(yy, [w[1] for w in wires], height=0.62,
         color=[w[2] for w in wires], alpha=0.85)
for y_, (name, kb, c) in zip(yy, wires):
    ax4.text(kb + 3, y_, f"{name} {kb}", fontsize=5.2, va="center",
             color=INK)
ax4.set_xlim(0, 165); ax4.set_ylim(-0.6, 3.6)
ax4.set_xticks([]); ax4.set_yticks([])
for sp in ax4.spines.values():
    sp.set_linewidth(1.0); sp.set_color(ORANGE)
tag(s4x + 0.024, FLOW - h4 / 2 + 0.022, "EF Eq. (5)", "#fbeadd",
    ORANGE)
stage_title(s4x + s4w / 2, "Wire packer", color=ORANGE)
sub(s4x + s4w / 2, FLOW - h4 / 2 - 0.028, "kB per exchange\nper-connection residual")

# ---- S5: link (real per-profile exchange time, log scale) -------------
s5x, s5w = 0.562, 0.096
profs = [("LAN", 0.02), ("LTE", 0.35), ("LTE-M", 2.9), ("NB-IoT", 43.0)]
h5 = 0.30
ax5 = fig.add_axes([s5x, FLOW - h5 / 2, s5w, h5])
yy = np.arange(len(profs))[::-1]
ax5.barh(yy, [p[1] for p in profs], height=0.62, color=RED, alpha=0.80,
         log=True)
for y_, (name, t) in zip(yy, profs):
    lbl = f"{name} {t:g}s" if t >= 0.1 else f"{name}"
    ax5.text(1.35 * max(t, 0.012), y_, lbl, fontsize=5.2, va="center",
             color=INK)
ax5.set_xlim(0.01, 4000); ax5.set_ylim(-0.6, 3.6)
ax5.set_xticks([]); ax5.set_yticks([])
for sp in ax5.spines.values():
    sp.set_linewidth(1.0); sp.set_color(RED)
stage_title(s5x + s5w / 2, "Link", color=RED)
sub(s5x + s5w / 2, FLOW - h5 / 2 - 0.028,
    "fp32 exchange time (log)\nserver-attached emulation")

# ---- S6: async server (iso block + real staleness histogram) ----------
s6x, s6w = 0.706, 0.070
iso_block(s6x, s6w, 0.24, FLOW + 0.10,
          r"$\alpha = \frac{0.6}{\sqrt{1+s}}$" + "\nEq. (3)",
          BLUE, "#e8f1f8", "#bcd8ee", "#93c0e4", ty=0.5, label_fs=6.4)
_, ups = load("fed_async_heterog.json")
stale = np.array([u["staleness"] for u in ups])
h6 = 0.17
ax6 = fig.add_axes([s6x, FLOW - 0.115 - h6 / 2, s6w, h6])
ax6.hist(np.clip(stale, 0, 40), bins=24, color=BLUE, alpha=0.75, log=True)
ax6.set_xticks([]); ax6.set_yticks([])
for sp in ax6.spines.values():
    sp.set_linewidth(1.0); sp.set_color(BLUE)
ax6.text(0.95, 0.78, r"staleness $s$", transform=ax6.transAxes,
         fontsize=5.2, ha="right", color=BLUE)
stage_title(s6x + s6w / 2, "Async mix", color=BLUE)
sub(s6x + s6w / 2, FLOW - 0.115 - h6 / 2 - 0.026,
    "988 mixes / 12 s\nmax $s$ = 145, no divergence")

# ---- S7: outcome (real 120 s loss trajectories) -----------------------
s7x, s7w = 0.850, 0.118
h7 = 0.36
ax7 = fig.add_axes([s7x, FLOW - h7 / 2, s7w, h7])
for f, c, lbl in (("camp_long_fp32.json", GREY, "fp32"),
                  ("camp_long_int8.json", RED, "int8"),
                  ("camp_long_int8_delta2_ef.json", BLUE, "delta+EF")):
    _, u = load(f)
    u = u[::25]
    ax7.plot([x["t_s"] for x in u],
             [x["global_test_weighted_mse"] for x in u], lw=0.9, color=c)
ax7.set_yscale("log")
ax7.set_xticks([]); ax7.set_yticks([])
for sp in ax7.spines.values():
    sp.set_linewidth(1.0); sp.set_color(GREEN)
ax7.text(0.97, 0.86, "int8", transform=ax7.transAxes, fontsize=5.4,
         color=RED, ha="right", fontweight="bold")
ax7.text(0.97, 0.10, "delta+EF = fp32", transform=ax7.transAxes,
         fontsize=5.4, color=BLUE, ha="right", fontweight="bold")
stage_title(s7x + s7w / 2, "Global loss", color=GREEN)
sub(s7x + s7w / 2, FLOW - h7 / 2 - 0.028,
    "120 s, $\\approx$11k mixes\neval after every mix")

# ---- flow arrows ------------------------------------------------------
arrow(s1x + s1w + 0.004, s2x - 0.004)
arrow(s2x + s2w + 0.004, s3x - 0.004)
arrow(s3x + s3w + 0.018, s4x - 0.004)
arrow(s4x + s4w + 0.004, s5x - 0.004)
arrow(s5x + s5w + 0.004, s6x - 0.004)
arrow(s6x + s6w + 0.018, s7x - 0.004)

# ---- adaptive-policy return path (red, under the row) -----------------
xs6 = s6x + s6w / 2
xs4 = s4x + s4w / 2
ylow = 0.070
bg.plot([xs6, xs6], [FLOW - 0.115 - h6 / 2 - 0.085, ylow], color=RED,
        lw=1.2, solid_capstyle="butt")
bg.plot([xs6, xs4], [ylow, ylow], color=RED, lw=1.2, solid_capstyle="butt")
bg.annotate("", xy=(xs4, FLOW - h4 / 2 - 0.078), xytext=(xs4, ylow),
            arrowprops=dict(arrowstyle="-|>", lw=1.2, color=RED,
                            shrinkA=0, shrinkB=0))
bg.text((xs4 + xs6) / 2, ylow - 0.030,
        r"adaptive policy: $\hat{c} > \hat{g}$ $\Rightarrow$ escalate wire; "
        r"$\hat{c} < 0.25\,\hat{g}$ $\Rightarrow$ relax   (Eqs. 8-9)",
        fontsize=6.0, color=RED, ha="center", va="top")

fig.savefig(FIGDIR + "fig_pipeline.pdf", bbox_inches="tight")
fig.savefig(FIGDIR + "fig_pipeline.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("fig_pipeline written")

# ======================================================================
# Fig 2: trajectories (restyled to house palette)
# ======================================================================
plt.rcParams.update({"font.size": 7.2})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
ax = axes[0]
sync = json.load(open("fed_real_heterog.json"))
t, cum = [], 0.0
for r in sync["rounds"]:
    cum += r["wall_s"]; t.append((cum, r["global_test_weighted_mse"]))
ax.plot([x for x, _ in t], [y for _, y in t], "o-", ms=3, lw=1,
        color=GREY, label="synchronous FedAvg")
_, a = load("fed_async_heterog.json")
ax.plot([u["t_s"] for u in a], [u["global_test_weighted_mse"] for u in a],
        lw=1, color=RED, label="async (FedAsync)")
ax.set_xlabel("wall-clock seconds"); ax.set_ylabel("global test loss")
ax.set_yscale("log"); ax.set_xlim(0, 12)
ax.legend(frameon=False)
ax.set_title("(a) Asynchrony reclaims idle time", fontsize=7.6)
ax = axes[1]
for f, lab, c in (("camp_long_fp32.json", "fp32", GREY),
                  ("camp_long_int8.json", "int8 (naive)", RED),
                  ("camp_long_int8_delta2_ef.json", "delta+EF (both)", BLUE),
                  ("camp_long_topk_ef.json", "top-k+EF", GREEN)):
    _, u = load(f)
    u2 = u[::20]
    ax.plot([x["t_s"] for x in u2],
            [x["global_test_weighted_mse"] for x in u2], lw=1, color=c,
            label=lab)
ax.set_xlabel("wall-clock seconds"); ax.set_ylabel("global test loss")
ax.set_yscale("log"); ax.set_xlim(0, 120)
ax.legend(frameon=False)
ax.set_title("(b) Naive int8 destabilizes; EF wires track fp32",
             fontsize=7.6)
for ax_ in axes:
    ax_.tick_params(labelsize=6.5)
    ax_.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIGDIR + "fig_trajectories.pdf", bbox_inches="tight")
plt.close(fig)
print("fig_trajectories written")

# ======================================================================
# Fig 3: participation bars (restyled)
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.0))
s = json.load(open("camp_summary.json"))["links"]
profs = ["lte", "ltem", "nbiot"]
fp = [s[f"{p}_fp32"]["arm_updates"] for p in profs]
dl = [s[f"{p}_int8_delta2_ef"]["arm_updates"] for p in profs]
ax = axes[0]
x = np.arange(len(profs)); w = 0.36
ax.bar(x - w / 2, fp, w, color=GREY, label="fp32")
ax.bar(x + w / 2, dl, w, color=BLUE, label="delta+EF")
for i, (a_, b_) in enumerate(zip(fp, dl)):
    ax.text(i - w / 2, a_, str(a_), ha="center", va="bottom", fontsize=6)
    ax.text(i + w / 2, b_, str(b_), ha="center", va="bottom", fontsize=6)
ax.set_xticks(x); ax.set_xticklabels(["LTE", "LTE-M", "NB-IoT"])
ax.set_ylabel("constrained-host updates")
ax.legend(frameon=False)
ax.set_title("(a) Compression decides participation", fontsize=7.6)
ax = axes[1]
d = json.load(open("camp4_mixed_fp32.json"))["updates_per_host"]
e = json.load(open("camp4_mixed_adapt.json"))["updates_per_host"]
hostlbl = ["Pi 4\n(LTE)", "Pi 5\n(LTE-M)", "Jetson\n(NB-IoT)"]
keys = ["pi4", "pi5", "jetson"]
fp = [d.get(k, 0) for k in keys]; ad = [e.get(k, 0) for k in keys]
x = np.arange(len(keys))
ax.bar(x - w / 2, fp, w, color=GREY, label="static fp32")
ax.bar(x + w / 2, ad, w, color=RED, label="adaptive")
for i, (a_, b_) in enumerate(zip(fp, ad)):
    ax.text(i - w / 2, a_, str(a_), ha="center", va="bottom", fontsize=6)
    ax.text(i + w / 2, b_, str(b_), ha="center", va="bottom", fontsize=6)
ax.set_xticks(x); ax.set_xticklabels(hostlbl, fontsize=6.5)
ax.set_ylabel("updates (90 s, 4-host)")
ax.legend(frameon=False)
ax.set_title("(b) Adaptive wires, mixed links", fontsize=7.6)
for ax_ in axes:
    ax_.tick_params(labelsize=6.5)
    ax_.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIGDIR + "fig_participation.pdf", bbox_inches="tight")
plt.close(fig)
print("fig_participation written")

# ======================================================================
# Fig 4: adaptive timeline (restyled)
# ======================================================================
d, ups = load("camp2_switch_adapt.json")
lv = {"fp32": 0, "fp16": 1, "int8_delta2_ef": 2}
fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.2), sharex=True,
                         gridspec_kw={"height_ratios": [1.15, 1]})
ax = axes[0]
pi5 = [(u["t_s"], lv[u["wire"]]) for u in ups if u["host"] == "pi5"]
ax.step([t for t, _ in pi5], [l for _, l in pi5], where="post",
        color=RED, lw=1.2)
ax.axvline(45, color=GREY, ls="--", lw=0.8)
ax.text(46.5, 0.35, "link collapse\nLTE $\\to$ NB-IoT", fontsize=5.8,
        color="0.35")
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["fp32", "fp16", "delta+EF"], fontsize=6)
ax.set_ylim(-0.25, 2.35)
ax.set_title("Pi 5 wire selected by the policy", fontsize=7.2)
ax = axes[1]
_, fpu = load("camp2_switch_fp32.json")
t_ad = [u["t_s"] for u in ups if u["host"] == "pi5"]
t_fp = [u["t_s"] for u in fpu if u["host"] == "pi5"]
ax.eventplot([t_fp, t_ad], lineoffsets=[0, 1], linelengths=0.7,
             colors=[GREY, RED], linewidths=0.6)
ax.axvline(45, color=GREY, ls="--", lw=0.8)
ax.set_yticks([0, 1]); ax.set_yticklabels(["static fp32", "adaptive"],
                                          fontsize=6)
ax.set_xlabel("wall-clock seconds", fontsize=7)
ax.set_title("Pi 5 update arrivals", fontsize=7.2)
for ax_ in axes:
    ax_.tick_params(labelsize=6)
    ax_.spines[["top", "right"]].set_visible(False)
fig.tight_layout(h_pad=0.5)
fig.savefig(FIGDIR + "fig_adapt_timeline.pdf", bbox_inches="tight")
plt.close(fig)
print("fig_adapt_timeline written")
