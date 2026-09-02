"""Append the Raspberry Pi 5 update pages to hardware_benchmark_brief.pdf.

Reads the per-device result JSONs in this directory and renders two pages
(headline + tables page, charts page), then appends them to the existing
brief. Unlike the original brief generator, this script is committed.
Usage: python3 make_pi5_brief_update.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader, PdfWriter

DEVS = [
    ("Raspberry Pi 5", "pi_benchmark_raspberry_pi_5.json"),
    ("Jetson Orin 15W", "pi_benchmark_jetson_orin_nano_15w.json"),
    ("Jetson Orin 7W", "pi_benchmark_jetson_orin_nano_7w.json"),
    ("Raspberry Pi 4", "pi_benchmark_raspberry_pi_4.json"),
    ("Raspberry Pi 400", "pi_benchmark_raspberry_pi_400.json"),
]
FL = [
    ("Pi 5 (CPU)", "fl_hardware_benchmark_raspberry_pi_5.json"),
    ("Jetson 15W (CPU)", "fl_hardware_benchmark_jetson_15w_cpu.json"),
    ("Jetson 15W (CUDA)", "fl_hardware_benchmark_jetson_15w_cuda.json"),
    ("Pi 4 (CPU)", "fl_hardware_benchmark_raspberry_pi_4.json"),
]
GW = [("Raspberry Pi 4", "event_gateway_benchmark_raspberry_pi_4.json"),
      ("Raspberry Pi 5", "event_gateway_benchmark_raspberry_pi_5.json")]

inf = {n: json.load(open(f)) for n, f in DEVS}
fl = {n: json.load(open(f)) for n, f in FL}
gw = {n: json.load(open(f)) for n, f in GW}

def steady(d):
    h = d["performance"]["history"]
    s = [r["seconds"] for r in h[1:]]
    return sum(s) / len(s)

HEADLINE = """Headline results — Raspberry Pi 5 update (2026-09-02)
 1. The Pi 5 is now the fastest board in the fleet on ALL THREE harnesses:
    inference triage, event gateway, and the FL client workload.
 2. Tiny-classifier inference: 23.7 us/event single-event, 12.6M events/s
    batched (3.0x Pi 4, 1.4x Jetson 15W); headroom vs the 1.38 Hz detector
    rate is ~9.2 million x.
 3. Event gateway: 3.8-4.5x the Pi 4 per policy; hybrid CPU duty at the
    real event rate is 0.016% (Pi 4: 0.041%); headroom 36,215x.
 4. FL client: 0.057 s/round steady-state (mean of rounds 2-6, computed
    uniformly for every device) — faster than the Jetson's own CPU (0.065)
    and 1.6x its CUDA path (0.089). GPU dispatch still does not pay at
    this model size; the Pi 5 extends the CPU's win.
 5. Energy columns for all Pi rows still await an inline USB-C meter
    (Jetson rows use onboard INA sensors). Pi 5 note: inline meters may
    renegotiate 5A PD down to 3A — harmless at these loads.
 Caveat: the Pi 5 gateway run uses the committed cw_gateway_bench.py
 driver; the original Pi 4 stream generator was ad hoc, and selection
 fractions differ (more selected on Pi 5 => timings slightly pessimistic).
 Only CPU cost, throughput, and duty fraction are hardware measurements."""

with PdfPages("/tmp/pi5_update_pages.pdf") as pdf:
    # page 1: headline + tables
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.08, 0.96, "CosmicWatch Edge Hardware Benchmarks — Pi 5 Update",
             fontsize=16, fontweight="bold")
    fig.text(0.08, 0.935, "Five-device matrix after adding Raspberry Pi 5 "
             "(16 GB, Python 3.13, numpy 2.2, torch 2.14 CPU)", fontsize=10)
    fig.text(0.08, 0.66, HEADLINE, fontsize=7.8, family="monospace",
             va="bottom", bbox=dict(boxstyle="round", fc="#f4f6f8",
                                    ec="#c8d0d8"))

    ax1 = fig.add_axes([0.08, 0.40, 0.84, 0.20]); ax1.axis("off")
    ax1.set_title("Table A — Tiny event-classifier inference "
                  "(49-param MLP, int8 weights)", fontsize=10, loc="left")
    rows = []
    for n, _ in DEVS:
        b = inf[n]["benchmark"]
        rows.append([n, f"{b['per_event_us_numpy']:.1f}",
                     f"{b['throughput_eps_numpy']:,.0f}",
                     f"{b['per_event_us_pure_python']:.1f}",
                     f"{b['headroom_vs_event_rate']:,.0f}x"])
    t = ax1.table(cellText=rows, colLabels=["Board", "numpy us/ev",
                  "numpy ev/s", "pure-py us/ev", "headroom vs 1.38 Hz"],
                  loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.4)

    ax2 = fig.add_axes([0.08, 0.20, 0.84, 0.16]); ax2.axis("off")
    ax2.set_title("Table B — FL client (device 13, 6 rounds, 4 threads)",
                  fontsize=10, loc="left")
    rows = []
    for n, _ in FL:
        p = fl[n]["performance"]
        rows.append([n, f"{steady(fl[n]):.3f}",
                     f"{p['mean_seconds_per_round']:.3f}",
                     f"{p['images_per_second']:,.0f}",
                     f"{p['max_rss_mib']:.0f}"])
    t = ax2.table(cellText=rows, colLabels=["Device", "steady s/round",
                  "6-round s/round", "img/s", "peak RSS MiB"],
                  loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.4)

    ax3 = fig.add_axes([0.08, 0.03, 0.84, 0.13]); ax3.axis("off")
    ax3.set_title("Table C — Event gateway, as-fast-as-possible "
                  "(200k synthetic events)", fontsize=10, loc="left")
    rows = []
    for pol in ("coincidence", "adc", "mlp", "hybrid"):
        a4 = gw["Raspberry Pi 4"]["throughput_as_fast_as_possible"][pol]
        a5 = gw["Raspberry Pi 5"]["throughput_as_fast_as_possible"][pol]
        rows.append([pol, f"{a4['cpu_microseconds_per_event']:.1f}",
                     f"{a5['cpu_microseconds_per_event']:.1f}",
                     f"{a4['cpu_microseconds_per_event']/a5['cpu_microseconds_per_event']:.1f}x"])
    t = ax3.table(cellText=rows, colLabels=["policy", "Pi 4 us/ev",
                  "Pi 5 us/ev", "speedup"], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.35)
    pdf.savefig(fig); plt.close(fig)

    # page 2: charts
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
    fig.suptitle("Pi 5 update — cross-device charts", fontsize=13,
                 fontweight="bold")
    names = [n for n, _ in DEVS]
    c = ["#c0392b" if "Pi 5" in n else "#7f8c8d" for n in names]

    ax = axes[0][0]
    v = [inf[n]["benchmark"]["throughput_eps_numpy"] / 1e6 for n in names]
    ax.barh(names[::-1], v[::-1], color=c[::-1])
    ax.set_xlabel("batched inference, M events/s")
    ax.set_title("Inference throughput", fontsize=10)

    ax = axes[0][1]
    v = [inf[n]["benchmark"]["per_event_us_numpy"] for n in names]
    ax.barh(names[::-1], v[::-1], color=c[::-1])
    ax.set_xlabel("single-event latency, us (numpy)")
    ax.set_title("Inference latency", fontsize=10)

    ax = axes[1][0]
    fn = [n for n, _ in FL]
    fc = ["#c0392b" if "Pi 5" in n else "#7f8c8d" for n in fn]
    v = [steady(fl[n]) for n in fn]
    ax.barh(fn[::-1], v[::-1], color=fc[::-1])
    ax.set_xlabel("steady-state seconds per FL round")
    ax.set_title("FL client round time", fontsize=10)

    ax = axes[1][1]
    pols = ["coincidence", "adc", "mlp", "hybrid"]
    x = range(len(pols)); w = 0.38
    a4 = [gw["Raspberry Pi 4"]["throughput_as_fast_as_possible"][p]
          ["cpu_microseconds_per_event"] for p in pols]
    a5 = [gw["Raspberry Pi 5"]["throughput_as_fast_as_possible"][p]
          ["cpu_microseconds_per_event"] for p in pols]
    ax.bar([i - w/2 for i in x], a4, w, label="Pi 4", color="#7f8c8d")
    ax.bar([i + w/2 for i in x], a5, w, label="Pi 5", color="#c0392b")
    ax.set_xticks(list(x)); ax.set_xticklabels(pols, fontsize=8)
    ax.set_ylabel("gateway CPU us/event"); ax.legend(fontsize=8,
                                                    frameon=False)
    ax.set_title("Event gateway per policy", fontsize=10)
    for ax_ in axes.flat:
        ax_.tick_params(labelsize=8)
        ax_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)

# append to the existing brief
writer = PdfWriter()
for page in PdfReader("hardware_benchmark_brief.pdf").pages:
    writer.add_page(page)
for page in PdfReader("/tmp/pi5_update_pages.pdf").pages:
    writer.add_page(page)
with open("hardware_benchmark_brief.pdf", "wb") as fh:
    writer.write(fh)
print("appended 2 Pi 5 update pages to hardware_benchmark_brief.pdf")
