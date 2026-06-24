#!/usr/bin/env python3
"""Generate a comprehensive PDF report (LaTeX -> pdflatex) from all result JSONs."""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent


def load(name):
    try:
        return json.load(open(ROOT / name))
    except Exception:
        return {}


def g(d, path, default=None):
    cur = d
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def c(x):
    return f"{x:,}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)


def esc(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"),
                 ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def fig(rel, caption, width=0.82):
    p = ROOT / rel
    if not p.exists():
        return ""
    return ("\\begin{figure}[h!]\\centering\n"
            f"\\includegraphics[width={width}\\textwidth]{{{rel}}}\n"
            f"\\caption{{{caption}}}\n\\end{{figure}}\n")


DA = load("data_analysis.json")
ED = load("edge_ai_experiment_full.json")
EF = load("edge_efficiency.json")
FL = load("fl_simulation_results.json")
GN = load("gnn_simulation_results.json")
RP = load("rate_physics.json")
AP = load("adc_physics.json")
EC = load("energy_calibration.json")
LG = load("legacy_images.json")
EM = load("event_ml.json")
PB = load("pi_benchmark.json")
EVENT_RATE_HZ = 1.3757

L = []
A = L.append

A(r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
\setlength{\parskip}{0.5em}
\setlength{\parindent}{0pt}
\title{\textbf{CosmicWatch / CREDO Sensor Network:\\ Data Analysis, Edge ML, and Single-Node Physics}}
\author{JPL11 --- prototype workspace}
\date{2026-06-22}
\begin{document}
\maketitle
\begin{abstract}
This report consolidates a full analysis of the CREDO \texttt{credo-detections} Elasticsearch index
and a suite of machine-learning and physics analyses built on it. The central, repeatedly-confirmed
finding is that the usable detector data comes from a \emph{single} CosmicWatch node: it supports
real single-node particle physics and edge-AI prototyping, but not network-scale (graph / ensemble)
physics, which requires multiple synchronized, geo-located detectors. We verify the data behaves
exactly as cosmic-ray physics predicts (Landau energy spectrum, Poisson arrivals, energy-selective
coincidence), calibrate the ADC scale to absolute energy via the minimum-ionizing-particle peak,
demonstrate a tiny quantizable edge classifier, recover the classic CREDO image morphologies by
unsupervised clustering, and report two honest negative ML results. Network-scale GNN and federated
results are included but are explicitly \textbf{simulation-only}.
\end{abstract}
\tableofcontents
\newpage
""")

# 1. Overview
total = g(DA, "total_docs", "?")
A(r"\section{Data Overview}")
A(f"The index \\texttt{{credo-detections}} holds \\textbf{{{total:,}}} documents across 5 sources and "
  "72 mapped fields. The sources are schema-disjoint and, critically, temporally disjoint. "
  "Table~\\ref{tab:sources} summarizes them.")
A(r"""\begin{table}[h!]\centering
\caption{Sources at a glance (verified live).}\label{tab:sources}
\begin{tabular}{lrrl}
\toprule
Source & Documents & Active days & Role \\
\midrule""")
roles = {"cosmicwatch-v3x": "detector events (edge/SNN)", "legacy": "PNG hit-crops + GPS (CV/clustering)",
         "credo.science": "degenerate (0,0 / 0 energy)", "phone-camera": "few images (toy)",
         "credo-science": "tiny image set"}
for s in ["cosmicwatch-v3x", "legacy", "credo.science", "phone-camera", "credo-science"]:
    src = (DA.get("sources") or {}).get(s, {})
    dc = src.get("doc_count", "?")
    ad = (src.get("time_coverage") or {}).get("active_days", "?")
    A(f"\\texttt{{{esc(s)}}} & {c(dc)} & {ad} & {roles[s]} \\\\")
A(r"""\bottomrule\end{tabular}\end{table}""")
A(fig("plots_analysis/source_volumes.png", "Document counts per source (log scale).", 0.6))

# Key data findings
parsed = g(DA, "cosmicwatch_deep_dive.schema_partition.parsed_with_wallclock_timestamp", "?")
raw = g(DA, "cosmicwatch_deep_dive.schema_partition.raw_with_walltime_field", "?")
cw_total = g(DA, "sources.cosmicwatch-v3x.doc_count", "?")
A(r"\subsection{Key data findings and corrections}")
A(r"\begin{itemize}")
A(f"\\item \\textbf{{Single unit, two schemas, both usable (~3.36M events).}} \\texttt{{cosmicwatch-v3x}} "
  f"is one detector; {c(parsed)} \\emph{{parsed}} events carry wall-clock \\texttt{{timestamp}}+"
  f"\\texttt{{coincident}}, and {c(raw)} \\emph{{raw AxLab}} events use \\texttt{{wall\\_time}} (real epoch, "
  "us precision) + \\texttt{coincidence\\_flag}. Both are usable via a canonical loader; still one unit, so "
  "the network/GNN conclusion is unchanged.")
A(r"\item \textbf{\texttt{credo.science} is degenerate:} latitude/longitude all $0,0$, energy all $0$, "
  r"\texttt{particle\_type} constant --- fields present but value-empty.")
A(r"\item \textbf{\texttt{legacy} is the real image asset:} 69{,}000 decodable PNG hit-crops with genuine "
  r"Poland GPS (2017--18), overlooked by earlier work.")
A(r"\item \textbf{Temporally disjoint sources} (2017--18 vs.\ 2025--26) $\Rightarrow$ zero cross-source "
  r"coincidence; cross-source learning is inherently heterogeneous/federated.")
A(r"\end{itemize}")

# 2. Physics
A(r"\section{Single-Node Physics}")
A("The data is verified real cosmic-ray muon data, confirmed four independent ways.")

A(r"\subsection{Energy-deposit spectrum (Landau fit)}")
mpv_all = g(AP, "adc_spectrum.all_moyal.mpv_adc", "?")
r2 = g(AP, "adc_spectrum.all_moyal.r2", "?")
mpv_co = g(AP, "adc_spectrum.coincident_moyal.mpv_adc", "?")
mpv_nc = g(AP, "adc_spectrum.noncoincident_moyal.mpv_adc", "?")
A(f"The ADC value is proportional to energy deposited by ionization. The spectrum fits a Landau (Moyal) "
  f"shape with most-probable value MPV~$\\approx${mpv_all} ADC ($R^2={r2}$) --- the textbook energy-loss "
  f"distribution of a minimum-ionizing muon in thin scintillator. The coincident population sits at "
  f"MPV~$={mpv_co}$ versus non-coincident MPV~$={mpv_nc}$ (Table~\\ref{{tab:adc}}): requiring both panels "
  "selects higher-energy through-going tracks, which is precisely why ADC alone is a strong classifier.")
A(r"""\begin{table}[h!]\centering\caption{ADC (energy) spectrum Landau/Moyal fits.}\label{tab:adc}
\begin{tabular}{lrr}\toprule
Population & MPV (ADC) & interpretation \\ \midrule""")
A(f"All events & {mpv_all} & mixed \\\\")
A(f"Coincident (muon tracks) & {mpv_co} & higher energy deposit \\\\")
A(f"Non-coincident & {mpv_nc} & lower / noise \\\\")
A(r"\bottomrule\end{tabular}\end{table}")
A(fig("plots_adc/adc_landau_fit.png", "ADC energy-deposit spectrum with Landau (Moyal) fit."))

A(r"\subsection{Absolute energy calibration (MIP peak)}")
gain = g(EC, "calibration.gain_keV_per_adc", "?")
empv = g(EC, "anchor.expected_mip_mpv_MeV", "?")
thr = g(EC, "energy_scale.threshold_~52_adc_MeV", "?")
sat = g(EC, "energy_scale.saturation_4095_adc_MeV", "?")
meanE = g(EC, "energy_scale.mean_deposit_MeV", "?")
A(f"Anchoring the coincident MPV ({mpv_co} ADC) to the known minimum-ionizing energy loss "
  f"($\\approx{empv}$~MeV for 1~cm plastic) yields an approximate ADC$\\to$MeV scale "
  f"(Table~\\ref{{tab:cal}}). The resulting numbers are physically sensible: a sub-MeV trigger, a "
  "$\\sim$1.6~MeV muon peak, and a tens-of-MeV ceiling. This is a single-point energy \\emph{scale} "
  "(assumes thickness, ADC linearity, zero pedestal), not a precision calibration.")
A(r"""\begin{table}[h!]\centering\caption{Approximate energy calibration.}\label{tab:cal}
\begin{tabular}{lr}\toprule Quantity & Value \\ \midrule""")
A(f"Gain & {gain} keV/ADC \\\\")
A(f"Trigger threshold ($\\sim$52 ADC) & {thr} MeV \\\\")
A(f"MIP (muon) peak & {empv} MeV \\\\")
A(f"Mean deposit & {meanE} MeV \\\\")
A(f"ADC saturation (4095) & {sat} MeV \\\\")
A(r"\bottomrule\end{tabular}\end{table}")
A(fig("plots_calibration/calibrated_spectrum.png", "Calibrated energy-deposit spectrum (MeV)."))

A(r"\subsection{Rate, timing, and dead-time}")
cv = g(AP, "timing.pico_timing.cv", "?")
verdict = g(AP, "timing.pico_timing.verdict", "?")
rate = g(AP, "timing.pico_timing.implied_rate_hz", g(RP, "rate_summary.mean_rate_hz", "?"))
deadf = g(AP, "timing.dead_time.dead_fraction", "?")
A(f"Using the microsecond \\texttt{{pico\\_timestamp\\_s}} field, the inter-arrival coefficient of "
  f"variation is $\\mathrm{{CV}}={cv}$ ({esc(verdict)}) at a mean rate of $\\approx{rate}$~Hz --- consistent "
  "with the expected sea-level muon flux. This \\textbf{corrects} an earlier $\\mathrm{CV}=0.75$ that was an "
  f"artifact of the 1-second \\texttt{{timestamp\\_ms}} quantization. Detector dead-time is negligible "
  f"(fraction~$\\approx{deadf}$).")
A(fig("plots_adc/interarrival_pico.png", "Inter-arrival distribution (pico timing) vs.\\ exponential (Poisson).", 0.62))

A(r"\subsection{Environmental systematic}")
pr = g(AP, "pressure_confound.simple_pearson_r_rate_pressure", "?")
beta = g(AP, "pressure_confound.standardized_betas.pressure", "?")
A(f"A positive rate--pressure correlation (Pearson $r={pr}$, standardized partial $\\beta={beta}$) "
  "\\emph{survives} controlling for temperature and time, but it has the \\emph{wrong sign} for the known "
  "barometric muon effect (which is negative). Over only $\\sim$16 active days this is most plausibly an "
  "instrumental/site systematic (e.g.\\ SiPM gain coupling or deployment drift), and is flagged rather "
  "than claimed as physics.")

A(r"\subsection{Where quantum mechanics enters}")
A("The measured process is genuinely quantum/particle physics: muon production and decay (weak "
  "interaction; arrivals are quantum-random, hence Poisson), relativistic time dilation (why muons reach "
  "the ground at all), and quantum-limited single-photon detection in the SiPM (the coincidence cut "
  "rejects quantum dark-count noise). The \\emph{analysis methods}, however, are entirely classical "
  "signal processing --- there is no quantum computing or quantum machine learning here.")

# 3. Edge ML
A(r"\section{Edge Machine Learning}")
A(r"\subsection{Supervised classification of the coincidence flag}")
adc_f1 = g(ED, "baselines.adc_threshold.test.f1", "?")
mlp_f1 = g(ED, "models.mlp.test.f1", "?")
snn_f1 = g(ED, "models.snn.test.f1", "?")
snn_b = g(ED, "models.snn.size.int8_bytes", "?")
A(f"On the clean 2026-01-23/24 window ({g(ED,'data.rows','?'):,} events, "
  f"{100*g(ED,'data.coincident_rate',0):.1f}\\% coincident), three models score nearly identically "
  "(Table~\\ref{{tab:edge}}): a tuned ADC threshold, a tiny MLP, and a toy spiking neural network. "
  "Because energy (ADC) is the physical discriminator and \\texttt{coincident} is a weak hardware label, "
  "accuracy is at its physical ceiling --- the SNN's value is its tiny footprint, not a higher score.")
A(r"""\begin{table}[h!]\centering\caption{Supervised coincidence classifier (chronological 80/20 split).}
\label{tab:edge}\begin{tabular}{lrrr}\toprule
Model & Test F1 & Test AUC & Size \\ \midrule""")
A(f"ADC threshold & {adc_f1} & {g(ED,'baselines.adc_threshold.test.auc','?')} & $\\sim$0 bytes \\\\")
A(f"Tiny MLP & {mlp_f1} & {g(ED,'models.mlp.test.auc','?')} & {g(ED,'models.mlp.size.int8_bytes','?')} bytes \\\\")
A(f"Toy SNN & {snn_f1} & {g(ED,'models.snn.test.auc','?')} & {snn_b} bytes \\\\")
A(r"\bottomrule\end{tabular}\end{table}")

A(r"\subsection{Quantization and latency}")
A("Table~\\ref{tab:quant} gives the accuracy/size trade-off under post-training quantization. The MLP "
  "compresses to int4 ($\\sim$33 bytes) for free; below 4-bit it breaks down. Latency is a non-issue "
  "($\\sim$7~\\textmu s/event, millions of times the event rate) --- size/energy is the real axis.")
A(r"""\begin{longtable}{lrrrr}\caption{Quantization sweep (MLP / SNN).}\label{tab:quant}\\ \toprule
Precision & MLP F1 & MLP bytes & SNN F1 & SNN bytes \\ \midrule \endhead""")
msw = {r["bits"]: r for r in g(EF, "models.mlp.quantization.sweep", [])}
ssw = {r["bits"]: r for r in g(EF, "models.snn.quantization.sweep", [])}
for b in [32, 8, 4, 2, 1]:
    m = msw.get(b, {}); s = ssw.get(b, {})
    A(f"{b}-bit & {m.get('f1','?')} & {m.get('bytes','?')} & {s.get('f1','?')} & {s.get('bytes','?')} \\\\")
A(r"\bottomrule\end{longtable}")
A(fig("plots_efficiency/accuracy_vs_size.png", "Accuracy vs.\\ model size under quantization.", 0.66))

A(r"\subsection{Feature study and label-light methods (honest negatives)}")
fa = g(EM, "feature_study.adc_only.f1", "?")
fall = g(EM, "feature_study.all_features.f1", "?")
ssl = g(EM, "self_supervised.linear_probe_f1", "?")
sup = g(EM, "self_supervised.supervised_all_features_f1", "?")
anf = g(EM, "anomaly.flagged_mean_adc", "?")
ano = g(EM, "anomaly.overall_mean_adc", "?")
A(f"\\textbf{{Multimodal features add nothing:}} ADC-only F1~$={fa}$ rises only to {fall} with all "
  f"features --- ADC dominates. \\textbf{{Self-supervision underperforms:}} a frozen-autoencoder linear "
  f"probe reaches F1~$={ssl}$ vs.\\ {sup} supervised, because reconstruction captures variance, not the "
  f"subtle coincidence signal. \\textbf{{Anomaly detection works label-free:}} reconstruction-error "
  f"ranking surfaces the high-energy tail (flagged mean ADC {anf} vs.\\ {ano} overall).")
A(fig("plots_event_ml/feature_subsets.png", "Marginal value of feature subsets over ADC alone.", 0.6))

# 4. Image track
A(r"\section{Image / Clustering Track (\texttt{legacy})}")
ndec = g(LG, "images_decoded", "?")
A(f"We decoded {ndec:,} of the 69{{,}}000 \\texttt{{legacy}} hit-crops (20$\\times$20 PNG). The "
  "\\texttt{visible} flag is constant \\texttt{False} (no usable supervised label), so unsupervised "
  "clustering is the correct route --- matching prior CREDO pseudo-labeling. PCA + $k$-means recovers the "
  "classic CREDO morphologies: round \\emph{spots}, elongated \\emph{tracks}, bright corner "
  "\\emph{artifacts}, and faint single-pixel hits (Fig.~\\ref{fig:montage}).")
mont = fig("plots_legacy/cluster_montage.png", "Representative crops per $k$-means cluster --- distinct CREDO hit morphologies.")
A(mont.replace("\\end{figure}", "\\label{fig:montage}\\end{figure}"))
A(fig("plots_legacy/geo_map.png", "Legacy detections mapped over Poland, colored by cluster.", 0.55))

# 5. Network-scale (simulation only)
A(r"\section{Network-Scale Prototypes (Simulation Only)}")
A(r"\textcolor{red}{\textbf{These results are simulation-only.}} They validate pipeline mechanics, not "
  "real network performance; the labels are synthetic, and the single-node data cannot support a real "
  "graph or federation.")
gf1 = g(GN, "metrics.test.f1", "?"); gauc = g(GN, "metrics.test.auc", "?"); gp = g(GN, "model.size.parameters", "?")
A(f"\\textbf{{GNN (simulated network):}} a {g(GN,'config.nodes','?')}-node synthetic detector array yields "
  f"test F1~$={gf1}$, AUC~$={gauc}$ ({gp:,} parameters) --- high only because positives are injected by "
  "construction.")
cen = g(FL, "centralized.test.f1", "?"); iid = g(FL, "iid.federated.test.f1", "?")
non = g(FL, "non_iid.federated.test.f1", "?"); loc = g(FL, "non_iid.local_only.mean_f1", "?")
ratio = g(FL, "communication.federated_vs_centralized_ratio", "?")
fold = round(1 / ratio) if isinstance(ratio, (int, float)) and ratio else "?"
A(f"\\textbf{{Federated learning (simulated federation over real edge features):}} FedAvg matches "
  f"centralized on IID data (F1 {iid} vs.\\ {cen}) and beats train-alone under non-IID skew ({non} vs.\\ "
  f"{loc} local-only), while moving $\\sim${fold}$\\times$ less data and never shipping raw events "
  "(Table~\\ref{{tab:fl}}).")
A(r"""\begin{table}[h!]\centering\caption{Federated learning (simulation-only).}\label{tab:fl}
\begin{tabular}{lr}\toprule Setting & Test F1 \\ \midrule""")
A(f"Centralized (ships raw data) & {cen} \\\\")
A(f"Federated --- IID & {iid} \\\\")
A(f"Federated --- non-IID & {non} \\\\")
A(f"Local-only --- non-IID & {loc} \\\\")
A(r"\bottomrule\end{tabular}\end{table}")
A(fig("plots_fl/fl_communication.png", "Communication cost: federated updates vs.\\ centralizing raw data.", 0.5))

# 6. Methods rationale
A(r"\section{Methods: Supervised vs.\ Clustering}")
A("We match the method to the data: \\textbf{supervised} where a label exists, "
  "\\textbf{clustering / self-supervision} only where it does not. For the CosmicWatch events we have the "
  "\\texttt{coincident} label and the physics says energy (ADC) is the discriminator, so the task is "
  "essentially a one-feature threshold --- clustering would merely re-derive that threshold without using "
  "the label and score worse. For the \\texttt{legacy} images there are no labels, so clustering is the "
  "right tool. Network correlation (GNN) and the training-distribution layer (FL) are the appropriate "
  "methods for the network problem, but require multi-node data and are simulation-only here.")

# 6b. Edge deployment + flight/satellite outlook
A(r"\section{Edge Deployment and Flight/Satellite Outlook}")
b = g(PB, "model.int8_bytes", "?")
lat = g(PB, "benchmark.per_event_us_numpy", "?")
latp = g(PB, "benchmark.per_event_us_pure_python", "?")
head = g(PB, "benchmark.headroom_vs_event_rate", "?")
A("The classifier exports to portable weights with \\textbf{torch-free} inference (pure numpy, plus a "
  "pure-Python path for MCU/MicroPython), so it runs on a Raspberry Pi or flight computer unchanged. "
  "Dev-machine baseline (Table~\\ref{tab:pi}); re-run on target hardware for on-device latency and, with "
  "a USB power meter, energy per inference (the flight-relevant metric).")
A(r"""\begin{table}[h!]\centering\caption{Edge inference benchmark (x86 baseline).}\label{tab:pi}
\begin{tabular}{lr}\toprule Metric & Value \\ \midrule""")
A(f"Model size & {b} bytes (int8) \\\\")
A(f"Latency (numpy) & {lat} us/event \\\\")
A(f"Latency (pure-Python, MCU proxy) & {latp} us/event \\\\")
A(f"Headroom vs {EVENT_RATE_HZ} Hz event rate & {c(head)}x \\\\")
A(r"\bottomrule\end{tabular}\end{table}")
A("\\textbf{Why this matters off the ground.} On the ground latency/size are non-issues, so the edge "
  "model is optional. At altitude/in space the headline physics is single-point (flux vs altitude — the "
  "Pfotzer maximum; geomagnetic latitude; the South Atlantic Anomaly; dosimetry/LET; solar particle "
  "events), so a \\emph{single} unit does real science, and downlink bandwidth is scarce — making an "
  "onboard tiny classifier for event selection / anomaly detection / data reduction genuinely valuable. "
  "A \\emph{constellation} of GPS-synchronized units would finally supply the spatially-separated, "
  "time-synchronized data the GNN/federated track needs. (See the flight/satellite concept note.)")

# 7. Conclusions
A(r"\section{Conclusions and Path Forward}")
A(r"\begin{itemize}")
A(r"\item \textbf{The data is one validated sensor.} It is real cosmic-ray muon data (Landau spectrum, "
  r"Poisson arrivals, energy-selective coincidence), now calibrated to absolute energy.")
A(r"\item \textbf{What current data supports (Tier A, now):} a data-readiness audit, a calibrated "
  r"single-node physics characterization, a tiny quantizable edge classifier, real-data image clustering, "
  r"and honest method comparisons --- a complete workshop/project package.")
A(r"\item \textbf{What it does not support:} air-shower / Cosmic-Ray-Ensemble physics, a real GNN, or "
  r"directional/lifetime measurements --- all require multiple synchronized, geo-located detectors.")
A(r"\item \textbf{The decision gate:} obtaining synchronized multi-node data is the single thing that "
  r"unlocks network-scale physics and a conference-grade result. Absent that, consolidate into the "
  r"Tier-A writeup.")
A(r"\end{itemize}")

# Appendix: quality flags
A(r"\section*{Appendix: Data-Quality Flags}")
A(r"\begin{itemize}")
for q in g(DA, "quality_flags", [])[:8]:
    A(f"\\item {esc(q)}")
A(r"\end{itemize}")

A(r"\end{document}")

tex = "\n".join(L)
(ROOT / "CosmicWatch_Report.tex").write_text(tex)

env = dict(os.environ, MPLCONFIGDIR="/tmp/matplotlib")
for _ in range(2):
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        "CosmicWatch_Report.tex"], cwd=ROOT, env=env,
                       capture_output=True, text=True)
ok = (ROOT / "CosmicWatch_Report.pdf").exists()
for ext in ("aux", "log", "out", "toc"):
    (ROOT / f"CosmicWatch_Report.{ext}").unlink(missing_ok=True)
if ok:
    print("Wrote CosmicWatch_Report.pdf")
else:
    print("pdflatex FAILED; tail of output:")
    print(r.stdout[-3000:])
