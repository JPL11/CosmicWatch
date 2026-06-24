#!/usr/bin/env python3
"""
Deploy + benchmark the tiny CosmicWatch edge classifier on low-power hardware
(e.g. a Raspberry Pi). Training uses PyTorch on a dev machine; INFERENCE and the
benchmark are pure-numpy (no torch), so they run on any Pi — and a pure-Python
forward is included for MCU/MicroPython targets (Pi Pico).

Workflow:
  dev machine (torch + ES):  python3 pi_benchmark.py --train    # trains, exports model_weights.json, benchmarks
  raspberry pi (numpy only): python3 pi_benchmark.py            # loads model_weights.json, benchmarks

Outputs: model_weights.json (portable weights), pi_benchmark.json (results).
"""
import argparse
import json
import math
import resource
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
WFILE = ROOT / "model_weights.json"
FEATS = ["adc_value", "sipm_mv", "temperature_c", "pressure_pa"]
EVENT_RATE_HZ = 1.3757


# ---------- portable inference (numpy only) ----------
def forward_np(X, w):
    W1 = np.asarray(w["W1"], np.float32); b1 = np.asarray(w["b1"], np.float32)
    W2 = np.asarray(w["W2"], np.float32); b2 = np.asarray(w["b2"], np.float32)
    h = np.maximum(0.0, X @ W1 + b1)
    o = h @ W2 + b2
    return 1.0 / (1.0 + np.exp(-o)).ravel()


# ---------- pure-Python inference (no numpy) -- for MCU / MicroPython ----------
def forward_pure(x, w):
    W1, b1, W2, b2 = w["W1"], w["b1"], w["W2"], w["b2"]
    hid = len(b1)
    h = [max(0.0, sum(x[i] * W1[i][j] for i in range(len(x))) + b1[j]) for j in range(hid)]
    o = sum(h[j] * W2[j][0] for j in range(hid)) + b2[0]
    return 1.0 / (1.0 + math.exp(-o))


def model_size(w):
    n = sum(len(r) for r in w["W1"]) + len(w["b1"]) + sum(len(r) for r in w["W2"]) + len(w["b2"])
    return {"parameters": n, "float32_bytes": n * 4, "int8_bytes": n}


def benchmark(w, dim, n=20000, single=4000):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    forward_np(X[:200], w)  # warmup
    t0 = time.perf_counter()
    for i in range(single):
        forward_np(X[i:i + 1], w)
    per_event_us = (time.perf_counter() - t0) / single * 1e6
    t0 = time.perf_counter()
    forward_np(X, w)
    batched_s = time.perf_counter() - t0
    eps = n / max(1e-9, batched_s)
    # pure-python single-event latency (MCU proxy)
    xl = X[0].tolist()
    forward_pure(xl, w)
    t0 = time.perf_counter()
    for _ in range(2000):
        forward_pure(xl, w)
    pure_us = (time.perf_counter() - t0) / 2000 * 1e6
    return {"per_event_us_numpy": round(per_event_us, 2),
            "throughput_eps_numpy": round(eps, 1),
            "per_event_us_pure_python": round(pure_us, 2),
            "headroom_vs_event_rate": round(eps / EVENT_RATE_HZ, 0)}


def read_sensors():
    """Best-effort on-device sensors (Pi/Linux); fields absent where unavailable."""
    import shutil
    import subprocess
    s = {}
    for path, key, div in [("/sys/class/thermal/thermal_zone0/temp", "cpu_temp_c", 1000.0),
                           ("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "cpu_freq_mhz", 1000.0)]:
        try:
            s[key] = round(int(open(path).read().strip()) / div, 1)
        except Exception:
            pass
    if shutil.which("vcgencmd"):  # Raspberry Pi specific
        for cmd in ("measure_temp", "measure_volts core", "get_throttled"):
            try:
                s[cmd.replace(" ", "_")] = subprocess.run(["vcgencmd", *cmd.split()],
                                                          capture_output=True, text=True, timeout=5).stdout.strip()
            except Exception:
                pass
    return s


def power_profile(w, dim, duration, idle_watts, load_watts):
    """Sustained-load throughput + sensors; energy/inference when a measured wattage is given."""
    before = read_sensors()
    rng = np.random.default_rng(1)
    X = rng.standard_normal((4096, dim)).astype(np.float32)
    n = 0
    t_end = time.perf_counter() + duration
    while time.perf_counter() < t_end:
        forward_np(X, w)
        n += len(X)
    eps = n / duration
    out = {"sustained_throughput_eps": round(eps, 1), "duration_s": duration,
           "sensors_before": before, "sensors_after": read_sensors()}
    if load_watts is not None:
        out["load_watts"] = load_watts
        out["energy_per_inference_uJ"] = round(load_watts / max(1e-9, eps) * 1e6, 4)
        if idle_watts is not None:
            out["idle_watts"] = idle_watts
            out["active_watts"] = round(load_watts - idle_watts, 3)
            out["active_energy_per_inference_uJ"] = round((load_watts - idle_watts) / max(1e-9, eps) * 1e6, 4)
    else:
        out["note"] = "pass --load-watts (and --idle-watts) from a USB power meter to get energy per inference"
    return out


def platform_info():
    try:
        import platform
        return {"machine": platform.machine(), "processor": platform.processor() or "?",
                "python": platform.python_version()}
    except Exception:
        return {}


def train_and_export(per_partition):
    """Train a tiny MLP on a real combined sample (both partitions) and export weights."""
    import torch
    from credo_loader import fetch
    rows = fetch("parsed", max_events=per_partition) + fetch("raw", max_events=per_partition)
    X, y = [], []
    for r in rows:
        adc = r["adc_value"]
        if adc is None or r["coincident"] is None:
            continue
        t = r["temperature_c"]; pr = r["pressure_pa"]; sip = r["sipm_mv"]
        t = t if (t is not None and -50 < t < 80) else np.nan
        pr = pr if (pr is not None and 80000 < pr < 110000) else np.nan
        X.append([adc, sip if sip is not None else np.nan, t, pr]); y.append(1.0 if r["coincident"] else 0.0)
    X = np.array(X, float); y = np.array(y, float)
    med = np.nanmedian(X, 0); X = np.where(np.isnan(X), med, X)
    mu, sd = X.mean(0), X.std(0); sd = np.where(sd < 1e-6, 1, sd); Xs = (X - mu) / sd

    torch.manual_seed(7)
    m = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
    pw = torch.tensor([float(len(y) - y.sum()) / max(1.0, float(y.sum()))])
    lf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(m.parameters(), lr=0.003)
    xt = torch.tensor(Xs, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    for _ in range(15):
        perm = torch.randperm(len(xt))
        for s in range(0, len(xt), 512):
            i = perm[s:s + 512]
            loss = lf(m(xt[i]).squeeze(-1), yt[i]); opt.zero_grad(); loss.backward(); opt.step()
    p = list(m.parameters())
    w = {"W1": p[0].detach().numpy().T.tolist(), "b1": p[1].detach().tolist(),
         "W2": p[2].detach().numpy().T.tolist(), "b2": p[3].detach().tolist(),
         "features": FEATS, "standardize_mean": mu.tolist(), "standardize_std": sd.tolist(),
         "note": "MLP(4->8 ReLU->1 sigmoid); standardize inputs with mean/std before forward"}
    WFILE.write_text(json.dumps(w, indent=2))
    return w, len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="train on real data (needs torch + ES) and export weights")
    ap.add_argument("--per-partition", type=int, default=80000)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--power-duration", type=float, default=2.0, help="sustained-load seconds for the power profile")
    ap.add_argument("--idle-watts", type=float, default=None, help="board idle power (USB meter)")
    ap.add_argument("--load-watts", type=float, default=None, help="board power under inference load (USB meter)")
    ap.add_argument("--out", default="pi_benchmark.json")
    args = ap.parse_args()

    trained_on = None
    if args.train:
        print("Training tiny MLP on combined sample ...")
        w, trained_on = train_and_export(args.per_partition)
        print(f"Exported weights to {WFILE.name} (trained on {trained_on:,} events)")
    else:
        if not WFILE.exists():
            raise SystemExit(f"No {WFILE.name}; run once with --train on a machine that has torch + ES.")
        w = json.loads(WFILE.read_text())

    size = model_size(w)
    dim = len(w["W1"])
    bench = benchmark(w, dim=dim, n=args.n)
    power = power_profile(w, dim, args.power_duration, args.idle_watts, args.load_watts)
    out = {"platform": platform_info(), "model": {"arch": "MLP(4->8->1)", **size},
           "trained_on_events": trained_on, "benchmark": bench, "power": power,
           "interpretation": (
               f"{size['int8_bytes']} bytes (int8). numpy inference "
               f"{bench['per_event_us_numpy']} us/event, {bench['throughput_eps_numpy']:,.0f} events/s "
               f"= {bench['headroom_vs_event_rate']:,.0f}x the {EVENT_RATE_HZ} Hz detector rate. "
               "Re-run on the target Raspberry Pi for on-device numbers.")}
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"platform: {out['platform'].get('machine')}  python {out['platform'].get('python')}")
    print(f"model: {size['parameters']} params, {size['int8_bytes']} B (int8)")
    print(f"numpy: {bench['per_event_us_numpy']} us/event, {bench['throughput_eps_numpy']:,.0f} ev/s "
          f"({bench['headroom_vs_event_rate']:,.0f}x rate)")
    print(f"pure-python (MCU proxy): {bench['per_event_us_pure_python']} us/event")
    if "energy_per_inference_uJ" in power:
        print(f"power: {power.get('load_watts')} W load -> {power['energy_per_inference_uJ']} uJ/inference")
    else:
        print(f"power: sustained {power['sustained_throughput_eps']:,.0f} ev/s; "
              "pass --load-watts (USB meter) for energy/inference")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
