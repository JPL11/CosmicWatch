#!/usr/bin/env python3
"""Benchmark a real-device FL client workload on Pi, Jetson, or another Linux host."""
import argparse
import json
import platform
import resource
import shutil
import subprocess
import time

import numpy as np
import torch



def model():
    return torch.nn.Sequential(
        torch.nn.Linear(400, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 8), torch.nn.ReLU(),
        torch.nn.Linear(8, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 400), torch.nn.Sigmoid(),
    )


def weighted_loss(prediction, target):
    return (((prediction - target) ** 2) * (1.0 + 4.0 * target)).mean()


def read_sensors():
    sensors = {}
    for path, key, divisor in (("/sys/class/thermal/thermal_zone0/temp", "cpu_temp_c", 1000.0),
                               ("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "cpu_freq_mhz", 1000.0)):
        try:
            with open(path) as fh: sensors[key] = round(float(fh.read().strip()) / divisor, 2)
        except (OSError, ValueError):
            pass
    if shutil.which("vcgencmd"):
        for command in (("measure_temp",), ("measure_volts", "core"), ("get_throttled",)):
            try:
                sensors["vcgencmd_" + "_".join(command)] = subprocess.run(
                    ["vcgencmd", *command], capture_output=True, text=True, timeout=5).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                pass
    if shutil.which("nvpmodel"):
        try:
            sensors["nvpmodel"] = subprocess.run(
                ["nvpmodel", "-q"], capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return sensors


def synchronize(device):
    if device.type == "cuda": torch.cuda.synchronize()


def train_epoch(net, data, batch_size, learning_rate, device, seed):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    order = torch.randperm(len(data))
    net.train(); total = 0.0
    for start in range(0, len(data), batch_size):
        sample = data[order[start:start + batch_size]].to(device, non_blocking=True)
        loss = weighted_loss(net(sample), sample)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += float(loss.detach()) * len(sample)
    synchronize(device)
    return total / len(data)


def evaluate(net, data, batch_size, device):
    net.eval(); total = 0.0
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            sample = data[start:start + batch_size].to(device, non_blocking=True)
            total += float(weighted_loss(net(sample), sample)) * len(sample)
    synchronize(device)
    return total / len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="legacy_fl_hardware.npz")
    ap.add_argument("--client-index", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=0.002)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--idle-watts", type=float)
    ap.add_argument("--load-watts", type=float)
    ap.add_argument("--out", default="fl_hardware_benchmark.json")
    args = ap.parse_args()
    if args.threads: torch.set_num_threads(args.threads)
    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device("cuda" if use_cuda else "cpu")
    archive = np.load(args.data)
    key = str(args.client_index)
    train = torch.from_numpy(archive[f"train_{key}"].reshape(-1, 400).astype(np.float32) / 255.0)
    test = torch.from_numpy(archive[f"test_{key}"].reshape(-1, 400).astype(np.float32) / 255.0)
    device_id = str(archive["device_ids"][args.client_index])
    net = model().to(device)
    initial = {key: value.detach().clone() for key, value in net.state_dict().items()}
    # Warm up kernels and allocator without including them in the timed workload.
    warm = train[:min(len(train), args.batch_size)].to(device)
    with torch.no_grad(): net(warm)
    synchronize(device); net.load_state_dict(initial)
    sensors_before = read_sensors(); started = time.perf_counter(); history = []
    for round_index in range(args.rounds):
        round_start = time.perf_counter(); losses = []
        for local_epoch in range(args.local_epochs):
            losses.append(train_epoch(net, train, args.batch_size, args.learning_rate, device,
                                      seed=7 + round_index * 100 + local_epoch))
        history.append({"round": round_index + 1,
                        "seconds": round(time.perf_counter() - round_start, 6),
                        "train_weighted_mse": round(losses[-1], 7)})
    synchronize(device); elapsed = time.perf_counter() - started
    test_loss = evaluate(net, test, args.batch_size, device)
    parameters = sum(parameter.numel() for parameter in net.parameters())
    result = {
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "platform": platform.platform(), "python": platform.python_version(),
                     "torch": torch.__version__, "compute_device": str(device),
                     "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                     "torch_threads": torch.get_num_threads()},
        "workload": {"real_device_id": device_id, "train_images": len(train), "test_images": len(test),
                     "rounds": args.rounds, "local_epochs_per_round": args.local_epochs,
                     "batch_size": args.batch_size, "parameters": parameters,
                     "float32_update_bytes": parameters * 4,
                     "raw_grayscale_train_bytes": len(train) * 400},
        "performance": {"total_train_seconds": round(elapsed, 6),
                        "mean_seconds_per_round": round(elapsed / args.rounds, 6),
                        "images_per_second": round(len(train) * args.rounds * args.local_epochs / elapsed, 2),
                        "final_test_weighted_mse": round(test_loss, 7),
                        "max_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
                        "history": history},
        "sensors": {"before": sensors_before, "after": read_sensors()},
        "power": {"measurement_required": args.load_watts is None},
        "interpretation": "This is one real FL client's local workload; FedAvg aggregation is negligible and is not timed.",
    }
    if args.load_watts is not None:
        total_joules = args.load_watts * elapsed
        result["power"].update({"load_watts": args.load_watts,
                                "idle_watts": args.idle_watts,
                                "total_joules": round(total_joules, 4),
                                "joules_per_round": round(total_joules / args.rounds, 4),
                                "joules_per_training_image": round(total_joules /
                                    (len(train) * args.rounds * args.local_epochs), 8)})
        if args.idle_watts is not None:
            active_joules = max(0.0, args.load_watts - args.idle_watts) * elapsed
            result["power"]["active_joules"] = round(active_joules, 4)
            result["power"]["active_joules_per_round"] = round(active_joules / args.rounds, 4)
    with open(args.out, "w") as fh: json.dump(result, fh, indent=2)
    print(f"Wrote {args.out}: {result['performance']['mean_seconds_per_round']} s/round on {device}")


if __name__ == "__main__":
    main()
