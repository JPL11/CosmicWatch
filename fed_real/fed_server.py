"""FedAvg server for the real-hardware federation experiment.

Synchronous FedAvg over persistent TCP connections. Each physical client
host registers with the logical device indices it will train; per round
the server broadcasts the global state, waits for all updates, averages
weighted by sample count, and evaluates on the pooled held-out test
sets. Records per-round wall time, per-host arrival times (straggler
gap), and bytes on the wire. Usage:
  python fed_server.py --hosts 3 --rounds 6 --data legacy_fl_hardware.npz
"""
import argparse
import json
import socket
import time

import numpy as np
import torch

from fed_common import (model, weighted_loss, send_msg, recv_msg,
                        eval_sets, tcp_path_stats, set_cc)

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=29500)
ap.add_argument("--hosts", type=int, default=3)
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--data", default="legacy_fl_hardware.npz")
ap.add_argument("--out", default="fed_real_results.json")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--seconds", type=float, default=0.0,
                help="wall-clock budget: stop after the round that crosses it "
                     "(0 = run exactly --rounds)")
ap.add_argument("--cc", default="", help="TCP congestion control for the "
                "server (downlink) sockets ('' = system default)")
ap.add_argument("--dead-after", type=float, default=120.0,
                help="a host silent this long aborts the run (results so far "
                     "are written with 'aborted')")
args = ap.parse_args()

z = np.load(args.data, allow_pickle=True)
n_clients, test_all, val_all = eval_sets(z)

torch.manual_seed(args.seed)
net = model()

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("0.0.0.0", args.port))
srv.listen(max(8, args.hosts))
print(f"listening :{args.port}, waiting for {args.hosts} hosts", flush=True)

conns = []
while len(conns) < args.hosts:
    c, addr = srv.accept()
    c.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    hello, _ = recv_msg(c)
    hello["cc_server"] = set_cc(c, args.cc)
    c.settimeout(args.dead_after)
    conns.append((c, hello))
    print(f"registered {hello['host']} clients={hello['clients']} from {addr[0]}",
          flush=True)

log = {"hosts": [h for _, h in conns], "rounds": [], "bytes_total": 0,
       "data": args.data, "n_clients": n_clients, "seed": args.seed,
       "budget_s": args.seconds,
       "cc": {h["host"]: {"client": h.get("cc"), "server": h["cc_server"]}
              for _, h in conns}, "has_val": val_all is not None}
t_exp = time.time()
for rnd in range(1, args.rounds + 1):
    if args.seconds and time.time() - t_exp >= args.seconds:
        break
    state = {k: v.cpu() for k, v in net.state_dict().items()}
    t0 = time.time()
    updates, arrivals = [], {}
    try:
        for c, h in conns:
            log["bytes_total"] += send_msg(c, {"round": rnd, "state": state})
        for c, h in conns:
            msg, nbytes = recv_msg(c)
            log["bytes_total"] += nbytes
            arrivals[h["host"]] = time.time() - t0
            updates.append(msg)
    except (OSError, ConnectionError, EOFError) as e:
        # a synchronous round cannot complete without every host
        log["aborted"] = {"round": rnd, "host": h["host"],
                          "error": f"{type(e).__name__}: {e}"}
        print(json.dumps({"aborted": log["aborted"]}), flush=True)
        break
    total_n = sum(u["n"] for u in updates)
    new_state = {}
    for k in state:
        new_state[k] = sum(u["state"][k].double() * (u["n"] / total_n)
                           for u in updates).to(state[k].dtype)
    net.load_state_dict(new_state)
    with torch.no_grad():
        mse = float(weighted_loss(net(test_all), test_all))
        vmse = (None if val_all is None
                else float(weighted_loss(net(val_all), val_all)))
    wall = time.time() - t0
    arr = sorted(arrivals.values())
    rec = {"round": rnd, "wall_s": round(wall, 3),
           "straggler_gap_s": round(arr[-1] - arr[0], 3),
           "arrivals_s": {k: round(v, 3) for k, v in arrivals.items()},
           "client_timings": {u["host"]: u["timing"] for u in updates},
           "global_test_weighted_mse": round(mse, 6),
           "tcp_path": {h["host"]: tcp_path_stats(c) for c, h in conns}}
    if vmse is not None:
        rec["global_val_weighted_mse"] = round(vmse, 6)
    log["rounds"].append(rec)
    print(json.dumps(rec), flush=True)

for c, _ in conns:
    try:
        send_msg(c, {"round": -1})
    except OSError:
        pass
    c.close()
log["total_wall_s"] = round(time.time() - t_exp, 3)
log["update_bytes_per_state"] = sum(v.numel() * 4 for v in net.state_dict().values())
json.dump(log, open(args.out, "w"), indent=1)
print("WROTE", args.out, flush=True)
