"""Asynchronous federation server (FedAsync-style) for the real-hardware
experiment.

No rounds and no barriers: each host loops (pull global, train, push)
at its own pace, and the server mixes every incoming update immediately
with a staleness-discounted weight

    alpha = alpha0 / (1 + staleness)**0.5     (Xie et al., FedAsync)

where staleness = global_version_now - version_the_update_started_from.
Runs for a fixed wall-time budget after all hosts register, evaluating
the global model on the pooled test set after every mix.

Wire formats (--wire): fp32 | fp16 | int8 | int8ef | int8_delta_ef |
int8_delta2_ef, or --adapt for per-connection ADAPTIVE selection: the
server keeps EWMAs of each connection's communication time vs training
time and walks the client along fp32 -> fp16 -> int8_delta2_ef,
escalating compression when the link dominates (comm > train) and
relaxing it when the link is nearly free (comm < 0.25 * train). No
client needs prior link knowledge.

Link emulation: --link <profile> applies one profile to --link-hosts;
--link-map "pi5=ltem,jetson=nbiot" sets per-host profiles; --link-switch
"host:from:to:t_s" changes a host's profile mid-run (server-directed,
so clients stay unmodified).
Usage:
  python fed_async_server.py --hosts 3 --seconds 12 --out fed_async.json
"""
import argparse
import json
import socket
import threading
import time

import numpy as np
import torch

from fed_common import (model, weighted_loss, send_msg, recv_msg,
                        pack_state, unpack_state, Int8EF, LINK_PROFILES,
                        unpack_topk)

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=29500)
ap.add_argument("--hosts", type=int, default=3)
ap.add_argument("--seconds", type=float, default=12.0)
ap.add_argument("--alpha0", type=float, default=0.6)
ap.add_argument("--wire", default="fp32", choices=["fp32", "fp16", "int8", "int8ef", "int8_delta_ef", "int8_delta2_ef", "topk_ef"])
ap.add_argument("--adapt", action="store_true",
                help="adaptive per-connection wire selection (overrides --wire)")
ap.add_argument("--adapt-low", type=float, default=0.25)
ap.add_argument("--adapt-high", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--link", default="none",
                choices=["none", "lte", "ltem", "nbiot"])
ap.add_argument("--link-hosts", default="pi5,jetson,pi4",
                help="hosts the emulated link applies to")
ap.add_argument("--link-map", default="",
                help='per-host profiles, e.g. "pi5=ltem,jetson=nbiot"')
ap.add_argument("--link-switch", default="",
                help='mid-run change: "host:from:to:t_s"')
ap.add_argument("--link-sched", default="",
                help='piecewise schedule: "host:prof@t0,prof@t1,..." '
                     '(prof "none" = unshaped); overrides --link-switch')
ap.add_argument("--data", default="legacy_fl_hardware.npz")
ap.add_argument("--out", default="fed_async_results.json")
args = ap.parse_args()
torch.set_num_threads(2)

ADAPT_LADDER = ["fp32", "fp16", "int8_delta2_ef"]

z = np.load(args.data, allow_pickle=True)
tests = [torch.tensor((z[f"test_{i}"].astype("float32") / 255.0)
                      .reshape(len(z[f"test_{i}"]), -1)) for i in range(8)]
test_all = torch.cat(tests)

torch.manual_seed(args.seed)
net = model()
lock = threading.Lock()
version = 0
updates_log = []
bytes_total = [0]

link_map = {}
if args.link_map:
    for part in args.link_map.split(","):
        h, p = part.split("=")
        link_map[h] = p
elif args.link != "none":
    for h in args.link_hosts.split(","):
        link_map[h] = args.link
link_sched = {}
if args.link_sched:
    h, segs = args.link_sched.split(":", 1)
    link_sched[h] = []
    for seg in segs.split(","):
        prof, ts = seg.split("@")
        link_sched[h].append((float(ts), prof))
    link_sched[h].sort()
link_switch = None
if args.link_switch:
    h, pfrom, pto, ts = args.link_switch.split(":")
    link_switch = (h, pfrom, pto, float(ts))
    link_map[h] = pfrom

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("0.0.0.0", args.port))
srv.listen(8)
print(f"listening :{args.port}, waiting for {args.hosts} hosts", flush=True)

conns = []
while len(conns) < args.hosts:
    c, addr = srv.accept()
    hello, _ = recv_msg(c)
    conns.append((c, hello))
    print(f"registered {hello['host']} clients={hello['clients']} "
          f"from {addr[0]}", flush=True)

t0 = time.time()
deadline = t0 + args.seconds


def current_link(host):
    prof = link_map.get(host)
    if host in link_sched:
        el = time.time() - t0
        for ts, pr in link_sched[host]:
            if el >= ts:
                prof = None if pr == "none" else pr
    if link_switch and host == link_switch[0]:
        prof = link_switch[2] if time.time() - t0 >= link_switch[3] \
            else link_switch[1]
    return LINK_PROFILES[prof] if prof else None


def serve(c, hello):
    global version
    host = hello["host"]
    level = 0                       # adaptive ladder position
    wire = ADAPT_LADDER[0] if args.adapt else args.wire
    comp = None                     # EF packer for the current wire
    sent_f32 = None                 # client's reconstructed view (delta wires)
    ewma_comm, ewma_train = None, None
    while True:
        with lock:
            base = version
            stop = time.time() >= deadline
            if wire in ("int8ef", "int8_delta_ef", "int8_delta2_ef",
                        "topk_ef") and comp is None:
                comp = Int8EF()
            if wire == "int8_delta2_ef" and sent_f32 is not None:
                cur = net.state_dict()
                delta = {k: cur[k].detach().cpu().float() - sent_f32[k]
                         for k in cur}
                state = comp.pack(delta)
                mode = "delta"
                d = unpack_state(state, "int8")
                sent_f32 = {k: sent_f32[k] + d[k] for k in d}
            else:
                state = (comp.pack(net.state_dict())
                         if wire in ("int8ef", "int8_delta_ef",
                                     "int8_delta2_ef", "topk_ef")
                         else pack_state(net.state_dict(), wire))
                mode = "full"
                if wire in ("int8_delta_ef", "int8_delta2_ef", "topk_ef"):
                    sent_f32 = unpack_state(state, "int8")
        t_send = time.time()
        n = send_msg(c, {"stop": stop, "version": base, "state": state,
                         "wire": wire, "mode": mode,
                         "link": current_link(host)})
        with lock:
            bytes_total[0] += n
        if stop:
            return
        msg, nbytes = recv_msg(c)
        t_recv = time.time()
        with lock:
            bytes_total[0] += nbytes
            staleness = version - msg["base_version"]
            alpha = args.alpha0 / (1.0 + staleness) ** 0.5
            if wire == "topk_ef":
                d = unpack_topk(msg["state"])
                up = {k: sent_f32[k] + d[k] for k in d}
            elif wire in ("int8_delta_ef", "int8_delta2_ef"):
                d = unpack_state(msg["state"], "int8")
                up = {k: sent_f32[k] + d[k] for k in d}
            else:
                up = unpack_state(msg["state"], wire)
            cur = net.state_dict()
            net.load_state_dict({
                k: (cur[k].double() * (1 - alpha)
                    + up[k].double() * alpha).to(cur[k].dtype)
                for k in cur})
            version += 1
            with torch.no_grad():
                mse = float(weighted_loss(net(test_all), test_all))
            rec = {"t_s": round(time.time() - t0, 3), "version": version,
                   "host": host, "staleness": staleness,
                   "alpha": round(alpha, 4), "wire": wire,
                   "train_s": msg["timing"]["train_s"],
                   "global_test_weighted_mse": round(mse, 6)}
            updates_log.append(rec)
        print(json.dumps(rec), flush=True)
        if args.adapt:
            train_s = msg["timing"]["train_s"]
            comm_s = max((t_recv - t_send) - train_s, 0.0)
            ewma_comm = comm_s if ewma_comm is None \
                else 0.5 * ewma_comm + 0.5 * comm_s
            ewma_train = train_s if ewma_train is None \
                else 0.5 * ewma_train + 0.5 * train_s
            new = level
            if ewma_comm > args.adapt_high * ewma_train \
                    and level < len(ADAPT_LADDER) - 1:
                new = level + 1
            elif ewma_comm < args.adapt_low * ewma_train and level > 0:
                new = level - 1
            if new != level:
                level = new
                wire = ADAPT_LADDER[level]
                comp = None
                sent_f32 = None    # forces a full bootstrap on delta entry
                print(json.dumps({"t_s": round(time.time() - t0, 3),
                                  "host": host, "adapt_to": wire,
                                  "ewma_comm_s": round(ewma_comm, 3),
                                  "ewma_train_s": round(ewma_train, 3)}),
                      flush=True)


threads = [threading.Thread(target=serve, args=(c, h)) for c, h in conns]
for t in threads:
    t.start()
for t in threads:
    t.join()
for c, _ in conns:
    c.close()

per_host = {}
for r in updates_log:
    per_host[r["host"]] = per_host.get(r["host"], 0) + 1
out = {"hosts": [h for _, h in conns], "alpha0": args.alpha0,
       "wire": ("adapt" if args.adapt else args.wire), "seed": args.seed,
       "link": args.link, "link_map": link_map,
       "link_switch": args.link_switch, "link_sched": args.link_sched,
       "link_hosts": args.link_hosts,
       "budget_s": args.seconds, "updates": updates_log,
       "updates_per_host": per_host, "bytes_total": bytes_total[0],
       "total_wall_s": round(time.time() - t0, 3),
       "update_bytes_per_state": sum(v.numel() * 4
                                     for v in net.state_dict().values())}
json.dump(out, open(args.out, "w"), indent=1)
print("WROTE", args.out, "updates_per_host", per_host, flush=True)
