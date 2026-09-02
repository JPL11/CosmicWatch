"""Asynchronous federation server (FedAsync-style) for the real-hardware
experiment.

No rounds and no barriers: each host loops (pull global, train, push)
at its own pace, and the server mixes every incoming update immediately
with a staleness-discounted weight

    alpha = alpha0 / (1 + staleness)**0.5     (Xie et al., FedAsync)

where staleness = global_version_now - version_the_update_started_from.
Runs for a fixed wall-time budget after all hosts register, evaluating
the global model on the pooled test set after every mix, so the output
is an (elapsed_s, mse) trajectory directly comparable to the
synchronous runs. Usage:
  python fed_async_server.py --hosts 3 --seconds 12 --out fed_async.json
"""
import argparse
import json
import socket
import threading
import time

import numpy as np
import torch

from fed_common import model, weighted_loss, send_msg, recv_msg

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=29500)
ap.add_argument("--hosts", type=int, default=3)
ap.add_argument("--seconds", type=float, default=12.0)
ap.add_argument("--alpha0", type=float, default=0.6)
ap.add_argument("--data", default="legacy_fl_hardware.npz")
ap.add_argument("--out", default="fed_async_results.json")
args = ap.parse_args()
torch.set_num_threads(2)

z = np.load(args.data, allow_pickle=True)
tests = [torch.tensor((z[f"test_{i}"].astype("float32") / 255.0)
                      .reshape(len(z[f"test_{i}"]), -1)) for i in range(8)]
test_all = torch.cat(tests)

torch.manual_seed(42)
net = model()
lock = threading.Lock()
version = 0
updates_log = []
bytes_total = [0]

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


def serve(c, hello):
    global version
    host = hello["host"]
    while True:
        with lock:
            state = {k: v.cpu() for k, v in net.state_dict().items()}
            base = version
            stop = time.time() >= deadline
        n = send_msg(c, {"stop": stop, "version": base, "state": state})
        with lock:
            bytes_total[0] += n
        if stop:
            return
        msg, nbytes = recv_msg(c)
        with lock:
            bytes_total[0] += nbytes
            staleness = version - msg["base_version"]
            alpha = args.alpha0 / (1.0 + staleness) ** 0.5
            cur = net.state_dict()
            net.load_state_dict({
                k: (cur[k].double() * (1 - alpha)
                    + msg["state"][k].double() * alpha).to(cur[k].dtype)
                for k in cur})
            version += 1
            with torch.no_grad():
                mse = float(weighted_loss(net(test_all), test_all))
            rec = {"t_s": round(time.time() - t0, 3), "version": version,
                   "host": host, "staleness": staleness,
                   "alpha": round(alpha, 4),
                   "train_s": msg["timing"]["train_s"],
                   "global_test_weighted_mse": round(mse, 6)}
            updates_log.append(rec)
        print(json.dumps(rec), flush=True)


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
       "budget_s": args.seconds, "updates": updates_log,
       "updates_per_host": per_host, "bytes_total": bytes_total[0],
       "total_wall_s": round(time.time() - t0, 3),
       "update_bytes_per_state": sum(v.numel() * 4
                                     for v in net.state_dict().values())}
json.dump(out, open(args.out, "w"), indent=1)
print("WROTE", args.out, "updates_per_host", per_host, flush=True)
