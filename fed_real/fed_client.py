"""Federation client host: trains its assigned logical device partitions
sequentially each round and returns a sample-weighted average update.
Usage: python fed_client.py --server <ip> --host pi5 --clients 0,1,2
"""
import argparse
import socket
import time

import numpy as np
import torch

from fed_common import model, weighted_loss, send_msg, recv_msg

ap = argparse.ArgumentParser()
ap.add_argument("--server", required=True)
ap.add_argument("--port", type=int, default=29500)
ap.add_argument("--host", required=True)
ap.add_argument("--clients", required=True, help="comma-separated device indices")
ap.add_argument("--data", default="legacy_fl_hardware.npz")
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=0.002)
ap.add_argument("--threads", type=int, default=4)
args = ap.parse_args()
torch.set_num_threads(args.threads)

cids = [int(x) for x in args.clients.split(",")]
z = np.load(args.data, allow_pickle=True)
trains = {}
for i in cids:
    key = f"train_{i}" if f"train_{i}" in z else f"test_{i}"
    a = z[key].astype("float32") / 255.0
    trains[i] = torch.tensor(a.reshape(len(a), -1))
print(f"{args.host}: clients {cids}, sizes {[len(trains[i]) for i in cids]}",
      flush=True)

sock = socket.create_connection((args.server, args.port))
send_msg(sock, {"host": args.host, "clients": cids})

net = model()
while True:
    msg, _ = recv_msg(sock)
    if msg["round"] < 0:
        break
    t_recv = time.time()
    per_client_states, ns = [], []
    t_train0 = time.time()
    for i in cids:
        net.load_state_dict(msg["state"])
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        X = trains[i]
        perm = torch.randperm(len(X))
        for b in range(0, len(X), args.batch):
            xb = X[perm[b:b + args.batch]]
            opt.zero_grad()
            loss = weighted_loss(net(xb), xb)
            loss.backward()
            opt.step()
        per_client_states.append({k: v.detach().cpu()
                                  for k, v in net.state_dict().items()})
        ns.append(len(X))
    t_train = time.time() - t_train0
    total = sum(ns)
    agg = {k: sum(s[k].double() * (n / total)
                  for s, n in zip(per_client_states, ns)).to(torch.float32)
           for k in per_client_states[0]}
    send_msg(sock, {"host": args.host, "n": total, "state": agg,
                    "timing": {"train_s": round(t_train, 3),
                               "n_logical_clients": len(cids)}})
    print(f"{args.host} round {msg['round']}: train {t_train:.2f}s", flush=True)
print(f"{args.host} done", flush=True)
