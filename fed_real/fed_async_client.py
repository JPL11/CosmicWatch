"""Async federation client host: loops (pull global, train assigned
logical clients, push sample-weighted update) at its own pace until the
server says stop. Same training step as the synchronous client.
Usage: python fed_async_client.py --server <ip> --host pi5 --clients 0,1,2
"""
import argparse
import socket
import time

import numpy as np
import torch

from fed_common import (model, weighted_loss, send_msg, recv_msg,
                        pack_state, unpack_state, Int8EF, TopKEF)

ap = argparse.ArgumentParser()
ap.add_argument("--server", required=True)
ap.add_argument("--port", type=int, default=29500)
ap.add_argument("--host", required=True)
ap.add_argument("--clients", required=True)
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
pulls = 0
comp = None
view = None
last_wire = None
while True:
    msg, nbytes = recv_msg(sock)
    if msg.get("stop"):
        break
    link = msg.get("link")
    if link:
        down_bps, up_bps, rtt = link
        time.sleep(nbytes * 8 / down_bps + rtt / 2)
    wire = msg.get("wire", "fp32")
    if wire != last_wire:
        # wire switched (adaptive server): fresh EF residuals
        if wire == "topk_ef":
            comp = TopKEF()
        elif wire in ("int8ef", "int8_delta_ef", "int8_delta2_ef"):
            comp = Int8EF()
        else:
            comp = None
        last_wire = wire
    if wire == "int8_delta2_ef" and msg.get("mode") == "delta":
        d = unpack_state(msg["state"], "int8")
        view = {k: view[k] + d[k] for k in d}
    else:
        view = unpack_state(msg["state"], wire)
    global_state = view
    per_client_states, ns = [], []
    t_train0 = time.time()
    for i in cids:
        net.load_state_dict(global_state)
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
    if wire in ("int8_delta_ef", "int8_delta2_ef", "topk_ef"):
        payload = comp.pack({k: agg[k] - global_state[k] for k in agg})
    elif wire == "int8ef":
        payload = comp.pack(agg)
    else:
        payload = pack_state(agg, wire)
    sent = send_msg(sock, {"host": args.host, "n": total,
                           "base_version": msg["version"],
                           "state": payload,
                           "timing": {"train_s": round(t_train, 3),
                                      "n_logical_clients": len(cids)}})
    if link:
        time.sleep(sent * 8 / up_bps + rtt / 2)
    pulls += 1
print(f"{args.host} done after {pulls} updates", flush=True)
