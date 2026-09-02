# Real-hardware federated learning over the LAN

Unlike `fl_hardware_benchmark_*.json` (single-board simulation of all
clients), this is an actual synchronous FedAvg federation over TCP
sockets between three physical machines on the home LAN:

| host    | hardware                  | logical clients |
|---------|---------------------------|-----------------|
| desktop | Ryzen + RTX 5070 Ti (CPU torch) | 6, 7      |
| pi5     | Raspberry Pi 5 16 GB      | 0, 1, 2         |
| jetson  | Jetson Orin Nano (CPU)    | 3, 4, 5         |

Server runs on the desktop (port 29500). Same model and data as the
committed FL benchmark: the 26.6k-param autoencoder
(400-32-8-32-400, weighted MSE, targets weighted 1+4t) on per-device
20x20 image crops from `fl_device_crops.npz`, normalized to [0,1].
One local epoch per round, sample-weighted FedAvg, 6 rounds.

## Config A — heterogeneous fleet (fed_real_heterog.json)

- Global test weighted MSE: 0.226 -> 0.078 over 6 rounds
  (monotone decrease; matches the single-board benchmark regime).
- Round 1 (includes torch warm-up) exposes the straggler clearly:
  desktop arrives at 0.40 s, Pi 5 at 1.18 s, Jetson at 2.19 s, so the
  round costs 2.22 s and the fast host idles 1.8 s (81% of the round).
- Steady state (rounds 2-6): ~0.19 s/round wall. Training is
  ~0.15-0.16 s on the ARM boards vs ~0.01 s on the desktop; the
  straggler gap stays ~0.17 s, i.e. the desktop idles ~90% of every
  round waiting for the ARM boards.
- Communication is nearly free at this model size: 106 kB per update
  (float32 state dict), 3.9 MB total over the whole run; wire time is
  hidden inside the ~0.02 s arrival spread between Pi 5 and Jetson.
- Takeaway: with a 26.6k-param model the federation is
  compute-bound on the slowest board, not network-bound. Straggler
  mitigation (async aggregation or client weighting) would matter
  long before compression of the updates would.

## Config B — all-Pi network (fed_real_allpi.json)

"Pi as a network": server + clients 0-3 on the Pi 5, clients 4-7 on a
Raspberry Pi 4 Model B (4 GB, Python 3.11, torch 2.14 CPU). Same
model, data, and protocol as Config A; two hosts instead of three.

- Global test weighted MSE: 0.228 -> 0.119 over 6 rounds (same
  regime as Config A; the different endpoint reflects a different
  random init on the Pi 5 server plus 4-client hosts, not a
  protocol difference).
- Round 1 warm-up: Pi 5 arrives at 1.24 s, Pi 4 at 3.50 s.
- Steady state: ~0.34 s/round wall, with the Pi 4 training its 4
  clients in 0.27 s vs the Pi 5's 0.20 s, so the straggler gap is
  ~0.12 s (Pi 5 idles ~35% of each round). One round showed a
  transient Pi 4 blip to 0.60 s (scheduling/thermal jitter).
- Wire: 2.6 MB total for the run; still invisible next to compute.
- Link: both Pis ran on Wi-Fi (`wlan0`), no Ethernet — at 106 kB per
  update the radio is nowhere near the bottleneck. A wired rerun
  would only matter for (a) removing Wi-Fi jitter from
  publication-grade round timings (the 0.60 s blip is ambiguous
  between scheduling and retransmits), or (b) much larger models
  (MB-scale updates, e.g. a federated neural operator).

## Config A vs Config B

The all-Pi network is ~1.8x slower per round than the heterogeneous
fleet (0.34 vs 0.19 s) but far better balanced: the worst-case idle
fraction drops from ~90% (desktop waiting on ARM boards) to ~35%
(Pi 5 waiting on Pi 4). In a synchronous federation, adding a fast
host does not speed up the round at all — wall time is set by the
slowest host — it only buys idle time. A homogeneous cheap fleet is
the efficient shape for this workload; the fast machine is better
spent as server-plus-evaluator or moved to async aggregation.

## Files

- `fed_common.py` — model, weighted loss, length-prefixed socket
  send/recv helpers.
- `fed_server.py` — synchronous FedAvg server; records per-round wall
  time, per-host arrival times, client-reported train times, straggler
  gap, bytes moved, and global test weighted MSE.
- `fed_client.py` — one process per host, trains its logical clients
  sequentially each round.

Run: server first (`fed_server.py --hosts 3 --rounds 6 --out ...`),
then one `fed_client.py --server <ip> --host <name> --clients i,j,k`
per machine. Data file `fl_device_crops.npz` must be present beside
the scripts on every host.
