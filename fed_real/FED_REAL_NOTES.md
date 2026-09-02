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

## Config B — all-Pi network (pending)

Planned rerun with Raspberry Pi boards only (Pi 5 + Pi 4 + Pi 400)
once the Pi 4 is powered: a homogeneous "Pi as a network" topology to
compare straggler gap and round wall time against Config A.

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
