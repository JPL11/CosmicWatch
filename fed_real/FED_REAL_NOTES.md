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

## Config C — asynchronous aggregation (fed_async_*.json)

FedAsync-style server (`fed_async_server.py`): no rounds, no barriers.
Each host loops (pull global, train, push) at its own pace; the server
mixes every incoming update immediately with a staleness-discounted
weight alpha = 0.6 / (1 + staleness)^0.5 and evaluates after every mix,
giving an (elapsed, MSE) trajectory. 12 s wall budget per run.

- Heterogeneous fleet: 988 global versions in 12 s. The desktop went
  from ~90% idle under sync to contributing 865 of 988 updates; the
  ARM boards contributed 58 (Pi 5) and 65 (Jetson). At the sync run's
  total wall time (~3.2 s) async had already reached MSE 0.032 vs
  sync's 0.078 — 2.4x lower at equal wall clock — and async passed
  sync's final MSE within the first ~1 s. Staleness reached 145 for
  ARM updates (median 0); the alpha discount absorbed it with no
  divergence.
- All-Pi network: 71 versions in 12 s (Pi 5: 47, Pi 4: 24), MSE 0.064
  at 6 s vs sync's 0.119 at ~5.6 s — still ~2x, but the gain is much
  smaller than on the heterogeneous fleet, as expected: async's win
  scales with the idle fraction it reclaims (~90% vs ~35%).
- The cost is bandwidth: 215 MB on the wire in 12 s for the
  heterogeneous run (vs 3.9 MB for all of sync) because the fast host
  converts its former idle time into pull/push cycles. On a LAN (and
  mostly loopback for the desktop client) this is free; over a real
  WAN it would not be. Async is therefore exactly the regime where
  update compression — pointless under sync at this model size —
  becomes relevant.

## Config D — update compression under async (fed_async_heterog_{fp16,int8}.json)

Wire formats for BOTH directions (broadcast and update): fp32 baseline,
fp16, and per-tensor symmetric int8 (`pack_state`/`unpack_state` in
fed_common.py, `--wire` flag). Heterogeneous fleet, same 12 s budget:

| wire | versions | MB on wire | MSE@3.2s | MSE@12s | best MSE |
|------|---------:|-----------:|---------:|--------:|---------:|
| fp32 |      988 |      215.4 |    0.032 |  0.0159 |   0.0148 |
| fp16 |     1060 |      114.0 |    0.033 |  0.0149 |   0.0140 |
| int8 |     1142 |       62.3 |    0.032 |  0.0196 |   0.0186 |

- **fp16 is free**: 47% fewer bytes, ~7% more versions (smaller
  payloads also serialize/copy faster, so the loop speeds up even on
  a LAN), and final quality slightly better than fp32.
- **int8 hits a noise floor**: 71% fewer bytes and the most versions,
  identical early trajectory (all three arms ~0.032 at 3.2 s), but
  convergence plateaus ~25% above fp32 (0.0196 vs 0.0159) — the
  per-mix quantization noise on a 26.6k-param model sets a floor the
  extra iterations cannot buy back.
- Reading: compression is harmless while gradients are large and
  binds only near convergence — the same "safe until the precision
  floor engages" shape as the PTQ results in the quantization work.
  At this scale fp16 is strictly the right wire format; int8 would
  only win where bytes are the actual constraint (WAN/cellular).
- **int8 + error feedback** (`int8ef`: each endpoint carries its
  quantization residual into the next pack, 1-bit-SGD / EF-SGD
  style): tail-window (9-12 s) mean MSE 0.0191 vs int8's 0.0207 and
  fp32's 0.0172 — EF recovers roughly HALF the int8 floor at the
  same byte cost (56.6 MB). Partial rather than full recovery makes
  sense in this design: we quantize full model *states*, so each
  sender's residual corrects its own sequence but the classic EF
  guarantee (for compressed *deltas*) does not directly apply, and
  the staleness-weighted mixing attenuates the correction. Full
  recovery would need delta-transmission (send state minus base,
  quantized, with EF) — done next as `int8_delta_ef`.
- **int8 delta + EF** (`int8_delta_ef`): the client sends the
  quantized *change* (trained minus the base it received) with error
  feedback; the server reconstructs base-plus-delta from the exact
  dequantized state it sent that connection. Deltas are ~100x smaller
  than weights, so the int8 scale is correspondingly finer. Result:
  tail MSE 0.0181 — ~75% of the int8-vs-fp32 gap recovered at the
  same 71% byte cut, with best-seen MSE (0.0159) inside the fp32
  run-to-run range. The full compression ladder (tail-window mean
  MSE, 9-12 s, heterogeneous fleet, 12 s budget):

  | wire                | MB    | tail MSE | vs fp32 gap |
  |---------------------|------:|---------:|------------:|
  | fp32                | 215.4 |  0.0172  |     —       |
  | fp16                | 114.0 |  0.0157  | better      |
  | int8                |  62.3 |  0.0207  | +0.0035     |
  | int8 + EF           |  56.6 |  0.0191  | +0.0019     |
  | int8 delta+EF (up)  |  62.1 |  0.0181  | +0.0009     |
  | int8 delta+EF (both)|  58.2 |  0.0171  |  +0.0000    |

- **Bidirectional delta + EF closes the gap completely**
  (`int8_delta2_ef`): the server also delta-encodes the broadcast —
  it tracks each connection's reconstructed view (full-state int8
  bootstrap on the first pull, then quantized global-minus-view with
  per-connection EF), and both endpoints advance the view with the
  same dequantized tensors, so they never drift. Uplink deltas then
  base off that shared view. Result: tail MSE 0.0171 — statistically
  identical to fp32 — at 58.2 MB (73% byte cut). The quantization
  floor is now fully attributed AND fully recovered: half was the
  uplink (fixed by delta+EF), the other half the downlink (fixed the
  same way). At int8 the wire format is no longer the thing that
  limits model quality; only bytes vs. bandwidth remains as a
  deployment trade.

## Strengthening campaign (camp_*.json: 27 runs, seeds + long + links)

Paper-grade rerun of the ladder: 6 wires x 3 seeds x 12 s; 120 s long
runs for fp32/int8/delta2; and emulated cellular links (application-
layer shaping of the ARM hosts, `LINK_PROFILES` in fed_common.py:
LTE 10/5 Mbps 70 ms, LTE-M 1 Mbps/375 kbps 150 ms, NB-IoT 60/30 kbps
1 s RTT). Three headline corrections/confirmations:

1. **Seeds collapse the fine-grained ladder.** At 12 s, fp32, fp16,
   int8ef, delta-EF, and delta2-EF are statistically
   indistinguishable (tail 0.0183-0.0192, seed std ~0.0008-0.0019).
   Plain int8 is the ONLY degraded wire: 0.0231 +/- 0.0016, ~4 sigma
   above the pack. The single-run readings that fp16 "beat" fp32 and
   that the EF variants formed a strict hierarchy were run-to-run
   noise. Honest claim: naive int8 costs quality; ANY error-feedback
   variant recovers it; everything else is a bytes decision.
2. **The long runs make the floor decisive — and worse than it
   looked.** Over 120 s (≈11k versions), int8's tail MSE degrades to
   0.0400 vs fp32's 0.0140 (its best point, 0.0173, comes early and
   it drifts UP as gradients shrink below the quantization noise).
   delta2-EF tracks fp32 the whole way (0.0145 tail, 0.0138 min).
   Naive int8 is not just a floor, it is late-run instability;
   EF-delta compression is quality-equivalent to fp32 at 1/4 bytes.
3. **Constrained links turn compression into a participation lever.**
   With ARM hosts on emulated cellular, global MSE stays similar
   (the unshaped desktop keeps learning, and this task's partitions
   are homogeneous enough to transfer), but ARM participation
   changes decisively: delta2 vs fp32 gets 390 vs 238 ARM updates on
   LTE, 107 vs 36 on LTE-M, and 22 vs 8 on NB-IoT — 1.6-3x more, at
   ~4x fewer bytes (e.g. 576 MB vs 2.2 GB on the NB-IoT run). In a
   non-IID deployment, that participation ratio IS the fairness /
   coverage story: under narrow links, wire compression decides
   whether cellular-attached sensors are in the federation at all.

## Config E — adaptive wire selection (camp2_*.json)

Server-side policy (`--adapt`): per connection, EWMAs of communication
time vs training time walk the client along fp32 -> fp16 ->
int8_delta2_ef, escalating when comm > train and relaxing when
comm < 0.25 x train. No client-side changes and no prior link
knowledge. Tested on mixed links (desktop unshaped, Pi 5 LTE-M,
Jetson NB-IoT, 90 s) and a degradation scenario (Pi 5 LTE -> NB-IoT
at t=45 s).

- **Per-host convergence is correct and fast on responsive links**:
  the desktop stayed at fp32 for all 7,532 of its updates (its link
  is free; compressing it would buy nothing), while the Pi 5 walked
  fp32 -> fp16 -> delta2 within ~9 s on LTE-M (~3 s on LTE) and then
  sat at delta2 — reaching 76 updates vs 27 under static fp32,
  within noise of the best static wire (81). The policy discovers
  per-host what the campaign needed a grid of static runs to find.
- **Degradation tracking**: when the Pi 5's link dropped LTE ->
  NB-IoT mid-run, the adaptive connection was already at delta2 and
  stayed there (6 post-switch updates vs 3 for static fp32; 147 vs
  90 pre-switch).
- **Honest caveats**: (a) global MSE differences between arms are
  within seed noise — the policy's claim is participation and
  constrained-link bytes, not accuracy; (b) TOTAL bytes under adapt
  ~= fp32 because the unshaped desktop dominates traffic by design —
  the right metric is per-constrained-host bytes (~4x lower);
  (c) on NB-IoT the EWMA converges slowly (escalations at 17 s and
  54 s) because each observation costs a 30-40 s exchange — seeding
  escalation from the first measured exchange would fix this.

## Files

- `fed_common.py` — model, weighted loss, length-prefixed socket
  send/recv helpers.
- `fed_server.py` — synchronous FedAvg server; records per-round wall
  time, per-host arrival times, client-reported train times, straggler
  gap, bytes moved, and global test weighted MSE.
- `fed_client.py` — one process per host, trains its logical clients
  sequentially each round.
- `fed_async_server.py` / `fed_async_client.py` — the asynchronous
  variant (Config C): per-host server threads, staleness-discounted
  immediate mixing, fixed wall-time budget.

Run: server first (`fed_server.py --hosts 3 --rounds 6 --out ...`),
then one `fed_client.py --server <ip> --host <name> --clients i,j,k`
per machine. Data file `fl_device_crops.npz` must be present beside
the scripts on every host.
