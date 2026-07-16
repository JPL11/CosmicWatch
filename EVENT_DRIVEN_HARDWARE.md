# Event-Driven Pi and Jetson Deployment

## What “event-driven” means here

`event_gateway.py` blocks in the Linux kernel on a UART, pipe, or file descriptor. No model loop runs while
the detector is quiet. One policy evaluation runs when a complete JSON event arrives, and selected events are
written in batches. UART reception normally wakes Linux through an interrupt, so this is a real
event-triggered software pipeline.

It is not neuromorphic execution. The Pi and Jetson remain powered while blocked, and their CPUs/GPUs execute
ordinary instructions when awakened. The existing rate-coded SNN expands each static event into artificial
steps and should not be described as a hardware event-driven energy result on either board.

## Event gateway

The input contract is one JSON object per line with `adc_value` and, when available, `sipm_mv`,
`temperature_c`, `pressure_pa`, and `coincident` or `coincidence_flag`.

```bash
# Blocking UART input. Install pyserial only for this mode.
python3 -m pip install pyserial
python3 event_gateway.py \
  --serial /dev/ttyACM0 --baud 115200 \
  --policy hybrid --adc-threshold 238 --mlp-threshold 0.390279 \
  --batch-size 64 --output selected_events.jsonl

# A producer can also pipe JSONL to blocking stdin.
detector-json-producer | python3 event_gateway.py --input - --policy hybrid
```

The current thresholds came from the chronological reduction experiment. Recalibrate them after detector
gain or threshold changes. If the detector emits its native non-JSON serial format, put a small parser in the
producer; do not treat malformed serial fields as zero-valued events.

Useful measurements are board idle watts while blocked, average watts during a realistic event stream,
selected bytes/day, batch/network overhead, CPU duty fraction, temperature, and dropped serial records.

## Prepare the FL workload

Run once on the development machine with the local export:

```bash
python3 prepare_fl_hardware_data.py --clients 8 --max-per-client 3000
```

This creates `legacy_fl_hardware.npz`, approximately 2.3 MB, containing chronological train/test arrays for
eight real legacy device IDs. Put the following on each target:

```text
fl_hardware_benchmark.py
legacy_fl_hardware.npz
```

The target needs NumPy and PyTorch. The benchmark times one real client's local training workload; it does
not pretend that cheap server-side FedAvg aggregation is the expensive part.

## Raspberry Pi 4

Use the official power supply; the previous Pi 4 result showed undervoltage throttling.

```bash
python3 fl_hardware_benchmark.py \
  --data legacy_fl_hardware.npz --client-index 0 \
  --rounds 6 --local-epochs 1 --threads 4 \
  --idle-watts 2.7 --load-watts 4.1 \
  --out fl_hardware_benchmark_raspberry_pi_4.json
```

Read idle and average load watts from an inline USB-C meter. Report both total joules and active joules above
idle. A Pi cannot enter deep sleep while acting as an always-listening UART gateway; a Pico or external power
controller is needed if deep sleep is a requirement.

## Jetson Orin Nano

Run both the constrained and maximum-power modes. Exact mode IDs vary by JetPack image, so inspect them first.

```bash
sudo nvpmodel -q
tegrastats --interval 500

python3 fl_hardware_benchmark.py \
  --data legacy_fl_hardware.npz --client-index 0 \
  --rounds 6 --local-epochs 1 --device cuda \
  --idle-watts 4.8 --load-watts 7.0 \
  --out fl_hardware_benchmark_jetson_7w.json
```

Use average `VDD_IN` readings from `tegrastats` for the wattage arguments, then repeat in MAXN mode. CUDA may
not win for this 26,584-parameter autoencoder because launch and transfer overhead are significant; that is a
result to measure, not assume.

## Decision criteria

- Prefer the Pi as the always-on gateway if it meets reliability and I/O requirements at lower board power.
- Prefer the Jetson only when image training, a realistic larger GNN, or another GPU-sized workload offsets
  its higher idle power.
- Compare FL update bytes with compressed raw images. The current six-round FedAvg experiment transfers 1.95x
  the raw grayscale bytes, so compute speed alone cannot make its communication case.
- Do not benchmark the synthetic GNN as a hardware claim until detector count, graph-window rate, and edge
  construction match a planned deployment.
