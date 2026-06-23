# Cosmic Ray Sensor Network Notes

Small working repo for exploring the CREDO / CosmicWatch Elasticsearch data and shaping the edge-AI research direction.

## Setup

1. Copy `.env.example` to `.env`.
2. Fill in `CREDO_USER` and `CREDO_PASS`.
3. Keep `.env` local; it is intentionally ignored by git.

This local workspace also has a `.venv` created with:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

The `--system-site-packages` flag lets the venv reuse the already-installed local PyTorch/Numpy stack without downloading large packages. For a fully isolated environment, create a fresh venv without that flag and run `pip install -r requirements.txt`.

## Scripts

- `data_readiness.py` checks source timelines, active detector days, and gaps.
- `cosmicwatch_summary.py` summarizes coincident CosmicWatch events and can dump recent examples.
- `edge_ai_experiment.py` pulls the clean Jan. 23-24 CosmicWatch window, builds event features, trains a tiny MLP baseline, and trains a toy pure-PyTorch SNN.
- `multi_node_probe.py` verifies whether the ES index contains synchronized multi-node data.
- `gnn_simulation.py` trains a simulation-only graph neural network prototype because the current ES index does not contain synchronized multi-node CosmicWatch/CREDO data.

Example:

```bash
source .venv/bin/activate
python3 data_readiness.py
python3 cosmicwatch_summary.py --events 200
python3 edge_ai_experiment.py --max-events 50000 --plots-dir plots
python3 multi_node_probe.py
python3 gnn_simulation.py
```

Use `--max-events 0` on `edge_ai_experiment.py` to pull the full selected date window.

