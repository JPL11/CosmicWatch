# CosmicWatch Analyst

Analysis and validation agent for distributed low-cost cosmic-ray muon
detector networks: CosmicWatch-class desktop counters (plastic
scintillator + SiPM + microcontroller) and CREDO-style federations of
many small devices.

## Overview

The agent encodes the working methodology of an open cosmic-ray
sensor-network research program
([JPL11/CosmicWatch](https://github.com/JPL11/CosmicWatch)): data-readiness
auditing, detector-health screening, single-node muon physics (Poisson
timing, Landau/Moyal MIP fits, coincidence efficiency, dead time),
atmospheric and space-weather correlation protocols with explicit
artifact guards, on-detector edge-AI triage and hardware benchmark
interpretation (Raspberry Pi / Jetson / microcontroller class), and
federated learning over real, strongly non-IID device partitions.

## Architecture

Prompt-only agent (`kind: prompt`): no bundled tools or containers. All
reference implementations live in the public repository above; the agent
guides analysis, writes dependency-light numpy-first code mirroring
those references, and enforces measurement-vs-claim hygiene.

## Prerequisites

- A chat model deployment (the `{{CHAT-MODEL}}` placeholder).
- User-supplied data: per-event CSV/JSONL exports with the fields
  described in the agent instructions, or benchmark result JSONs.

## Configuration

None beyond the model binding. No external services are called.

## Usage

Load the agent and supply detector exports or benchmark JSONs in the
conversation, or use the `cosmic-ray-sensor-network` starter kit's
sample prompts.

## Known limitations

- Domain scope is counting-detector networks; it is not a general
  particle-physics or PDE agent.
- Physics protocols assume single-counter or two-layer-coincidence
  devices at roughly 1-3 Hz; air-shower array reconstruction is out of
  scope.
- The agent will decline to substitute simulations for hardware
  measurements; this is by design.

## Contributing

Issues and improvements via
https://github.com/JPL11/CosmicWatch/issues.
