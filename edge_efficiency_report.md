# Edge-Efficiency Study — CosmicWatch Event Classifier

Real Jan 23–24 window · 237,716 events · 12.25% coincident. Accuracy is near-ceiling on this single-node data, so the contribution is the **size/latency** trade-off, not the F1.

Reference floor — **ADC threshold**: F1 0.4067, AUC 0.787, ≈0 model bytes.

## MLP (57 params)

| precision | bytes | F1 | AUC | F1 retained vs 32-bit |
|---|--:|--:|--:|--:|
| 32-bit | 201 | 0.4069 | 0.7968 | 100% |
| 8-bit | 57 | 0.4071 | 0.7967 | 100% |
| 4-bit | 33 | 0.4055 | 0.7952 | 100% |
| 2-bit | 21 | 0.3359 | 0.6658 | 83% |
| 1-bit | 15 | 0.3207 | 0.7511 | 79% |

Latency: **7.2 µs/event** single-shot, **12,857,596 events/s** batched — 9,346,220× the 1.3757 Hz detector rate.
Compute: ~48 MACs/event

## SNN (73 params)

| precision | bytes | F1 | AUC | F1 retained vs 32-bit |
|---|--:|--:|--:|--:|
| 32-bit | 289 | 0.3812 | 0.7703 | 100% |
| 8-bit | 73 | 0.3808 | 0.7699 | 100% |
| 4-bit | 37 | 0.3641 | 0.7735 | 96% |
| 2-bit | 19 | 0.2146 | 0.5 | 56% |
| 1-bit | 10 | 0.2395 | 0.6257 | 63% |

Latency: **304.6 µs/event** single-shot, **1,629,123 events/s** batched — 1,184,214× the 1.3757 Hz detector rate.
Compute: ~1152 sparse synaptic ops/event over 16 timesteps

## Takeaways

- MLP compresses to **4-bit / 33 bytes** while keeping F1 0.4055 (≥95% of the 0.4069 float baseline) — a 6.1× size cut for free.
- Below that it breaks down (e.g. 2-bit F1 0.3359), so int8/int4 is the usable floor — aggressive sub-4-bit quantization is NOT free here.
- SNN holds accuracy to **4-bit / 37 bytes** (F1 0.3641); its edge case is footprint + sparse synaptic ops, not winning the F1.
- Both run far faster than needed: MLP 9,346,220× and SNN 1,184,214× the 1.3757 Hz event rate — latency is a non-issue; size/energy is the real axis.
- Accuracy is bounded by the weak intra-unit `coincident` label, not model capacity. A real MCU power/latency measurement is the next step that would make the edge story conference-grade.
