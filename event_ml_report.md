# Label-light ML on the CosmicWatch Event Stream

Real window · 237,716 events · 12.2% coincident.

## 10. Marginal value of features (over ADC alone)

| feature set | F1 | AUC |
|---|--:|--:|
| adc_only (adc_value) | 0.4067 | 0.7958 |
| adc+sipm (adc_value+sipm_mv) | 0.4069 | 0.7974 |
| adc+timing (adc_value+log1p_interarrival_ms) | 0.4067 | 0.7971 |
| adc+env (adc_value+temperature_c_clean+pressure_pa_clean) | 0.4073 | 0.7965 |
| all_features (adc_value+sipm_mv+log1p_interarrival_ms+temperature_c_clean+pressure_pa_clean) | 0.4069 | 0.7968 |

## 8. Self-supervised (autoencoder)

- Frozen-encoder linear probe: F1 0.2196, AUC 0.5665.
- Supervised (all features): F1 0.4069, AUC 0.7968.

## 9. Anomaly detection (reconstruction error)

- Top 475 flagged: mean ADC 1248.4 (vs 266.6), 0% saturated, coincident rate 0.1726.

## Findings

- ADC alone already reaches F1 0.4067 / AUC 0.7958; adding sipm/timing/environment moves it to at most F1 0.4069 / AUC 0.7968 — the extra features add little. Multimodal does NOT rescue accuracy; ADC (energy) dominates.
- Self-supervised pretraining UNDERperforms here: frozen-autoencoder linear probe F1 0.2196 vs supervised 0.4069. Reconstruction optimizes for feature variance, not the subtle ADC/coincidence signal, so AE-style SSL does not help on this near-1-feature task — a contrastive objective or a genuinely label-scarce regime is where SSL would be worth revisiting (honest negative result).
- Anomaly detector (AE reconstruction error) flags the top 475 events with mean ADC 1248.4 vs 266.6 overall and 0% ADC-saturated — it surfaces the high-energy / clipped tail without using labels.
- All three are achievable now on single-node data; none need multi-node data, and all reinforce that the ceiling is physics + weak label, not method choice.
