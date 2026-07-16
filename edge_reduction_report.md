# Edge Data-Reduction Policies

| policy | transmit | reduction | coincident recall | KiB/day |
|---|--:|--:|--:|--:|
| transmit_all | 1.000 | 1.0x | 1.000 | 3714.4 |
| hardware_coincidence_only | 0.084 | 11.9x | 1.000 | 312.1 |
| adc_target_recall_50 | 0.198 | 5.0x | 0.676 | 736.8 |
| mlp_target_recall_50 | 0.244 | 4.1x | 0.783 | 905.0 |
| adc_target_recall_75 | 0.326 | 3.1x | 0.888 | 1210.0 |
| mlp_target_recall_75 | 0.382 | 2.6x | 0.907 | 1420.2 |
| adc_target_recall_90 | 0.474 | 2.1x | 0.934 | 1760.0 |
| mlp_target_recall_90 | 0.525 | 1.9x | 0.946 | 1950.1 |
| adc_target_recall_95 | 0.628 | 1.6x | 0.963 | 2333.5 |
| mlp_target_recall_95 | 0.698 | 1.4x | 0.972 | 2592.3 |
| adc_target_recall_99 | 0.922 | 1.1x | 0.994 | 3423.8 |
| mlp_target_recall_99 | 0.999 | 1.0x | 1.000 | 3709.5 |

These are payload simulations; radio and board power require hardware measurement.
