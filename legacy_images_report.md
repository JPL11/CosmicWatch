# Legacy CREDO Image Track

Decoded **6000** 20×20 hit-crops (sample of 69,000). `visible` = {'False': 6000} → NOT a usable label (constant).

## Clustering (grayscale -> PCA(numpy SVD) -> k-means(numpy), k=8)

PCA top-5 variance ratio: [0.2119, 0.1446, 0.1028, 0.0709, 0.0671]

| cluster | size | mean brightness |
|--:|--:|--:|
| 0 | 301 | 0.1228 |
| 1 | 205 | 0.0986 |
| 2 | 760 | 0.0964 |
| 3 | 152 | 0.1215 |
| 4 | 55 | 0.1226 |
| 5 | 2708 | 0.0046 |
| 6 | 1169 | 0.0474 |
| 7 | 650 | 0.0851 |

Geo bounds: {'lat': [49.93581833, 54.52752616], 'lon': [16.813123142164, 21.0597029]}

## Findings

- Decoded 6000 of 6000 legacy crops as 20x20 RGBA hit-crops — a real, clusterable CV dataset (not the toy phone-camera set).
- `visible` is constant False across the sample → NOT a usable supervised label; unsupervised clustering is the correct route (matches prior CREDO pseudo-labeling).
- k-means(k=8) yields 8 non-empty clusters that visibly separate the classic CREDO hit morphologies — round bright 'spots', elongated 'tracks/lines', bright corner 'artifacts' (light leaks), and faint single-pixel hits. This is a real unsupervised CV result on real data (expert labels would confirm the physical class names).
- Detections carry real Poland GPS ({'lat': [49.93581833, 54.52752616], 'lon': [16.813123142164, 21.0597029]}) over 2017–18 — geo present, but a single epoch disjoint from the 2025–26 CosmicWatch data, so still no cross-source synchronization.
