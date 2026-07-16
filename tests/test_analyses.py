import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from detector_health import robust_z
from edge_reduction import metrics
from event_gateway import normalized_features, parse_line, select_event
from legacy_common import iter_legacy
from legacy_timing import count_pairs


class AnalysisTests(unittest.TestCase):
    def test_legacy_iterator_removes_exact_duplicates(self):
        fields = ["source", "device_id", "timestamp", "frame_content", "location", "visible"]
        rows = [
            ["legacy", "a", "1000", "same", '{"lat":1,"lon":2}', "False"],
            ["legacy", "a", "1000", "same", '{"lat":1,"lon":2}', "False"],
            ["legacy", "b", "1001", "other", "", "False"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            with path.open("w", newline="") as fh:
                writer = csv.writer(fh); writer.writerow(fields); writer.writerows(rows)
            result = list(iter_legacy(path, include_image=False))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[-1]["duplicates_before"], 1)
        self.assertEqual(result[0]["location"], (1.0, 2.0))

    def test_cross_device_pair_windows(self):
        rows = [(1000, "a", None), (1005, "b", None), (1050, "c", None), (1051, "c", None)]
        counts, pairs, _ = count_pairs(rows)
        self.assertEqual(counts[10], 1)
        self.assertEqual(counts[100], 5)
        self.assertEqual(pairs[10], 1)

    def test_reduction_metrics(self):
        result = metrics([True, False, True, False], [True, True, False, False], 32, 1.0)
        self.assertEqual(result["coincident_recall"], 0.5)
        self.assertEqual(result["coincident_precision"], 0.5)
        self.assertEqual(result["data_reduction_x"], 2.0)

    def test_robust_z_flags_large_shift(self):
        values = robust_z([1, 1, 1.1, 0.9, 10])
        self.assertGreater(values[-1], 3)
        self.assertTrue(np.isfinite(values).all())

    def test_event_gateway_policies(self):
        weights = {
            "standardize_mean": [100, 10, 25, 101000],
            "standardize_std": [10, 2, 1, 100],
            "W1": [[0], [0], [0], [0]], "b1": [0], "W2": [[0]], "b2": [0],
        }
        event = parse_line(b'{"adc_value":250,"coincidence_flag":0}')
        self.assertTrue(select_event(event, "adc", weights, 238, 0.9)[0])
        self.assertFalse(select_event(event, "coincidence", weights, 238, 0.9)[0])
        values = normalized_features(event, weights)
        self.assertEqual(values[0], 15.0)
        self.assertEqual(values[1:], [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
