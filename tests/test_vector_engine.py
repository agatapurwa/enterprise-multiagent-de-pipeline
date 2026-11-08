import unittest
import numpy as np
import polars as pl
from analytics.vector_engine import VectorEngine


class TestVectorEngine(unittest.TestCase):
    """
    Unit test suite validating VectorEngine anomaly detection, magnitude tolerance,
    schema enforcement, and non-finite value isolation.
    """

    def setUp(self):
        self.engine = VectorEngine(similarity_threshold=0.8, magnitude_tolerance=0.5)
        self.reference_centroid = np.array([10.0, 20.0, 30.0])
        self.expected_features = ["feature_a", "feature_b", "feature_c"]

    def test_detect_anomalies_in_batch(self):
        test_data = pl.DataFrame({
            "feature_a": [10.0, 9.8, 10.2, 50.0, float("nan")],
            "feature_b": [20.0, 19.5, 20.5, 100.0, 20.0],
            "feature_c": [30.0, 30.2, 29.8, 150.0, 30.0],
            "category": ["A", "A", "B", "C", "D"],  # Non-numeric column filter test
        })

        report = self.engine.detect_anomalies(
            sample_df=test_data,
            reference_centroid=self.reference_centroid,
            expected_features=self.expected_features,
        )

        self.assertEqual(report["status"], "SUCCESS")
        self.assertEqual(report["total_evaluated"], 5)
        self.assertEqual(report["features_analyzed"], 3)
        self.assertEqual(report["non_finite_count"], 1)

        # Index 3 (scaled far from magnitude norm) & Index 4 (NaN) must be flagged
        self.assertIn(3, report["anomaly_indices"])
        self.assertIn(4, report["anomaly_indices"])
        # Normal rows (0, 1, 2) must not be flagged
        self.assertNotIn(0, report["anomaly_indices"])

    def test_schema_mismatch_raises_value_error(self):
        # Missing expected column 'feature_c'
        invalid_data = pl.DataFrame({
            "feature_a": [10.0],
            "feature_b": [20.0],
        })

        with self.assertRaises(ValueError):
            self.engine.detect_anomalies(
                sample_df=invalid_data,
                reference_centroid=self.reference_centroid,
                expected_features=self.expected_features,
            )


if __name__ == "__main__":
    unittest.main()