import numpy as np
import polars as pl
from analytics.vector_engine import VectorEngine

# 1. Initialize Engine
engine = VectorEngine(similarity_threshold=0.8, magnitude_tolerance=0.5)

# 2. Define a baseline reference centroid (e.g., expected average metrics for 3 features)
reference_centroid = np.array([10.0, 20.0, 30.0])

# 3. Create a test batch DataFrame with normal rows, a directional drift row, and a corrupted row
test_data = pl.DataFrame({
    "feature_a": [10.0, 9.8, 10.2, 50.0, float('nan')],
    "feature_b": [20.0, 19.5, 20.5, 100.0, 20.0],
    "feature_c": [30.0, 30.2, 29.8, 150.0, 30.0],
    "category": ["A", "A", "B", "C", "D"] # Non-numeric column to test dynamic filtering
})

# 4. Run anomaly detection
report = engine.detect_anomalies(test_data, reference_centroid, expected_features=["feature_a", "feature_b", "feature_c"])

print("=== ANOMALY DETECTION REPORT ===")
for key, value in report.items():
    print(f"{key}: {value}")