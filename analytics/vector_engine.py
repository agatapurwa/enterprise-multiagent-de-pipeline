import logging
import numpy as np
import polars as pl
from typing import Dict, Any, Union, List, Optional

# Configure structured enterprise logging
logger = logging.getLogger("enterprise_de.analytics.vector_engine")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class VectorEngine:
    """
    Enterprise Vector Engine for pre-ingestion schema drift & data quality anomaly detection.
    Optimized for high-throughput batch processing with strict memory boundaries and magnitude checks.
    """

    def __init__(
        self, 
        similarity_threshold: float = 0.8, 
        magnitude_tolerance: float = 0.5, 
        sample_size: int = 5000
    ):
        """
        Initializes the Vector Engine.

        Args:
            similarity_threshold (float): Cosine similarity floor below which directional drift is flagged.
            magnitude_tolerance (float): Allowed percentage deviation from centroid norm (e.g., 0.5 = ±50%).
            sample_size (int): Maximum row limit to sample for streaming evaluation (default: 5,000).
        """
        self.similarity_threshold = similarity_threshold
        self.magnitude_tolerance = magnitude_tolerance
        self.sample_size = sample_size

    def calculate_cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Computes cosine similarity between two vectors from scratch using NumPy."""
        if v1.size == 0 or v2.size == 0:
            logger.error("VectorEngine: Encountered empty vector(s) during similarity calculation.")
            raise ValueError("Input vectors cannot be empty.")

        if v1.shape != v2.shape:
            logger.error(f"VectorEngine: Dimension mismatch. v1 shape: {v1.shape}, v2 shape: {v2.shape}")
            raise ValueError(f"Vector dimension mismatch: {v1.shape} vs {v2.shape}")

        v1_clean = np.nan_to_num(v1, nan=0.0, posinf=0.0, neginf=0.0)
        v2_clean = np.nan_to_num(v2, nan=0.0, posinf=0.0, neginf=0.0)

        dot_product = np.dot(v1_clean, v2_clean)
        norm_v1 = np.linalg.norm(v1_clean)
        norm_v2 = np.linalg.norm(v2_clean)

        if norm_v1 == 0.0 or norm_v2 == 0.0:
            return 0.0

        similarity = dot_product / (norm_v1 * norm_v2)
        return float(np.clip(similarity, -1.0, 1.0))

    def detect_anomalies(
        self, 
        sample_df: Union[pl.DataFrame, pl.LazyFrame], 
        reference_centroid: np.ndarray,
        expected_features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates incoming data batch against a reference centroid with feature order enforcement,
        magnitude-aware checks, and true Polars streaming.
        """
        try:
            # Memory Efficiency & True Streaming: Request streaming collection
            if isinstance(sample_df, pl.LazyFrame):
                eval_df = sample_df.head(self.sample_size).collect(streaming=True)
            else:
                eval_df = sample_df.head(self.sample_size)

            row_count = eval_df.height
            if row_count == 0:
                logger.warning("VectorEngine: Received empty DataFrame batch for anomaly detection.")
                return {
                    "status": "WARNING",
                    "message": "Empty batch received",
                    "total_evaluated": 0,
                    "anomaly_count": 0,
                    "anomaly_indices": []
                }

            # Extract numeric columns dynamically
            numeric_cols = [
                name for name, dtype in zip(eval_df.columns, eval_df.dtypes)
                if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64)
            ]

            if not numeric_cols:
                logger.error("VectorEngine: No numerical features found in dataframe columns.")
                raise ValueError("DataFrame contains no numerical features for vector comparison.")

            # Guardrail: Enforce exact feature order match if expected feature names are provided
            if expected_features is not None:
                if numeric_cols != expected_features:
                    logger.error("VectorEngine: Feature schema order mismatch detected!")
                    raise ValueError(
                        f"Schema drift / feature order mismatch. "
                        f"Expected order: {expected_features}, Received order: {numeric_cols}"
                    )

            # Extract raw array and track rows with non-finite values (NaN / Inf)
            raw_matrix = eval_df.select(numeric_cols).to_numpy()
            has_non_finite = ~np.isfinite(raw_matrix).all(axis=1)

            # Sanitize matrix for math computations
            data_matrix = np.nan_to_num(raw_matrix, nan=0.0, posinf=0.0, neginf=0.0)

            # Guardrail: Check feature dimension parity
            if data_matrix.shape[1] != reference_centroid.shape[0]:
                logger.error(
                    f"VectorEngine: Feature dimension mismatch. "
                    f"Data features ({data_matrix.shape[1]}) != Centroid features ({reference_centroid.shape[0]})."
                )
                raise ValueError(
                    f"Feature dimension mismatch: dataframe has {data_matrix.shape[1]} features, "
                    f"but reference centroid expects {reference_centroid.shape[0]}."
                )

            # Vectorized Matrix-to-Centroid Calculations
            dot_products = np.dot(data_matrix, reference_centroid)
            matrix_norms = np.linalg.norm(data_matrix, axis=1)
            centroid_norm = np.linalg.norm(reference_centroid)

            denominators = matrix_norms * centroid_norm
            denominators[denominators == 0.0] = 1e-10

            similarities = dot_products / denominators
            similarities = np.clip(similarities, -1.0, 1.0)

            # Guardrail: Magnitude Check (Catches rows scaled far from baseline magnitude despite good direction)
            centroid_norm_safe = centroid_norm if centroid_norm != 0.0 else 1e-10
            norm_ratios = np.abs(matrix_norms - centroid_norm_safe) / centroid_norm_safe
            magnitude_anomaly_mask = norm_ratios > self.magnitude_tolerance

            # Combine anomaly criteria: Directional drop OR Magnitude spike OR Data Corruption (NaN/Inf)
            similarity_anomaly_mask = similarities < self.similarity_threshold
            final_anomaly_mask = similarity_anomaly_mask | magnitude_anomaly_mask | has_non_finite
            
            anomaly_indices = np.where(final_anomaly_mask)[0].tolist()

            logger.info(
                f"VectorEngine evaluation complete. Rows evaluated: {row_count}, "
                f"Anomalies detected: {len(anomaly_indices)}"
            )

            return {
                "status": "SUCCESS",
                "total_evaluated": row_count,
                "features_analyzed": len(numeric_cols),
                "anomaly_count": len(anomaly_indices),
                "anomaly_indices": anomaly_indices,
                "similarity_scores": similarities.tolist(),
                "norm_deviations": norm_ratios.tolist(),
                "threshold_applied": self.similarity_threshold
            }

        except Exception as e:
            logger.error(f"VectorEngine execution failed: {str(e)}")
            raise