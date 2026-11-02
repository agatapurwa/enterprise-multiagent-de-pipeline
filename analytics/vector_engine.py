import logging
import numpy as np
import polars as pl
from typing import Dict, Any, Union, List, Optional

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
    Enterprise Vector Engine for pre-ingestion schema drift & data anomaly detection.
    Enforces standardized feature scaling, strict schema matching, and non-destructive evaluation.
    """

    SUPPORTED_NUMERIC_TYPES = (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64, pl.Decimal
    )

    def __init__(
        self, 
        similarity_threshold: float = 0.8, 
        magnitude_tolerance: float = 0.5, 
        sample_size: int = 5000
    ):
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0.")
        if magnitude_tolerance < 0.0:
            raise ValueError("magnitude_tolerance must be non-negative.")
        if sample_size <= 0:
            raise ValueError("sample_size must be greater than 0.")

        self.similarity_threshold = similarity_threshold
        self.magnitude_tolerance = magnitude_tolerance
        self.sample_size = sample_size

    def detect_anomalies(
        self, 
        sample_df: Union[pl.DataFrame, pl.LazyFrame], 
        reference_centroid: np.ndarray,
        expected_features: List[str],
        centroid_means: Optional[np.ndarray] = None,
        centroid_stds: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluates incoming batch vectors against a standardized reference centroid.

        Args:
            sample_df: Input Polars DataFrame or LazyFrame.
            reference_centroid: 1D baseline vector centroid.
            expected_features: Explicit list of feature names matching reference_centroid ordering.
            centroid_means: Optional mean vector for Z-score feature standardization.
            centroid_stds: Optional standard deviation vector for Z-score standardization.
        """
        try:
            # 1. Validate Centroid Inputs
            centroid = np.asarray(reference_centroid, dtype=np.float64)
            if centroid.ndim != 1 or centroid.size == 0:
                raise ValueError("reference_centroid must be a non-empty 1D array.")
            if not np.isfinite(centroid).all():
                raise ValueError("reference_centroid contains non-finite values (NaN/Inf).")
            
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm == 0.0:
                raise ValueError("reference_centroid cannot be a zero vector.")

            if len(expected_features) != centroid.shape[0]:
                raise ValueError(
                    f"Dimension mismatch: expected_features count ({len(expected_features)}) "
                    f"does not match centroid size ({centroid.shape[0]})."
                )

            # 2. Materialize Bounded Sample
            if isinstance(sample_df, pl.LazyFrame):
                eval_df = sample_df.head(self.sample_size).collect()
            else:
                eval_df = sample_df.head(self.sample_size)

            row_count = eval_df.height
            if row_count == 0:
                logger.warning("VectorEngine: Received empty batch.")
                return {
                    "status": "WARNING",
                    "message": "Empty batch received",
                    "total_evaluated": 0,
                    "anomaly_count": 0,
                    "anomaly_indices": []
                }

            # 3. Schema Verification & Exact Ordering Alignment
            missing_cols = [col for col in expected_features if col not in eval_df.columns]
            if missing_cols:
                raise ValueError(f"Schema drift: Missing expected numeric columns: {missing_cols}")

            # Verify target dtypes
            for col in expected_features:
                dtype = eval_df.schema[col]
                if dtype not in self.SUPPORTED_NUMERIC_TYPES:
                    raise TypeError(f"Column '{col}' has non-numeric dtype '{dtype}'")

            # Reorder columns explicitly to match expected_features alignment
            ordered_df = eval_df.select(expected_features)
            raw_matrix = ordered_df.to_numpy().astype(np.float64)

            # 4. Separate Non-Finite Rows Prior to Matrix Operations
            has_non_finite = ~np.isfinite(raw_matrix).all(axis=1)
            valid_mask = ~has_non_finite

            # Pre-allocate output arrays
            similarities = np.full(row_count, -1.0, dtype=np.float64)
            norm_ratios = np.full(row_count, np.inf, dtype=np.float64)

            if np.any(valid_mask):
                valid_matrix = raw_matrix[valid_mask]

                # 5. Apply Z-Score Feature Standardization (if provided)
                if centroid_means is not None and centroid_stds is not None:
                    means = np.asarray(centroid_means, dtype=np.float64)
                    stds = np.asarray(centroid_stds, dtype=np.float64)
                    stds[stds == 0.0] = 1e-10  # Prevent zero-variance division
                    valid_matrix = (valid_matrix - means) / stds

                # 6. Vectorized Linear Algebra Evaluations
                dot_products = np.dot(valid_matrix, centroid)
                matrix_norms = np.linalg.norm(valid_matrix, axis=1)

                denominators = matrix_norms * centroid_norm
                zero_norm_mask = denominators == 0.0
                denominators[zero_norm_mask] = 1e-10

                calc_sims = dot_products / denominators
                calc_sims[zero_norm_mask] = 0.0
                similarities[valid_mask] = np.clip(calc_sims, -1.0, 1.0)

                # Magnitude ratio deviation calculation
                norm_ratios[valid_mask] = np.abs(matrix_norms - centroid_norm) / centroid_norm

            # 7. Evaluate Flag Thresholds
            similarity_anomalies = similarities < self.similarity_threshold
            magnitude_anomalies = norm_ratios > self.magnitude_tolerance
            final_anomaly_mask = similarity_anomalies | magnitude_anomalies | has_non_finite

            anomaly_indices = np.where(final_anomaly_mask)[0].tolist()

            logger.info(
                f"VectorEngine complete. Evaluated: {row_count}, Anomalies: {len(anomaly_indices)}"
            )

            return {
                "status": "SUCCESS",
                "total_evaluated": row_count,
                "features_analyzed": len(expected_features),
                "anomaly_count": len(anomaly_indices),
                "anomaly_indices": anomaly_indices,
                "similarity_scores": similarities.tolist(),
                "norm_deviations": norm_ratios.tolist(),
                "non_finite_count": int(np.sum(has_non_finite)),
                "threshold_applied": self.similarity_threshold
            }

        except Exception as e:
            logger.error(f"VectorEngine execution failed: {str(e)}")
            raise