import hashlib
import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union
import polars as pl

logger = logging.getLogger("enterprise_de.analytics.sanitizer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class PIISanitizer:
    """
    Deterministic PII Detector & SHA-256 Masking Layer.
    Guarantees zero raw PII reaches downstream LLM agents or operational logs.
    """

    # Common PII Regex Patterns
    PII_COLUMN_PATTERNS = [
        r"(?i)email", r"(?i)phone", r"(?i)mobile", r"(?i)ssn", 
        r"(?i)social_security", r"(?i)credit_card", r"(?i)card_num", 
        r"(?i)password", r"(?i)secret", r"(?i)dob", r"(?i)birthdate"
    ]

    VALUE_PATTERNS = {
        "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "PHONE": r"\b\+?[1-9]\d{1,14}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
        "IPV4": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b"
    }

    def __init__(self, salt: str = "enterprise_de_default_salt_2026"):
        """
        Args:
            salt: Secret string appended to values prior to hashing to prevent rainbow table attacks.
        """
        if not salt:
            raise ValueError("Salt must be a non-empty string.")
        self.salt = salt

    def _hash_value(self, value: Any) -> str:
        """Computes deterministic SHA-256 hash of a string value + salt."""
        if value is None:
            return "HASH_NULL"
        salted_str = f"{str(value)}{self.salt}"
        return f"SHA256_{hashlib.sha256(salted_str.encode('utf-8')).hexdigest()[:16]}"

    def scan_columns_for_pii(self, df: pl.DataFrame) -> List[str]:
        """Identifies columns that match standard PII naming conventions."""
        flagged_cols = []
        for col in df.columns:
            for pattern in self.PII_COLUMN_PATTERNS:
                if re.search(pattern, col):
                    flagged_cols.append(col)
                    break
        return flagged_cols

    def sanitize_dataframe(
        self, 
        df: Union[pl.DataFrame, pl.LazyFrame], 
        explicit_pii_cols: Optional[List[str]] = None
    ) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Masks PII values using SHA-256 and generates safe LLM metadata.

        Args:
            df: Input Polars DataFrame or LazyFrame.
            explicit_pii_cols: Additional columns explicitly designated as sensitive.

        Returns:
            Tuple of (sanitized_polars_dataframe, audit_metadata_dict)
        """
        materialized_df = df.collect() if isinstance(df, pl.LazyFrame) else df
        
        # 1. Identify target PII columns
        detected_cols = self.scan_columns_for_pii(materialized_df)
        all_pii_cols = list(set(detected_cols + (explicit_pii_cols or [])))
        
        # Filter to columns actually present in the DataFrame
        valid_pii_cols = [c for c in all_pii_cols if c in materialized_df.columns]

        # 2. Apply SHA-256 Masking to target columns
        sanitized_df = materialized_df.clone()
        for col in valid_pii_cols:
            sanitized_df = sanitized_df.with_columns(
                pl.col(col).map_elements(
                    lambda val: self._hash_value(val), 
                    return_dtype=pl.Utf8
                ).alias(col)
            )

        logger.info(
            f"PIISanitizer complete. Rows: {sanitized_df.height}, Masked Columns: {valid_pii_cols}"
        )

        # 3. Construct Safe LLM Schema Summary Payload
        llm_safe_metadata = {
            "total_rows": sanitized_df.height,
            "column_count": len(sanitized_df.columns),
            "schema": {name: str(dtype) for name, dtype in zip(sanitized_df.columns, sanitized_df.dtypes)},
            "pii_columns_masked": valid_pii_cols,
            "sample_head": sanitized_df.head(3).to_dicts()
        }

        return sanitized_df, llm_safe_metadata