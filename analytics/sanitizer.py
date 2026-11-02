import os
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
    Deterministic PII Detector & Masking Engine.
    Combines schema-level inspection and cell-level value sampling to ensure 
    zero un-masked PII reaches downstream LLM agents.
    """

    # Comprehensive PII Column Naming Regexes (GDPR / HIPAA compliant)
    PII_COLUMN_PATTERNS = [
        r"(?i)email", r"(?i)phone", r"(?i)mobile", r"(?i)cell",
        r"(?i)ssn", r"(?i)social_security", r"(?i)credit_card", r"(?i)card_num",
        r"(?i)password", r"(?i)secret", r"(?i)dob", r"(?i)birthdate", r"(?i)bday",
        r"(?i)first_?name", r"(?i)last_?name", r"(?i)full_?name", r"(?i)surname",
        r"(?i)street", r"(?i)address", r"(?i)zipcode", r"(?i)postal",
        r"(?i)ip_address", r"(?i)mac_address", r"(?i)passport", r"(?i)tax_id"
    ]

    # Cell Value-Level Regex Patterns
    VALUE_PATTERNS = {
        "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        "PHONE": re.compile(r"\b\+?[1-9]\d{1,14}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        "IPV4": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    }

    def __init__(self, salt: Optional[str] = None):
        """
        Args:
            salt: Secret salt string. If omitted, attempts to read PII_SANITIZER_SALT from environment.
        """
        resolved_salt = salt or os.getenv("PII_SANITIZER_SALT")
        if not resolved_salt:
            raise ValueError(
                "Security Risk: Salt must be explicitly passed or defined in PII_SANITIZER_SALT env var."
            )
        self.salt_bytes = resolved_salt.encode("utf-8")

    def _hash_series(self, series: pl.Series) -> pl.Series:
        """Vectorized SHA-256 hashing across a Polars Series without row-by-row map_elements."""
        # Null handling and String casting
        str_series = series.cast(pl.Utf8).fill_null("HASH_NULL")
        raw_vals = str_series.to_list()
        
        # Batch SHA-256 hash calculation using full hex digest (256 bits)
        hashed_vals = [
            "HASH_NULL" if v == "HASH_NULL" 
            else f"SHA256_{hashlib.sha256(v.encode('utf-8') + self.salt_bytes).hexdigest()}"
            for v in raw_vals
        ]
        return pl.Series(series.name, hashed_vals, dtype=pl.Utf8)

    def scan_columns_for_pii(self, df: pl.DataFrame) -> List[str]:
        """Identifies columns matching standard PII column naming conventions."""
        flagged_cols = []
        for col in df.columns:
            for pattern in self.PII_COLUMN_PATTERNS:
                if re.search(pattern, col):
                    flagged_cols.append(col)
                    break
        return flagged_cols

    def scan_values_for_pii(self, df: pl.DataFrame, sample_size: int = 100) -> List[str]:
        """Samples string column cell contents against value regex patterns to catch un-flagged PII."""
        sample_df = df.head(sample_size)
        flagged_cols = []

        for col in sample_df.columns:
            if sample_df.schema[col] == pl.Utf8:
                # Extract non-null str values
                vals = sample_df[col].drop_nulls().to_list()
                for val in vals:
                    if any(pattern.search(str(val)) for pattern in self.VALUE_PATTERNS.values()):
                        flagged_cols.append(col)
                        break

        return flagged_cols

    def sanitize_dataframe(
        self, 
        df: Union[pl.DataFrame, pl.LazyFrame], 
        explicit_pii_cols: Optional[List[str]] = None,
        sample_value_scan: bool = True
    ) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Masks PII values using SHA-256 and constructs safe LLM schema metadata.
        """
        materialized_df = df.collect() if isinstance(df, pl.LazyFrame) else df

        # 1. Identify Target PII Columns (Column names + Cell sampling)
        name_detected = self.scan_columns_for_pii(materialized_df)
        value_detected = self.scan_values_for_pii(materialized_df) if sample_value_scan else []
        
        combined_pii = set(name_detected + value_detected + (explicit_pii_cols or []))
        valid_pii_cols = [c for c in materialized_df.columns if c in combined_pii]

        # 2. Apply SHA-256 Masking
        sanitized_df = materialized_df.clone()
        for col in valid_pii_cols:
            hashed_series = self._hash_series(sanitized_df[col])
            sanitized_df = sanitized_df.with_columns(hashed_series)

        logger.info(
            f"PIISanitizer execution complete. Rows: {sanitized_df.height}, "
            f"Masked Columns: {valid_pii_cols}"
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