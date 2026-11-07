import os
import polars as pl
from analytics.sanitizer import PIISanitizer

def main():
    print("Starting PII Sanitizer test batch...")
    
    # Ensure salt environment variable is available
    salt = os.getenv("PII_SANITIZER_SALT")
    if not salt:
        raise ValueError("PII_SANITIZER_SALT environment variable is missing!")

    raw_stream = pl.DataFrame({
        "customer_id": [9001, 9002],
        "email": ["alice@enterprise.com", "bob@enterprise.com"],
        "support_ticket": ["Issue reported by dev@enterprise.com", "No PII here"],
        "metric_value": [42.0, 108.5]
    })

    sanitizer = PIISanitizer(salt=salt)
    clean_df, metadata = sanitizer.sanitize_dataframe(
        df=raw_stream,
        explicit_pii_cols=["customer_id"],
        sample_value_scan=True
    )

    print("\n✅ KESTRA SANITIZATION SUCCESS:")
    print(f"Masked Columns: {metadata['pii_columns_masked']}")
    print(f"Sample Head Payload:\n{metadata['sample_head']}")

if __name__ == "__main__":
    main()