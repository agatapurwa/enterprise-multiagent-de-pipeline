import unittest
from agent.agent_operator import AgenticOperator


class TestAgenticOperatorGuardrails(unittest.TestCase):
    """
    Unit test suite validating AgenticOperator AST guardrails against destructive DDL
    and multi-statement SQL injection attacks.
    """

    def setUp(self):
        self.operator = AgenticOperator()

    # --- 1. SAFE ADDITIVE MIGRATIONS ---

    def test_safe_alter_table_add_column_bigquery(self):
        query = "ALTER TABLE analytics.events ADD COLUMN user_score FLOAT64"
        is_safe, reason = self.operator.validate_ddl_safety(query, target_dialect="BigQuery")
        self.assertTrue(is_safe)
        self.assertIsNone(reason)

    def test_safe_alter_table_add_column_postgres(self):
        query = "ALTER TABLE public.orders ADD COLUMN is_discounted BOOLEAN"
        is_safe, reason = self.operator.validate_ddl_safety(query, target_dialect="Postgres")
        self.assertTrue(is_safe)
        self.assertIsNone(reason)

    def test_safe_create_or_replace_view(self):
        query = "CREATE OR REPLACE VIEW analytics.daily_summary AS SELECT 1 AS id"
        is_safe, reason = self.operator.validate_ddl_safety(query, target_dialect="BigQuery")
        self.assertTrue(is_safe)
        self.assertIsNone(reason)

    def test_safe_create_table_if_not_exists(self):
        query = "CREATE TABLE IF NOT EXISTS analytics.staging_events (id INT64, payload STRING)"
        is_safe, reason = self.operator.validate_ddl_safety(query, target_dialect="BigQuery")
        self.assertTrue(is_safe)
        self.assertIsNone(reason)

    # --- 2. FALSE-POSITIVE NAME SAFETY (IDENTIFIERS CONTAINING RESTRICTED WORDS) ---

    def test_false_positive_column_and_table_names(self):
        """Verifies that columns or tables containing 'drop' or 'updated' do not trigger false positives."""
        query_1 = "ALTER TABLE drop_logs ADD COLUMN last_updated TIMESTAMP"
        query_2 = "ALTER TABLE analytics.events ADD COLUMN drop_count INT64"

        for query in [query_1, query_2]:
            is_safe, reason = self.operator.validate_ddl_safety(query, target_dialect="BigQuery")
            self.assertTrue(is_safe, f"Failed false-positive check for query: {query}")
            self.assertIsNone(reason)

    # --- 3. BLOCKED DESTRUCTIVE DDL NODES ---

    def test_blocked_drop_table(self):
        query = "DROP TABLE analytics.events"
        is_safe, reason = self.operator.validate_ddl_safety(query, target_dialect="BigQuery")
        self.assertFalse(is_safe)
        self.assertIn("DROP", reason)

    def test_blocked_alter_table_drop_column(self):
        query = "ALTER TABLE analytics.events DROP COLUMN user_email"
        is_safe, reason = self.operator.validate_ddl_safety(query, target_dialect="BigQuery")
        self.assertFalse(is_safe)
        self.assertIn("DROP", reason)

    def test_blocked_truncate_table(self):
        query = "TRUNCATE TABLE analytics.events"
        is_safe, reason = self.operator.validate_ddl_safety(query, target_dialect="BigQuery")
        self.assertFalse(is_safe)
        self.assertIn("TRUNCATE", reason)

    def test_blocked_delete_from_table(self):
        query = "DELETE FROM analytics.events WHERE status = 'EXPIRED'"
        is_safe, reason = self.operator.validate_ddl_safety(query, target_dialect="BigQuery")
        self.assertFalse(is_safe)
        self.assertIn("DELETE", reason)

    # --- 4. MULTI-STATEMENT INJECTION ATTACKS ---

    def test_blocked_multi_statement_injection(self):
        query = "ALTER TABLE events ADD COLUMN new_col INT64; DROP TABLE events;"
        is_safe, reason = self.operator.validate_ddl_safety(query, target_dialect="BigQuery")
        self.assertFalse(is_safe)
        self.assertIn("Multi-statement SQL injection risk", reason)


if __name__ == "__main__":
    unittest.main()