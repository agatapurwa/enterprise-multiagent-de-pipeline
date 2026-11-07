import os
import re
import logging
from typing import Dict, Any, Optional, Tuple, List
import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError, ParseError
from litellm import completion

logger = logging.getLogger("enterprise_de.agent.operator")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class AgenticOperator:
    """
    Agentic AI Operations Layer (LLMOps) for Autonomous Schema Healing.
    Generates non-destructive SQL DDL migrations based on sanitized metadata
    and enforces deterministic AST-level (sqlglot) guardrails against destructive queries.
    """

    # Valid AST Expression Types in sqlglot.expressions
    FORBIDDEN_AST_NODES = (
        exp.Drop,
        exp.Delete,
        exp.Insert,
        exp.Update,
    )

    # Dialect mapping from common names to sqlglot dialect identifiers
    SQLGLOT_DIALECT_MAP = {
        "bigquery": "bigquery",
        "postgres": "postgres",
        "postgresql": "postgres",
        "snowflake": "snowflake",
        "duckdb": "duckdb",
        "spark": "spark",
    }

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_key: Optional[str] = None
    ):
        """
        Args:
            model: LiteLLM-compatible model string (e.g., 'gpt-4o-mini', 'claude-3-5-sonnet', 'gemini/gemini-1.5-pro').
            temperature: Sampling temperature (0.0 for deterministic execution).
            api_key: Optional explicit API key. Defaults to LiteLLM provider environment detection.
        """
        self.model = model
        self.temperature = temperature
        self.api_key = api_key

    def _extract_sql_from_response(self, response_text: str) -> str:
        """Parses raw LLM string response to extract SQL code inside markdown blocks."""
        sql_block_match = re.search(
            r"```(?:sql)?\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE
        )
        if sql_block_match:
            return sql_block_match.group(1).strip()
        return response_text.strip()

    def _map_dialect(self, dialect: str) -> str:
        """Translates user target dialect string to sqlglot read key."""
        return self.SQLGLOT_DIALECT_MAP.get(dialect.lower(), "sql")

    def validate_ddl_safety(
        self, sql_query: str, target_dialect: str = "BigQuery"
    ) -> Tuple[bool, Optional[str]]:
        """
        AST-based Guardrail Evaluation using sqlglot.

        Returns:
            Tuple of (is_safe: bool, rejection_reason: Optional[str])
        """
        clean_query = sql_query.strip().rstrip(";")
        if not clean_query:
            return False, "Empty SQL query generated."

        sg_dialect = self._map_dialect(target_dialect)

        # Check 1: Multi-Statement SQL Injection Guard
        try:
            parsed_statements = sqlglot.parse(clean_query, read=sg_dialect)
            non_empty_statements = [stmt for stmt in parsed_statements if stmt is not None]

            if len(non_empty_statements) != 1:
                logger.error(
                    f"Guardrail Blocked: Query contains {len(non_empty_statements)} SQL statements. "
                    "Only single-statement DDL execution is permitted."
                )
                return (
                    False,
                    f"Multi-statement SQL injection risk: Found {len(non_empty_statements)} statements.",
                )
            expression = non_empty_statements[0]
        except (SqlglotError, ParseError) as e:
            logger.error(f"Guardrail AST Syntax Parsing Failure: {str(e)}")
            return False, f"SQL Syntax Error: Unable to parse query AST under dialect '{target_dialect}'."

        # Check 2: AST Traversal for Forbidden Nodes (DROP, DELETE, TRUNCATE, INSERT, UPDATE)
        for node in expression.walk():
            if isinstance(node, self.FORBIDDEN_AST_NODES):
                forbidden_type = node.__class__.__name__.upper()
                logger.error(f"Guardrail Blocked AST Node: Found forbidden operation '{forbidden_type}'.")
                return False, f"Destructive AST node detected: '{forbidden_type}' is forbidden."

            # Catch TRUNCATE statement nodes and Command expressions
            if node.key.lower() == "truncate" or (isinstance(node, exp.Command) and "TRUNCATE" in str(node).upper()):
                logger.error("Guardrail Blocked AST Node: Found TRUNCATE command.")
                return False, "Destructive AST node detected: 'TRUNCATE' is forbidden."

        # Check 3: Enforce Non-Destructive Structural Intent
        if isinstance(expression, exp.Alter):
            query_upper = clean_query.upper()
            if "ADD COLUMN" not in query_upper and "ADD" not in query_upper:
                return False, "ALTER TABLE statement must strictly be an additive operation (ADD COLUMN)."

        elif isinstance(expression, exp.Create):
            is_view = expression.args.get("kind") == "VIEW" or "VIEW" in clean_query.upper()
            exists_ok = expression.args.get("exists") is True
            if not (is_view or exists_ok):
                return False, "CREATE statement must be either CREATE VIEW or CREATE TABLE IF NOT EXISTS."
        else:
            statement_type = expression.key.upper()
            logger.error(f"Guardrail Blocked: Unsupported statement type '{statement_type}'.")
            return (
                False,
                f"Forbidden statement type '{statement_type}'. Query must be ALTER TABLE ADD COLUMN or CREATE VIEW.",
            )

        return True, None

    def construct_system_prompt(self, target_dialect: str) -> str:
        """Builds strict system context instructing the model to behave as a Data Engineer."""
        return (
            f"You are an expert Data Engineer specializing in {target_dialect} schema migrations.\n"
            "Your task is to propose non-destructive DDL queries to handle schema drift or missing columns.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. ONLY output valid non-destructive DDL (e.g., ALTER TABLE <table_name> ADD COLUMN <column_name> <TYPE>).\n"
            "2. NEVER output DROP, DELETE, TRUNCATE, UPDATE, INSERT, or column removal statements.\n"
            "3. Output ONLY the raw executable SQL inside a single Markdown ```sql ``` block.\n"
            f"4. Enforce precise native data types for target engine '{target_dialect}'."
        )

    def generate_ddl_proposal(
        self,
        sanitized_metadata: Dict[str, Any],
        target_table: str,
        target_dialect: str = "BigQuery"
    ) -> Dict[str, Any]:
        """
        Passes sanitized schema metadata to LiteLLM and evaluates safety of the resulting DDL query.
        """
        logger.info(f"AgenticOperator: Generating DDL proposal for table '{target_table}' ({target_dialect})")

        prompt_user_content = (
            f"Target Table: {target_table}\n"
            f"Target Dialect: {target_dialect}\n"
            f"Current Schema Metadata: {sanitized_metadata.get('schema', {})}\n"
            f"Sample Record Structure (Sanitized): {sanitized_metadata.get('sample_head', [])}\n\n"
            "Analyze the schema metadata above. If new columns are detected or missing from target specifications, "
            "generate an additive ALTER TABLE ADD COLUMN statement or CREATE OR REPLACE VIEW to accommodate the stream."
        )

        try:
            litellm_kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.construct_system_prompt(target_dialect)},
                    {"role": "user", "content": prompt_user_content},
                ],
                "temperature": self.temperature,
            }
            if self.api_key:
                litellm_kwargs["api_key"] = self.api_key

            response = completion(**litellm_kwargs)

            raw_reply = response.choices[0].message.content
            extracted_sql = self._extract_sql_from_response(raw_reply)

            # Enforce AST Guardrail Verification
            is_safe, rejection_reason = self.validate_ddl_safety(extracted_sql, target_dialect=target_dialect)

            if not is_safe:
                logger.warning(f"DDL Proposal Rejected by Guardrail: {rejection_reason}")
                return {
                    "status": "REJECTED_BY_GUARDRAIL",
                    "is_safe": False,
                    "target_table": target_table,
                    "proposed_sql": extracted_sql,
                    "error": rejection_reason,
                    "risk_level": "CRITICAL",
                }

            logger.info("DDL Proposal Generated & Passed AST Safety Verification.")
            return {
                "status": "PROPOSED",
                "is_safe": True,
                "target_table": target_table,
                "proposed_sql": extracted_sql,
                "risk_level": "NON_DESTRUCTIVE",
                "error": None,
            }

        except Exception as e:
            logger.error(f"AgenticOperator Execution Error: {str(e)}")
            return {
                "status": "ERROR",
                "is_safe": False,
                "target_table": target_table,
                "proposed_sql": None,
                "error": str(e),
                "risk_level": "UNKNOWN",
            }