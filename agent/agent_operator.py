import logging
import os
import re
from typing import Dict, Any, Optional, Tuple
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
    and enforces strict regex-based security guardrails against destructive queries.
    """

    # Destructive SQL Keywords (Strictly Blocked)
    DESTRUCTIVE_PATTERNS = [
        r"(?i)\bDROP\b",
        r"(?i)\bDELETE\b",
        r"(?i)\bTRUNCATE\b",
        r"(?i)\bUPDATE\b",
        r"(?i)\bINSERT\b",
        r"(?i)\bALTER\s+TABLE\s+[\w\.\`\-]+\s+DROP\b",
        r"(?i)\bCASCADE\b",
    ]

    # Explicitly Allowed Non-Destructive Patterns
    ALLOWED_DDL_PATTERNS = [
        r"(?i)^\s*ALTER\s+TABLE\s+[\w\.\`\-]+\s+ADD\s+COLUMN\b",
        r"(?i)^\s*CREATE\s+(OR\s+REPLACE\s+)?VIEW\b",
        r"(?i)^\s*CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\b",
    ]

    def __init__(
        self, 
        model: str = "gpt-4o-mini", 
        temperature: float = 0.0,
        api_key: Optional[str] = None
    ):
        """
        Args:
            model: LiteLLM-compatible model string (e.g., 'gpt-4o-mini', 'claude-3-5-sonnet', 'gemini/gemini-1.5-pro').
            temperature: Sampling temperature (set to 0.0 for deterministic code generation).
            api_key: Optional explicit API key. Defaults to loading from environment.
        """
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LITELLM_API_KEY")

    def _extract_sql_from_response(self, response_text: str) -> str:
        """Parses raw LLM string response to extract SQL code inside markdown blocks."""
        sql_block_match = re.search(r"```(?:sql)?\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE)
        if sql_block_match:
            return sql_block_match.group(1).strip()
        return response_text.strip()

    def validate_ddl_safety(self, sql_query: str) -> Tuple[bool, Optional[str]]:
        """
        Deterministic Guardrail: Evaluates DDL proposals against destructive SQL patterns.

        Returns:
            Tuple of (is_safe: bool, rejection_reason: Optional[str])
        """
        clean_query = sql_query.strip()
        if not clean_query:
            return False, "Empty SQL query generated."

        # Check 1: Scan for destructive SQL operations
        for pattern in self.DESTRUCTIVE_PATTERNS:
            match = re.search(pattern, clean_query)
            if match:
                violating_keyword = match.group(0)
                logger.error(f"Guardrail Triggered: Destructive keyword detected '{violating_keyword}' in SQL.")
                return False, f"Destructive operation blocked: '{violating_keyword}' is forbidden."

        # Check 2: Enforce strict non-destructive structural patterns
        is_allowed = any(re.search(pattern, clean_query) for pattern in self.ALLOWED_DDL_PATTERNS)
        if not is_allowed:
            logger.error(f"Guardrail Triggered: Query does not conform to allowed non-destructive DDL templates.")
            return False, "Query rejected: Must be an additive ALTER TABLE, CREATE VIEW, or CREATE TABLE IF NOT EXISTS statement."

        return True, None

    def construct_system_prompt(self, target_dialect: str) -> str:
        """Builds strict system context instructing the model to behave as a Data Engineer."""
        return (
            f"You are an expert Data Engineer specializing in {target_dialect} schema migrations.\n"
            "Your task is to propose non-destructive DDL queries to handle schema drift or missing columns.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. ONLY output valid non-destructive DDL (e.g., ALTER TABLE ... ADD COLUMN ... or CREATE OR REPLACE VIEW).\n"
            "2. NEVER output DROP, DELETE, TRUNCATE, UPDATE, INSERT, or column removal statements.\n"
            "3. Output ONLY the raw executable SQL inside a single Markdown ```sql ``` block.\n"
            "4. Match target data types precisely to the warehouse specification."
        )

    def generate_ddl_proposal(
        self, 
        sanitized_metadata: Dict[str, Any], 
        target_table: str, 
        target_dialect: str = "BigQuery"
    ) -> Dict[str, Any]:
        """
        Passes sanitized schema metadata to LiteLLM and evaluates safety of the resulting DDL query.

        Args:
            sanitized_metadata: Output payload from PIISanitizer containing non-sensitive schema info.
            target_table: Full warehouse destination table path (e.g., `project.dataset.staging_table`).
            target_dialect: SQL engine dialect (default: 'BigQuery').

        Returns:
            Execution dictionary containing safety status, proposed SQL, and risk assessment.
        """
        logger.info(f"AgenticOperator: Generating DDL proposal for table '{target_table}' ({target_dialect})")

        prompt_user_content = (
            f"Target Table: {target_table}\n"
            f"Dialect: {target_dialect}\n"
            f"Current Schema Metadata: {sanitized_metadata.get('schema', {})}\n"
            f"Sample Record Structure (Sanitized): {sanitized_metadata.get('sample_head', [])}\n\n"
            "Analyze the schema metadata above. If new columns are detected or missing from target specifications, "
            "generate an additive ALTER TABLE ADD COLUMN statement or CREATE OR REPLACE VIEW to accommodate the stream."
        )

        try:
            # Invoke LiteLLM Completion API
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.construct_system_prompt(target_dialect)},
                    {"role": "user", "content": prompt_user_content}
                ],
                temperature=self.temperature,
                api_key=self.api_key
            )

            raw_reply = response.choices[0].message.content
            extracted_sql = self._extract_sql_from_response(raw_reply)

            # Enforce Deterministic Safety Guardrail
            is_safe, rejection_reason = self.validate_ddl_safety(extracted_sql)

            if not is_safe:
                logger.warning(f"DDL Proposal Rejected by Guardrail: {rejection_reason}")
                return {
                    "status": "REJECTED_BY_GUARDRAIL",
                    "is_safe": False,
                    "target_table": target_table,
                    "proposed_sql": extracted_sql,
                    "error": rejection_reason,
                    "risk_level": "CRITICAL"
                }

            logger.info("DDL Proposal Generated & Passed Safety Verification.")
            return {
                "status": "PROPOSED",
                "is_safe": True,
                "target_table": target_table,
                "proposed_sql": extracted_sql,
                "risk_level": "NON_DESTRUCTIVE",
                "error": None
            }

        except Exception as e:
            logger.error(f"AgenticOperator Execution Error: {str(e)}")
            return {
                "status": "ERROR",
                "is_safe": False,
                "target_table": target_table,
                "proposed_sql": None,
                "error": str(e),
                "risk_level": "UNKNOWN"
            }