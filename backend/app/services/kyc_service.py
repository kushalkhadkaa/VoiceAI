"""KYC natural-language to SQL query service (MySQL kyc_db.customer_kyc_v)."""
from __future__ import annotations

import datetime
import logging
import re
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)

# Single consolidated MySQL database for the banking app.
MYSQL_DB = "kyc_db"

KYC_SQL_SYSTEM_PROMPT = """You are a MySQL 8.0+ SQL generator for table `customer_kyc_v` in database `kyc_db`.
Strict Execution Guidelines:
1. Output ONLY the raw SQL string. Do not wrap queries in markdown formatting like ```sql.
2. Only generate SELECT queries. If a user command implies data modification (DROP, UPDATE, DELETE, INSERT, ALTER, TRUNCATE), output exactly: ERROR: Only read-only SELECT queries are permitted.
3. Account for common banking terminology in Nepal:
   - "Lakh" or "Lakhs" -> multiply by 100000. Examples: 1 Lakh = 100000, 5 Lakhs = 500000, 50 Lakhs = 5000000, 80 Lakhs = 8000000.
   - "Cr" or "Crore" -> multiply by 10000000. Examples: 1 Crore = 10000000, 2.5 Crore = 25000000, 4 Crore = 40000000.
   - "Overdue" or "Defaulter" -> overdue_amount > 0 OR days_past_due > 0.
   - "Low Risk / High Risk" -> risk_rating or high_risk_customer_flag columns.
4. Use LIKE '%...%' when matching text such as occupations, names, districts, employers.
5. Always LIMIT results to 50 unless the user asks for a count/aggregate.
Schema columns: customer_id, account_number, customer_type, customer_name, first_name, middle_name, last_name, nepali_name, gender, date_of_birth, age, marital_status, nationality, resident_status, citizenship_number, pan_number, mobile_number, email_address, permanent_province, permanent_district, permanent_municipality, permanent_address, current_province, current_district, current_municipality, current_address, employment_status, occupation, employer_name, designation, monthly_salary, business_income, rental_income, remittance_income, total_monthly_income, total_monthly_expense, net_disposable_income, source_of_funds, customer_since, account_type, average_balance, digital_banking_status, mobile_banking_status, kyc_status, risk_rating, pep_flag, aml_flag, high_risk_customer_flag, customer_risk_score, loan_customer_flag, loan_account_number, loan_type, loan_amount, approved_amount, loan_tenure_months, interest_rate, emi_amount, outstanding_balance, overdue_amount, days_past_due, collateral_type, collateral_value, loan_purpose_code, gross_monthly_income, net_monthly_income, existing_emi, proposed_emi, total_emi_obligation, debt_service_ratio, debt_to_income_ratio, affordability_score, credit_bureau_score, repayment_capacity"""

_FORBIDDEN_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE|INTO OUTFILE|LOAD_FILE)\b",
    re.IGNORECASE,
)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # remove opening fence (with optional language tag) and closing fence
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


class KYCService:
    def __init__(self, settings) -> None:
        self.settings = settings

    def _connect(self):
        import mysql.connector

        return mysql.connector.connect(
            host="localhost",
            port=3306,
            user="root",
            password="",
            database=MYSQL_DB,
            connection_timeout=5,
        )

    def status(self) -> dict:
        try:
            conn = self._connect()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM customer_kyc")
                (count,) = cursor.fetchone()
                cursor.close()
            finally:
                conn.close()
            return {"ok": True, "connected": True, "row_count": int(count)}
        except Exception as exc:
            return {"ok": False, "connected": False, "row_count": 0, "error": str(exc)}

    def generate_sql(self, question: str, llm) -> str:
        result = llm.chat(
            question,
            system_prompt=KYC_SQL_SYSTEM_PROMPT,
            options={"temperature": 0.0, "max_tokens": 400},
        )
        return _strip_fences(result.text)

    def validate_sql(self, sql: str) -> str | None:
        candidate = sql.strip()
        if not candidate:
            return "Empty SQL generated."
        if candidate.upper().startswith("ERROR:"):
            return candidate
        if not candidate.upper().startswith("SELECT"):
            return "Only SELECT queries are permitted."
        if _FORBIDDEN_RE.search(candidate):
            return "Query contains a forbidden statement; only read-only SELECT queries are permitted."
        # multiple statements: semicolon followed by non-whitespace
        if re.search(r";\s*\S", candidate):
            return "Multiple SQL statements are not permitted."
        return None

    def execute_query(self, sql: str) -> dict:
        conn = self._connect()
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(sql)
            raw_rows = cursor.fetchall()
            cursor.close()
        finally:
            conn.close()
        rows = [
            {key: _json_safe(value) for key, value in row.items()}
            for row in raw_rows[:100]
        ]
        return {"ok": True, "sql": sql, "rows": rows, "row_count": len(rows)}

    def ask(self, question: str, llm) -> dict:
        sql = self.generate_sql(question, llm)
        error = self.validate_sql(sql)
        if error:
            return {"ok": False, "sql": sql, "error": error}
        try:
            return self.execute_query(sql)
        except Exception as exc:
            logger.warning("KYC SQL execution failed, retrying once: %s", exc)
            retry_question = (
                f"{question}\n\nThe previous SQL failed with: {exc}. "
                "Fix the query. Same rules apply."
            )
            try:
                sql2 = self.generate_sql(retry_question, llm)
                error2 = self.validate_sql(sql2)
                if error2:
                    return {"ok": False, "sql": sql2, "error": error2}
                return self.execute_query(sql2)
            except Exception as exc2:
                return {"ok": False, "sql": sql, "error": str(exc2)}
