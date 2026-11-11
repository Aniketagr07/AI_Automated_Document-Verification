"""
Database module for Automated Document Verification System (SIH1709).
Provides SQLite-based persistent storage for:
- User accounts and role-based access control (USER vs ADMIN)
- Documents linked to user_id for strict user data privacy isolation
- Extracted structured fields and OCR evidence
- Production rule triggers and ML evaluations
- Selective human-in-the-loop admin review queue and decisions
- Formal verification reports with unique references
- Comprehensive audit trails and GDPR hard-delete controls
"""

import sqlite3
import json
import os
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from auth_manager import hash_password, verify_password, sanitize_role

DB_FILE = Path(__file__).parent / "verification_system.db"


def get_db_connection() -> sqlite3.Connection:
    """Creates a connection with row factory enabled."""
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    """Initializes and migrates the database schema."""
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.cursor()

            # 1. Users table for role-based authentication
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'USER',
                    created_at TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    reset_token TEXT,
                    reset_token_expiry TEXT
                )
            """)

            # 2. Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER DEFAULT 1,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    declared_category TEXT NOT NULL,
                    predicted_category TEXT,
                    upload_timestamp TEXT NOT NULL,
                    ocr_confidence REAL DEFAULT 0.0,
                    page_count INTEGER DEFAULT 1,
                    raw_text TEXT,
                    verification_status TEXT NOT NULL,
                    requires_admin_review INTEGER DEFAULT 0,
                    escalation_reason TEXT,
                    admin_review_status TEXT DEFAULT 'NONE',
                    admin_notes TEXT,
                    reviewed_by TEXT,
                    reviewed_at TEXT,
                    is_deleted INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
                )
            """)

            # Check and perform non-destructive schema migrations for documents
            cursor.execute("PRAGMA table_info(documents)")
            existing_doc_cols = [row[1] for row in cursor.fetchall()]

            new_cols = [
                ("user_id", "INTEGER DEFAULT 1"),
                ("requires_admin_review", "INTEGER DEFAULT 0"),
                ("escalation_reason", "TEXT"),
                ("admin_review_status", "TEXT DEFAULT 'NONE'"),
                ("admin_notes", "TEXT"),
                ("reviewed_by", "TEXT"),
                ("reviewed_at", "TEXT")
            ]
            for col_name, col_def in new_cols:
                if col_name not in existing_doc_cols:
                    cursor.execute(f"ALTER TABLE documents ADD COLUMN {col_name} {col_def}")

            # Migration for user reset_token_expiry column
            cursor.execute("PRAGMA table_info(users)")
            existing_user_cols = [row[1] for row in cursor.fetchall()]
            if "reset_token_expiry" not in existing_user_cols:
                cursor.execute("ALTER TABLE users ADD COLUMN reset_token_expiry TEXT")

            # Migration for audit_history
            cursor.execute("PRAGMA table_info(audit_history)")
            existing_audit_cols = [row[1] for row in cursor.fetchall()]
            if existing_audit_cols and "user_id" not in existing_audit_cols:
                cursor.execute("ALTER TABLE audit_history ADD COLUMN user_id INTEGER")

            # Migration for human_reviews
            cursor.execute("PRAGMA table_info(human_reviews)")
            existing_hr_cols = [row[1] for row in cursor.fetchall()]
            if existing_hr_cols and "reviewer_id" not in existing_hr_cols:
                cursor.execute("ALTER TABLE human_reviews ADD COLUMN reviewer_id INTEGER")

            # 3. Extracted fields table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS extracted_fields (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    field_name TEXT NOT NULL,
                    extracted_value TEXT,
                    confidence REAL DEFAULT 0.0,
                    source_page INTEGER DEFAULT 1,
                    evidence_text TEXT,
                    is_manually_corrected INTEGER DEFAULT 0,
                    corrected_value TEXT,
                    manual_verification_required INTEGER DEFAULT 0,
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
                )
            """)

            # 4. Rule evaluation results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rule_triggers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    rule_id TEXT NOT NULL,
                    rule_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    evidence TEXT,
                    recommended_action TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
                )
            """)

            # 5. ML Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    predicted_category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    probabilities_json TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
                )
            """)

            # 6. Human Review decisions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS human_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    reviewer_id INTEGER,
                    reviewer_name TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    review_notes TEXT,
                    reviewed_at TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
                )
            """)

            # 7. Audit History table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER,
                    user_id INTEGER,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    details TEXT
                )
            """)

            # 8. Reports table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    report_reference TEXT UNIQUE NOT NULL,
                    report_type TEXT NOT NULL DEFAULT 'USER_REPORT',
                    generated_at TEXT NOT NULL,
                    final_status TEXT NOT NULL,
                    summary_json TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            """)

            # 9. Seed initial admin from environment variables ONLY (no hardcoded passwords)
            cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'ADMIN'")
            admin_count = cursor.fetchone()[0]
            if admin_count == 0:
                admin_email = os.environ.get("DOCVERIFY_ADMIN_EMAIL", "").strip()
                admin_pwd = os.environ.get("DOCVERIFY_ADMIN_PASSWORD", "").strip()

                if admin_email and admin_pwd:
                    # Provision from env vars
                    pwd_hash, salt = hash_password(admin_pwd)
                    now = datetime.now().isoformat()
                    cursor.execute("""
                        INSERT INTO users (email, password_hash, salt, full_name, role, created_at, is_active)
                        VALUES (?, ?, ?, ?, 'ADMIN', ?, 1)
                    """, (admin_email, pwd_hash, salt, "Lead Verification Officer", now))

                    cursor.execute("""
                        INSERT INTO audit_history (action, actor, timestamp, details)
                        VALUES ('INITIAL_ADMIN_PROVISIONED', 'SYSTEM_ENV', ?, 'Administrator account provisioned via environment variables.')
                    """, (now,))
                else:
                    # Log a warning — admin must be created via bootstrap_admin.py
                    import warnings
                    warnings.warn(
                        "\n[DocVerify] No ADMIN account exists. "
                        "Set DOCVERIFY_ADMIN_EMAIL and DOCVERIFY_ADMIN_PASSWORD environment variables "
                        "OR run: python bootstrap_admin.py\n",
                        UserWarning,
                        stacklevel=2
                    )

            # 10. Seed standard demo user if none exists
            cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'USER'")
            user_count = cursor.fetchone()[0]
            if user_count == 0:
                demo_email = "applicant@college.edu"
                demo_pwd = "User@DocVerify2024!"
                pwd_hash, salt = hash_password(demo_pwd)
                now = datetime.now().isoformat()
                cursor.execute("""
                    INSERT INTO users (email, password_hash, salt, full_name, role, created_at, is_active)
                    VALUES (?, ?, ?, ?, 'USER', ?, 1)
                """, (demo_email, pwd_hash, salt, "Aarav Sharma", now))

    finally:
        conn.close()


# ===========================================================================
# USER MANAGEMENT & AUTHENTICATION
# ===========================================================================

def create_user(
    email: str,
    password: str,
    full_name: str,
    role: str = "USER",
    is_admin_creator: bool = False
) -> Tuple[bool, str, Optional[int]]:
    """
    Registers a new user. Strictly prevents self-assignment of ADMIN role
    unless explicitly created by an existing administrator.
    """
    init_database()
    email_clean = email.strip().lower()
    name_clean = full_name.strip()

    safe_role = sanitize_role(role, is_admin_creator=is_admin_creator)
    pwd_hash, salt = hash_password(password)
    now = datetime.now().isoformat()

    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE email = ?", (email_clean,))
            if cursor.fetchone():
                return False, f"An account with email '{email_clean}' already exists.", None

            cursor.execute("""
                INSERT INTO users (email, password_hash, salt, full_name, role, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, 1)
            """, (email_clean, pwd_hash, salt, name_clean, safe_role, now))
            new_id = cursor.lastrowid

            cursor.execute("""
                INSERT INTO audit_history (user_id, action, actor, timestamp, details)
                VALUES (?, 'USER_REGISTERED', ?, ?, ?)
            """, (new_id, email_clean, now, f"User '{name_clean}' registered with role '{safe_role}'."))

            return True, f"Registration successful for '{email_clean}'.", new_id
    except Exception as e:
        return False, f"Database error during registration: {str(e)}", None
    finally:
        conn.close()


def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Validates user credentials against stored hash.
    Returns safe user profile dictionary or None.
    """
    init_database()
    email_clean = email.strip().lower()

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ? AND is_active = 1", (email_clean,))
        user_row = cursor.fetchone()
        if not user_row:
            return None

        if verify_password(password, user_row["password_hash"], user_row["salt"]):
            now = datetime.now().isoformat()
            cursor.execute("""
                INSERT INTO audit_history (user_id, action, actor, timestamp, details)
                VALUES (?, 'USER_LOGIN_SUCCESS', ?, ?, 'Successful login authentication.')
            """, (user_row["id"], email_clean, now))
            conn.commit()

            return {
                "id": user_row["id"],
                "email": user_row["email"],
                "full_name": user_row["full_name"],
                "role": user_row["role"],
                "created_at": user_row["created_at"]
            }
        else:
            now = datetime.now().isoformat()
            cursor.execute("""
                INSERT INTO audit_history (user_id, action, actor, timestamp, details)
                VALUES (?, 'USER_LOGIN_FAILED', ?, ?, 'Failed login attempt: incorrect password.')
            """, (user_row["id"], email_clean, now))
            conn.commit()
            return None
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Fetches user profile by ID."""
    init_database()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, email, full_name, role, created_at, is_active FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Fetches user profile by email."""
    init_database()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, email, full_name, role, created_at, is_active FROM users WHERE email = ?", (email.strip().lower(),))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_user_password(user_id: int, new_password: str) -> bool:
    """Updates user password securely."""
    init_database()
    pwd_hash, salt = hash_password(new_password)
    now = datetime.now().isoformat()
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users SET password_hash = ?, salt = ? WHERE id = ?
            """, (pwd_hash, salt, user_id))

            cursor.execute("""
                INSERT INTO audit_history (user_id, action, actor, timestamp, details)
                VALUES (?, 'PASSWORD_UPDATED', 'USER', ?, 'User password successfully changed.')
            """, (user_id, now))
            return True
    finally:
        conn.close()



def create_password_reset_token(email: str) -> Optional[str]:
    """
    Generates a secure time-limited password reset token for the given email.
    Token is valid for 1 hour. Returns the token string or None if email not found.
    """
    init_database()
    email_clean = email.strip().lower()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = ? AND is_active = 1", (email_clean,))
        user_row = cursor.fetchone()
        if not user_row:
            return None

        token = secrets.token_urlsafe(32)
        expiry = (datetime.now() + timedelta(hours=1)).isoformat()

        with conn:
            cursor.execute(
                "UPDATE users SET reset_token = ?, reset_token_expiry = ? WHERE id = ?",
                (token, expiry, user_row["id"])
            )
            cursor.execute(
                "INSERT INTO audit_history (user_id, action, actor, timestamp, details) VALUES (?, ?, ?, ?, ?)",
                (user_row["id"], "PASSWORD_RESET_TOKEN_CREATED", "SYSTEM",
                 datetime.now().isoformat(), "Password reset token generated.")
            )

        return token
    finally:
        conn.close()


def validate_and_consume_reset_token(token: str, new_password: str) -> Tuple[bool, str]:
    """
    Validates a reset token (non-expired, matching) then resets the password.
    Returns (success: bool, message: str).
    """
    init_database()
    if not token or not token.strip():
        return False, "Invalid token provided."

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, reset_token_expiry FROM users WHERE reset_token = ? AND is_active = 1",
            (token.strip(),)
        )
        row = cursor.fetchone()

        if not row:
            return False, "Invalid or already-used reset token."

        expiry_str = row["reset_token_expiry"]
        if not expiry_str:
            return False, "Token has no expiry. Please request a new reset token."

        try:
            expiry_dt = datetime.fromisoformat(expiry_str)
        except ValueError:
            return False, "Corrupted token expiry. Please request a new reset token."

        if datetime.now() > expiry_dt:
            with conn:
                cursor.execute(
                    "UPDATE users SET reset_token = NULL, reset_token_expiry = NULL WHERE id = ?",
                    (row["id"],)
                )
            return False, "This reset token has expired (valid for 1 hour). Please request a new one."

        pwd_hash, salt = hash_password(new_password)
        now = datetime.now().isoformat()
        with conn:
            cursor.execute(
                "UPDATE users SET password_hash = ?, salt = ?, reset_token = NULL, reset_token_expiry = NULL WHERE id = ?",
                (pwd_hash, salt, row["id"])
            )
            cursor.execute(
                "INSERT INTO audit_history (user_id, action, actor, timestamp, details) VALUES (?, ?, ?, ?, ?)",
                (row["id"], "PASSWORD_RESET_COMPLETED", "USER", now,
                 "Password successfully reset via token.")
            )

        return True, "Password has been successfully reset. You can now sign in with your new credentials."
    finally:
        conn.close()


def get_all_users() -> List[Dict[str, Any]]:
    """Fetches all registered users for Admin User Management."""
    init_database()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT u.id, u.email, u.full_name, u.role, u.created_at, u.is_active,
                   COUNT(d.id) as document_count
            FROM users u
            LEFT JOIN documents d ON u.id = d.user_id AND d.is_deleted = 0
            GROUP BY u.id
            ORDER BY u.id ASC
        """)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def set_user_active_status(user_id: int, is_active: bool, actor: str = "ADMIN") -> bool:
    """Enables or disables a user account."""
    init_database()
    now = datetime.now().isoformat()
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET is_active = ? WHERE id = ?", (1 if is_active else 0, user_id))
            cursor.execute("""
                INSERT INTO audit_history (user_id, action, actor, timestamp, details)
                VALUES (?, 'USER_STATUS_TOGGLED', ?, ?, ?)
            """, (user_id, actor, now, f"User account active status changed to {is_active}."))
            return True
    finally:
        conn.close()


# ===========================================================================
# DOCUMENT RECORD PERSISTENCE & USER ISOLATION
# ===========================================================================

def save_verification_record(
    filename: str,
    file_type: str,
    file_size: int,
    declared_category: str,
    predicted_category: str,
    ocr_confidence: float,
    page_count: int,
    raw_text: str,
    verification_status: str,
    extracted_fields: Dict[str, Dict[str, Any]],
    triggered_rules: List[Dict[str, Any]],
    ml_result: Dict[str, Any],
    user_id: int = 1,
    requires_admin_review: bool = False,
    escalation_reason: Optional[str] = None
) -> int:
    """
    Saves complete document verification output into database linked to the user_id.
    """
    init_database()
    timestamp = datetime.now().isoformat()
    conn = get_db_connection()

    try:
        with conn:
            cursor = conn.cursor()

            admin_status = "PENDING" if requires_admin_review else "NONE"

            # 1. Insert document record with user_id and escalation metadata
            cursor.execute("""
                INSERT INTO documents (
                    user_id, filename, file_type, file_size, declared_category,
                    predicted_category, upload_timestamp, ocr_confidence,
                    page_count, raw_text, verification_status,
                    requires_admin_review, escalation_reason, admin_review_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, filename, file_type, file_size, declared_category,
                predicted_category, timestamp, ocr_confidence,
                page_count, raw_text, verification_status,
                1 if requires_admin_review else 0,
                escalation_reason,
                admin_status
            ))
            document_id = cursor.lastrowid

            # 2. Insert extracted fields
            for field_name, meta in extracted_fields.items():
                val = meta.get("value") if isinstance(meta, dict) else meta
                conf = meta.get("confidence", 0.0) if isinstance(meta, dict) else 0.0
                page_n = meta.get("page", 1) if isinstance(meta, dict) else 1
                ev = meta.get("evidence", "") if isinstance(meta, dict) else ""
                req_rev = 1 if (isinstance(meta, dict) and meta.get("manual_verification_required", False)) else 0

                cursor.execute("""
                    INSERT INTO extracted_fields (
                        document_id, field_name, extracted_value, confidence,
                        source_page, evidence_text, manual_verification_required
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (document_id, field_name, val, conf, page_n, ev, req_rev))

            # 3. Insert triggered rules
            for rule in triggered_rules:
                cursor.execute("""
                    INSERT INTO rule_triggers (
                        document_id, rule_id, rule_name, severity,
                        explanation, evidence, recommended_action
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    document_id,
                    rule.get("Rule ID", "UNKNOWN"),
                    rule.get("Rule Name", "Generic Check"),
                    rule.get("Severity", "INFO"),
                    rule.get("Explanation", ""),
                    rule.get("Evidence", ""),
                    rule.get("Recommended Action", "")
                ))

            # 4. Insert ML Prediction
            cursor.execute("""
                INSERT INTO ml_predictions (
                    document_id, predicted_category, confidence,
                    probabilities_json, model_name, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                document_id,
                ml_result.get("predicted_category", "UNKNOWN"),
                ml_result.get("confidence", 0.0),
                json.dumps(ml_result.get("probabilities", {})),
                ml_result.get("model_name", "Random Forest Document Classifier"),
                timestamp
            ))

            # 5. Insert audit log
            escalation_note = f" (Escalated to admin: {escalation_reason})" if requires_admin_review else " (Auto-verified)"
            cursor.execute("""
                INSERT INTO audit_history (
                    document_id, user_id, action, actor, timestamp, details
                ) VALUES (?, ?, 'DOCUMENT_PROCESSED', 'SYSTEM', ?, ?)
            """, (
                document_id,
                user_id,
                timestamp,
                f"Processed '{filename}' with status '{verification_status}'{escalation_note}."
            ))

            return document_id
    finally:
        conn.close()


def get_documents_by_user(user_id: int, include_deleted: bool = False) -> List[Dict[str, Any]]:
    """
    Fetches documents strictly for the designated user (Privacy Isolation).
    """
    init_database()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM documents WHERE user_id = ?"
        if not include_deleted:
            query += " AND is_deleted = 0"
        query += " ORDER BY id DESC"
        cursor.execute(query, (user_id,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_all_documents(include_deleted: bool = False) -> List[Dict[str, Any]]:
    """Fetches all documents (Admin only)."""
    init_database()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM documents"
        if not include_deleted:
            query += " WHERE is_deleted = 0"
        query += " ORDER BY id DESC"
        cursor.execute(query)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_document_details(
    document_id: int,
    requesting_user_id: Optional[int] = None,
    is_admin: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Fetches comprehensive details for a single document ID.
    Enforces privacy access controls: Non-admins can only view their own documents.
    """
    init_database()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
        doc = cursor.fetchone()
        if not doc:
            return None

        doc_dict = dict(doc)

        # Enforce Privacy Access Boundary:
        if not is_admin and requesting_user_id is not None:
            if doc_dict.get("user_id") != requesting_user_id:
                # Unauthorized access attempt
                now = datetime.now().isoformat()
                cursor.execute("""
                    INSERT INTO audit_history (document_id, user_id, action, actor, timestamp, details)
                    VALUES (?, ?, 'UNAUTHORIZED_ACCESS_BLOCKED', 'SECURITY', ?, 'User attempted to access unauthorized document.')
                """, (document_id, requesting_user_id, now))
                conn.commit()
                return None

        cursor.execute("SELECT * FROM extracted_fields WHERE document_id = ?", (document_id,))
        fields = [dict(row) for row in cursor.fetchall()]

        cursor.execute("SELECT * FROM rule_triggers WHERE document_id = ?", (document_id,))
        rules = [dict(row) for row in cursor.fetchall()]

        cursor.execute("SELECT * FROM ml_predictions WHERE document_id = ?", (document_id,))
        ml = cursor.fetchone()

        cursor.execute("SELECT * FROM human_reviews WHERE document_id = ? ORDER BY id DESC", (document_id,))
        reviews = [dict(row) for row in cursor.fetchall()]

        cursor.execute("SELECT * FROM audit_history WHERE document_id = ? ORDER BY id DESC", (document_id,))
        audit = [dict(row) for row in cursor.fetchall()]

        doc_dict["fields"] = fields
        doc_dict["rules"] = rules
        doc_dict["ml"] = dict(ml) if ml else None
        doc_dict["reviews"] = reviews
        doc_dict["audit"] = audit

        return doc_dict
    finally:
        conn.close()


# ===========================================================================
# ADMIN REVIEW QUEUE & DECISION RECORDING
# ===========================================================================

def get_admin_review_queue(filter_type: str = "PENDING") -> List[Dict[str, Any]]:
    """
    Retrieves filtered documents for the Admin Review Dashboard.
    Filters:
    - 'PENDING': requires_admin_review = 1 and admin_review_status = 'PENDING'
    - 'AI_FLAGGED': Inconsistencies detected by rule engine or ML
    - 'LOW_CONFIDENCE': OCR confidence < 65%
    - 'INCONSISTENCIES': Status contains INCONSISTENT
    - 'ALL': All non-deleted documents
    """
    init_database()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        base_query = """
            SELECT d.*, u.full_name as applicant_name, u.email as applicant_email
            FROM documents d
            LEFT JOIN users u ON d.user_id = u.id
            WHERE d.is_deleted = 0
        """

        if filter_type == "PENDING":
            query = base_query + " AND d.requires_admin_review = 1 AND d.admin_review_status = 'PENDING' ORDER BY d.id DESC"
        elif filter_type == "AI_FLAGGED":
            query = base_query + " AND (d.verification_status LIKE '%INCONSISTEN%' OR d.escalation_reason LIKE '%contradiction%') ORDER BY d.id DESC"
        elif filter_type == "LOW_CONFIDENCE":
            query = base_query + " AND d.ocr_confidence < 65.0 ORDER BY d.id DESC"
        elif filter_type == "INCONSISTENCIES":
            query = base_query + " AND d.verification_status LIKE '%INCONSISTEN%' ORDER BY d.id DESC"
        else:
            query = base_query + " ORDER BY d.id DESC"

        cursor.execute(query)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def record_admin_review_decision(
    document_id: int,
    admin_user_id: int,
    admin_name: str,
    decision: str,  # 'APPROVED', 'REJECTED', 'INFO_REQUESTED', 'UNRESOLVED'
    notes: str,
    field_corrections: Optional[Dict[str, str]] = None
) -> bool:
    """
    Records an official admin review determination and updates document status.
    """
    init_database()
    timestamp = datetime.now().isoformat()
    conn = get_db_connection()

    status_map = {
        "APPROVED": "CONSISTENCY CHECKS PASSED",
        "REJECTED": "INCONSISTENCIES DETECTED",
        "INFO_REQUESTED": "NEEDS MANUAL REVIEW",
        "UNRESOLVED": "UNABLE TO VERIFY"
    }
    final_verification_status = status_map.get(decision, "NEEDS MANUAL REVIEW")

    try:
        with conn:
            cursor = conn.cursor()

            # 1. Update Document record
            cursor.execute("""
                UPDATE documents
                SET admin_review_status = ?,
                    verification_status = ?,
                    admin_notes = ?,
                    reviewed_by = ?,
                    reviewed_at = ?,
                    requires_admin_review = ?
                WHERE id = ?
            """, (
                decision,
                final_verification_status,
                notes,
                admin_name,
                timestamp,
                0 if decision in ["APPROVED", "REJECTED"] else 1,
                document_id
            ))

            # 2. Record in human_reviews
            cursor.execute("""
                INSERT INTO human_reviews (
                    document_id, reviewer_id, reviewer_name, decision, review_notes, reviewed_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (document_id, admin_user_id, admin_name, decision, notes, timestamp))

            # 3. Apply field corrections if supplied
            if field_corrections:
                for f_name, new_val in field_corrections.items():
                    cursor.execute("""
                        UPDATE extracted_fields
                        SET corrected_value = ?, is_manually_corrected = 1
                        WHERE document_id = ? AND field_name = ?
                    """, (new_val, document_id, f_name))

                    cursor.execute("""
                        INSERT INTO audit_history (document_id, user_id, action, actor, timestamp, details)
                        VALUES (?, ?, 'FIELD_CORRECTED_BY_ADMIN', ?, ?, ?)
                    """, (document_id, admin_user_id, admin_name, timestamp, f"Field '{f_name}' corrected to: '{new_val}'"))

            # 4. Insert Audit Log
            cursor.execute("""
                INSERT INTO audit_history (document_id, user_id, action, actor, timestamp, details)
                VALUES (?, ?, 'ADMIN_DECISION_RECORDED', ?, ?, ?)
            """, (
                document_id,
                admin_user_id,
                admin_name,
                timestamp,
                f"Admin decision: '{decision}'. Final Status: '{final_verification_status}'. Notes: {notes}"
            ))

            return True
    finally:
        conn.close()


def update_field_value(
    document_id: int,
    field_name: str,
    new_value: str,
    reviewer_name: str
) -> bool:
    """Updates an individual field value."""
    init_database()
    timestamp = datetime.now().isoformat()
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE extracted_fields
                SET corrected_value = ?, is_manually_corrected = 1
                WHERE document_id = ? AND field_name = ?
            """, (new_value, document_id, field_name))

            cursor.execute("""
                INSERT INTO audit_history (
                    document_id, action, actor, timestamp, details
                ) VALUES (?, 'FIELD_CORRECTED', ?, ?, ?)
            """, (
                document_id,
                reviewer_name,
                timestamp,
                f"Field '{field_name}' corrected to: '{new_value}'"
            ))
            return True
    finally:
        conn.close()


def record_human_decision(
    document_id: int,
    reviewer_name: str,
    decision: str,
    notes: str
) -> bool:
    """Legacy wrapper for human decision recording."""
    return record_admin_review_decision(
        document_id=document_id,
        admin_user_id=1,
        admin_name=reviewer_name,
        decision=decision,
        notes=notes
    )


# ===========================================================================
# REPORT RECORDS PERSISTENCE
# ===========================================================================

def generate_report_reference(document_id: int) -> str:
    """Generates a verifiable report reference code (e.g. DOCV-2024-8841-A9)."""
    rand_hex = secrets.token_hex(2).upper()
    return f"DOCV-2024-{document_id:04d}-{rand_hex}"


def create_report_record(
    document_id: int,
    user_id: int,
    report_type: str,
    final_status: str,
    summary_data: Dict[str, Any]
) -> str:
    """Persists a report record and returns its unique reference code."""
    init_database()
    now = datetime.now().isoformat()
    ref = generate_report_reference(document_id)
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO reports (document_id, user_id, report_reference, report_type, generated_at, final_status, summary_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (document_id, user_id, ref, report_type, now, final_status, json.dumps(summary_data)))

            cursor.execute("""
                INSERT INTO audit_history (document_id, user_id, action, actor, timestamp, details)
                VALUES (?, ?, 'REPORT_GENERATED', 'SYSTEM', ?, ?)
            """, (document_id, user_id, now, f"Generated {report_type} reference {ref}."))
            return ref
    finally:
        conn.close()


def get_reports_for_user(user_id: int) -> List[Dict[str, Any]]:
    """Retrieves all reports belonging to a specific user."""
    init_database()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT r.*, d.filename, d.declared_category
            FROM reports r
            JOIN documents d ON r.document_id = d.id
            WHERE r.user_id = ? AND d.is_deleted = 0
            ORDER BY r.id DESC
        """, (user_id,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_all_reports() -> List[Dict[str, Any]]:
    """Retrieves all reports for admin review."""
    init_database()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT r.*, d.filename, d.declared_category, u.email as user_email
            FROM reports r
            JOIN documents d ON r.document_id = d.id
            JOIN users u ON r.user_id = u.id
            WHERE d.is_deleted = 0
            ORDER BY r.id DESC
        """)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


# ===========================================================================
# PRIVACY DELETION & SYSTEM STATISTICS
# ===========================================================================

def delete_document_record(document_id: int, actor: str = "USER", user_id: Optional[int] = None) -> bool:
    """
    Privacy compliance: Purges all stored data related to a document.
    Enforces authorization check if user_id is provided.
    """
    init_database()
    timestamp = datetime.now().isoformat()
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.cursor()

            if user_id is not None:
                cursor.execute("SELECT user_id FROM documents WHERE id = ?", (document_id,))
                row = cursor.fetchone()
                if not row or row["user_id"] != user_id:
                    return False

            cursor.execute("DELETE FROM reports WHERE document_id = ?", (document_id,))
            cursor.execute("DELETE FROM extracted_fields WHERE document_id = ?", (document_id,))
            cursor.execute("DELETE FROM rule_triggers WHERE document_id = ?", (document_id,))
            cursor.execute("DELETE FROM ml_predictions WHERE document_id = ?", (document_id,))
            cursor.execute("DELETE FROM human_reviews WHERE document_id = ?", (document_id,))
            cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))

            cursor.execute("""
                INSERT INTO audit_history (
                    document_id, user_id, action, actor, timestamp, details
                ) VALUES (?, ?, 'DOCUMENT_PURGED', ?, ?, ?)
            """, (
                document_id,
                user_id,
                actor,
                timestamp,
                f"Document ID #{document_id} and all associated private records permanently deleted."
            ))
            return True
    finally:
        conn.close()


def get_recent_activity_for_user(user_id: int, limit: int = 6) -> List[Dict[str, Any]]:
    """Retrieves recent meaningful activity records for a specific applicant."""
    init_database()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT a.id, a.action, a.actor, a.timestamp, a.details, a.document_id, d.filename
            FROM audit_history a
            LEFT JOIN documents d ON a.document_id = d.id
            WHERE a.user_id = ? AND a.action NOT IN ('USER_LOGIN_FAILED', 'UNAUTHORIZED_ACCESS_BLOCKED')
            ORDER BY a.id DESC LIMIT ?
        """, (user_id, limit))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_system_statistics() -> Dict[str, Any]:
    """Computes comprehensive summary statistics for Dashboard view."""
    init_database()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM documents WHERE is_deleted = 0")
        total_docs = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM documents 
            WHERE is_deleted = 0 AND verification_status = 'CONSISTENCY CHECKS PASSED'
        """)
        consistent_docs = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM documents 
            WHERE is_deleted = 0 AND verification_status = 'NEEDS MANUAL REVIEW'
        """)
        needs_review_docs = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM documents 
            WHERE is_deleted = 0 AND verification_status = 'INCONSISTENCIES DETECTED'
        """)
        inconsistent_docs = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM documents 
            WHERE is_deleted = 0 AND verification_status = 'UNABLE TO VERIFY'
        """)
        unable_docs = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM documents 
            WHERE is_deleted = 0 AND requires_admin_review = 1 AND admin_review_status = 'PENDING'
        """)
        pending_admin_reviews = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM human_reviews")
        manual_reviews_done = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]

        return {
            "total_documents": total_docs,
            "consistent_documents": consistent_docs,
            "needs_review_documents": needs_review_docs,
            "inconsistent_documents": inconsistent_docs,
            "unable_to_verify_documents": unable_docs,
            "pending_admin_reviews": pending_admin_reviews,
            "manual_reviews_done": manual_reviews_done,
            "total_users": total_users
        }
    finally:
        conn.close()
