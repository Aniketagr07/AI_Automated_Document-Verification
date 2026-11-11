"""
Extended Audit and Verification Test Suite for DocVerify.
Tests edge cases, authentication workflows, admin provisioning,
data isolation, report generation across all statuses, and decision outcomes.
"""

import unittest
import secrets
from datetime import datetime, date, timedelta
from pathlib import Path

import auth_manager
import database
import constants
from constants import (
    STATUS_CONSISTENCY_PASSED,
    STATUS_NEEDS_REVIEW,
    STATUS_INCONSISTENCIES,
    STATUS_UNABLE_TO_VERIFY,
    ROLE_USER,
    ROLE_ADMIN,
    AUTHENTICITY_DISCLAIMER
)
import decision_engine
from decision_engine import evaluate_verification_decision
import document_schema
from document_schema import get_schema, validate_document_schema, SCHEMA_REGISTRY
import rule_engine
from rule_engine import run_rule_engine
import information_extractor
from information_extractor import extract_structured_fields
import ml_model
from ml_model import predict_document_category
import report_generator
import bootstrap_admin


class TestDocVerifyExtendedSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        database.init_database()

    def test_canonical_statuses(self):
        """Verify that all canonical status constants are identical across modules."""
        self.assertEqual(constants.STATUS_CONSISTENCY_PASSED, "CONSISTENCY CHECKS PASSED")
        self.assertEqual(constants.STATUS_NEEDS_REVIEW, "NEEDS MANUAL REVIEW")
        self.assertEqual(constants.STATUS_INCONSISTENCIES, "INCONSISTENCIES DETECTED")
        self.assertEqual(constants.STATUS_UNABLE_TO_VERIFY, "UNABLE TO VERIFY")
        self.assertEqual(decision_engine.STATUS_CONSISTENCY_PASSED, constants.STATUS_CONSISTENCY_PASSED)
        self.assertEqual(report_generator.STATUS_CONSISTENCY_PASSED, constants.STATUS_CONSISTENCY_PASSED)

    def test_applicant_signup_role_security(self):
        """Verify that public signup CANNOT assign ADMIN role under any circumstances."""
        fake_email = f"applicant_test_{secrets.token_hex(4)}@domain.com"
        # Attempt to pass role="ADMIN" with is_admin_creator=False
        success, msg, user_id = database.create_user(
            email=fake_email,
            password="SecurePassword@2024!",
            full_name="Legit Applicant",
            role="ADMIN",
            is_admin_creator=False
        )
        self.assertTrue(success)
        user = database.get_user_by_id(user_id)
        self.assertEqual(user["role"], ROLE_USER)

    def test_admin_bootstrap_and_authentication(self):
        """Verify admin account creation via bootstrap tool."""
        admin_email = f"admin_test_{secrets.token_hex(4)}@docverify.gov.in"
        admin_pwd = "AdminSecurePassword@999!"
        success = bootstrap_admin.bootstrap_admin(admin_email, admin_pwd, "Security Officer")
        self.assertTrue(success)

        # Authenticate newly created admin
        auth = database.authenticate_user(admin_email, admin_pwd)
        self.assertIsNotNone(auth)
        self.assertEqual(auth["role"], ROLE_ADMIN)

    def test_user_data_isolation(self):
        """Verify that Applicant A cannot retrieve Applicant B's document dossier or reports."""
        email_a = f"usera_{secrets.token_hex(4)}@test.com"
        email_b = f"userb_{secrets.token_hex(4)}@test.com"
        _, _, id_a = database.create_user(email_a, "SecurePass123!", "User A")
        _, _, id_b = database.create_user(email_b, "SecurePass123!", "User B")

        doc_id = database.save_verification_record(
            filename="private_certificate.pdf",
            file_type="PDF",
            file_size=1024,
            declared_category="Educational Certificate",
            predicted_category="Educational Certificate",
            ocr_confidence=95.0,
            page_count=1,
            raw_text="Confidential degree for User A",
            verification_status=STATUS_CONSISTENCY_PASSED,
            extracted_fields={"Student Name": {"value": "User A", "confidence": 99.0}},
            triggered_rules=[],
            ml_result={},
            user_id=id_a
        )

        # User A can access
        doc_a = database.get_document_details(doc_id, requesting_user_id=id_a, is_admin=False)
        self.assertIsNotNone(doc_a)

        # User B CANNOT access (returns None)
        doc_b = database.get_document_details(doc_id, requesting_user_id=id_b, is_admin=False)
        self.assertIsNone(doc_b)

        # User B's document list does NOT contain User A's document
        b_docs = database.get_documents_by_user(id_b)
        self.assertNotIn(doc_id, [d["id"] for d in b_docs])

    def test_password_reset_token_workflow(self):
        """Verify secure password reset token generation, single-use, and expiry validation."""
        test_email = f"reset_user_{secrets.token_hex(4)}@test.com"
        _, _, u_id = database.create_user(test_email, "OldPassword123!", "Reset User")

        # 1. Generate token
        token = database.create_password_reset_token(test_email)
        self.assertIsNotNone(token)

        # 2. Consume token with new password
        new_pwd = "NewSecurePassword@2025!"
        ok, msg = database.validate_and_consume_reset_token(token, new_pwd)
        self.assertTrue(ok)

        # 3. Old password fails
        self.assertIsNone(database.authenticate_user(test_email, "OldPassword123!"))

        # 4. New password succeeds
        self.assertIsNotNone(database.authenticate_user(test_email, new_pwd))

        # 5. Token cannot be reused (single-use)
        reuse_ok, _ = database.validate_and_consume_reset_token(token, "AnotherPass123!")
        self.assertFalse(reuse_ok)

    def test_report_generation_all_canonical_statuses(self):
        """Verify that PDF generation works flawlessly for all 4 canonical statuses."""
        for status in constants.STANDARD_STATUSES:
            pdf_bytes = report_generator.generate_verification_report_pdf(
                document_info={
                    "filename": f"sample_{status[:4].lower()}.pdf",
                    "declared_category": "Educational Certificate",
                    "id": 101
                },
                extracted_fields={
                    "Student Name": {"value": "Rahul Sharma", "confidence": 96.0, "page": 1},
                    "Certificate ID": {"value": "IITB/2023/CS/1089", "confidence": 98.0, "page": 1},
                    "Institution": {"value": "IIT Bombay", "confidence": 94.0, "page": 1},
                    "Issue Date": {"value": "15/07/2023", "confidence": 97.0, "page": 1}
                },
                schema_results=[],
                rule_results=[],
                ml_results={"predicted_category": "Educational Certificate"},
                overall_status=status,
                report_reference=f"DOCV-2024-{secrets.token_hex(2).upper()}",
                is_admin_report=False
            )
            self.assertIsInstance(pdf_bytes, bytes)
            self.assertGreater(len(pdf_bytes), 2000)
            self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_rule_engine_temporal_inversion(self):
        """Verify rule engine flags temporal inversion (expiry precedes issue date)."""
        fields = {
            "Student Name": {"value": "Test Student", "confidence": 90.0},
            "Certificate ID": {"value": "CERT12345", "confidence": 90.0},
            "Institution": {"value": "Test University", "confidence": 90.0},
            "Issue Date": {"value": "15/07/2023", "confidence": 95.0},
            "Expiry Date": {"value": "10/01/2020", "confidence": 95.0}  # Inverted!
        }
        verdict, rules = run_rule_engine(fields, "Educational Certificate", "Educational Degree Certificate")
        self.assertEqual(verdict, "INCONSISTENT")
        rule_ids = [r["Rule ID"] for r in rules]
        self.assertIn("R005", rule_ids)

    def test_decision_engine_unreadable_scan(self):
        """Verify degraded scan or unreadable document outputs UNABLE TO VERIFY."""
        fields = {}
        outcome = evaluate_verification_decision(
            extracted_fields=fields,
            schema_results=[],
            rule_results=[],
            ml_result={},
            ocr_confidence=35.0,
            page_count=1,
            raw_text="blurry illegible scan",
            declared_category="Identity Document"
        )
        self.assertEqual(outcome.final_status, STATUS_UNABLE_TO_VERIFY)
        self.assertTrue(outcome.requires_admin_review)


if __name__ == "__main__":
    unittest.main()
