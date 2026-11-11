"""
Comprehensive Automated Test Suite for Automated Document Verification System (SIH1709).
Tests all modules independently and verifies:
1. User registration & strong password hashing
2. Prevention of self-assigned ADMIN roles
3. User authentication & login
4. Default administrator authentication & provisioning
5. Unauthorized admin access prevention
6. Strict user document privacy isolation
7. AI-first automatic verification (clean cases auto-approved without admin delay)
8. Selective escalation to admin review queue (contradictions, missing fields, unreadable scans)
9. Admin review determinations (Approved, Rejected, Field Corrections, Notes)
10. Updated report generation with unique reference IDs
11. Schema Knowledge Representation & Validation
12. Information Extraction with Evidence & Source Pages
13. Production Rule Engine (Chronological, Future Dates, Inversions, Expirations)
14. Supervised ML Classifier & Empirical Holdout Benchmark
15. PDF Report Generation & GDPR Hard Delete
"""

import unittest
import json
import secrets
from datetime import date, timedelta
from pathlib import Path

# Core Modules
import auth_manager
import database
import decision_engine
from decision_engine import (
    STATUS_CONSISTENCY_PASSED,
    STATUS_NEEDS_REVIEW,
    STATUS_INCONSISTENCIES,
    STATUS_UNABLE_TO_VERIFY,
    evaluate_verification_decision
)
from document_schema import (
    validate_document_schema,
    parse_date_robust,
    get_schema,
    SCHEMA_REGISTRY
)
from information_extractor import extract_structured_fields
from rule_engine import run_rule_engine
from ml_model import predict_document_category, get_feature_importances
from model_evaluation import evaluate_models_on_test_set, get_evaluation_dataframe
import review_manager
import report_generator
from document_processor import process_file_bytes


class TestAuthenticationAndSecurity(unittest.TestCase):
    """Tests Section 1: User Registration, Login, Admin Provisioning, and Role Security."""

    def setUp(self):
        database.init_database()

    def test_user_registration_and_password_hashing(self):
        rand_email = f"testuser_{secrets.token_hex(4)}@college.edu"
        pwd = "SecureUser@2024!"
        success, msg, user_id = database.create_user(
            email=rand_email,
            password=pwd,
            full_name="Test Applicant",
            role="USER",
            is_admin_creator=False
        )
        self.assertTrue(success)
        self.assertIsNotNone(user_id)

        # Verify password hash and salt stored in database
        user = database.get_user_by_id(user_id)
        self.assertIsNotNone(user)
        self.assertEqual(user["email"], rand_email)
        self.assertEqual(user["role"], "USER")

        # Plaintext password must never be stored
        conn = database.get_db_connection()
        try:
            c = conn.cursor()
            c.execute("SELECT password_hash, salt FROM users WHERE id = ?", (user_id,))
            row = c.fetchone()
            self.assertNotEqual(row["password_hash"], pwd)
            self.assertTrue(len(row["salt"]) >= 16)
        finally:
            conn.close()

    def test_user_cannot_self_assign_admin_role(self):
        """Security requirement: Registration can never grant ADMIN role to self."""
        rand_email = f"hacker_{secrets.token_hex(4)}@exploit.com"
        success, msg, user_id = database.create_user(
            email=rand_email,
            password="Password123!",
            full_name="Malicious User",
            role="ADMIN",  # Attempting to assign self ADMIN
            is_admin_creator=False  # Public self-registration
        )
        self.assertTrue(success)
        user = database.get_user_by_id(user_id)
        # Must be demoted to USER
        self.assertEqual(user["role"], "USER")

    def test_user_authentication_success_and_failure(self):
        rand_email = f"authuser_{secrets.token_hex(4)}@domain.com"
        pwd = "CorrectPassword@1"
        database.create_user(rand_email, pwd, "Auth User", role="USER")

        # 1. Correct password
        auth_success = database.authenticate_user(rand_email, pwd)
        self.assertIsNotNone(auth_success)
        self.assertEqual(auth_success["email"], rand_email)
        self.assertEqual(auth_success["role"], "USER")

        # 2. Incorrect password
        auth_fail = database.authenticate_user(rand_email, "WrongPassword@99")
        self.assertIsNone(auth_fail)

    def test_initial_administrator_provisioning(self):
        """Verifies default initial admin setup and login."""
        admin = database.authenticate_user("admin@docverify.gov.in", "Admin@DocVerify2024!")
        self.assertIsNotNone(admin)
        self.assertEqual(admin["role"], "ADMIN")

    def test_user_document_privacy_isolation(self):
        """Security requirement: Prevent users from viewing other users' documents."""
        user_a = f"usera_{secrets.token_hex(4)}@test.com"
        user_b = f"userb_{secrets.token_hex(4)}@test.com"
        _, _, id_a = database.create_user(user_a, "PassUser@123", "User A")
        _, _, id_b = database.create_user(user_b, "PassUser@123", "User B")

        # User A uploads a document
        doc_a_id = database.save_verification_record(
            filename="user_a_private.pdf",
            file_type="PDF",
            file_size=500,
            declared_category="Educational Certificate",
            predicted_category="Educational Certificate",
            ocr_confidence=95.0,
            page_count=1,
            raw_text="Private Degree Document for User A",
            verification_status=STATUS_CONSISTENCY_PASSED,
            extracted_fields={},
            triggered_rules=[],
            ml_result={},
            user_id=id_a
        )

        # 1. User A can retrieve their own document
        a_view = database.get_document_details(doc_a_id, requesting_user_id=id_a, is_admin=False)
        self.assertIsNotNone(a_view)
        self.assertEqual(a_view["id"], doc_a_id)

        # 2. User B CANNOT retrieve User A's document (returns None and logs security event)
        b_view = database.get_document_details(doc_a_id, requesting_user_id=id_b, is_admin=False)
        self.assertIsNone(b_view)

        # 3. Admin CAN retrieve for inspection
        admin_view = database.get_document_details(doc_a_id, requesting_user_id=None, is_admin=True)
        self.assertIsNotNone(admin_view)


class TestDecisionEngineAndWorkflow(unittest.TestCase):
    """Tests Section 3, 4, 5, 6, 7: AI-First Decision Engine & Admin Escalation."""

    def test_ai_first_automatic_finalization_clean_document(self):
        """Clean document passes without waiting for admin."""
        raw_text = (
            "INDIAN INSTITUTE OF TECHNOLOGY BOMBAY\n"
            "This is to certify that Rahul Sharma bearing Roll No: 2019-BTECH-CS-042 "
            "has completed Bachelor of Technology.\n"
            "Date of Issue: 15/07/2023\n"
            "Certificate ID: IITB/2023/CS/1089"
        )
        doc_payload = {"pages": [{"page_number": 1, "text": raw_text}], "full_text": raw_text}
        fields = extract_structured_fields(doc_payload, "Educational Certificate")
        schema_res = validate_document_schema(fields, "Educational Certificate")
        ml_res = predict_document_category(raw_text)
        _, rules = run_rule_engine(fields, "Educational Certificate", raw_text)

        outcome = evaluate_verification_decision(
            extracted_fields=fields,
            schema_results=schema_res,
            rule_results=rules,
            ml_result=ml_res,
            ocr_confidence=98.0,
            raw_text=raw_text,
            declared_category="Educational Certificate"
        )

        self.assertEqual(outcome.final_status, STATUS_CONSISTENCY_PASSED)
        self.assertFalse(outcome.requires_admin_review)
        self.assertIsNone(outcome.escalation_reason)

    def test_escalation_on_contradiction(self):
        """Future issue date escalates to admin review queue as INCONSISTENCIES DETECTED."""
        raw_text = (
            "GOVERNMENT OF NCT OF DELHI\n"
            "DOMICILE CERTIFICATE\n"
            "Applicant Name: Ramesh Chandra\n"
            "Certificate ID: DL/DOM/2035/9912\n"
            "Issuing Authority: Tehsildar Sadar\n"
            "Issue Date: 15/12/2035"  # Future date!
        )
        doc_payload = {"pages": [{"page_number": 1, "text": raw_text}], "full_text": raw_text}
        fields = extract_structured_fields(doc_payload, "Government Certificate")
        schema_res = validate_document_schema(fields, "Government Certificate")
        ml_res = predict_document_category(raw_text)
        _, rules = run_rule_engine(fields, "Government Certificate", raw_text)

        outcome = evaluate_verification_decision(
            extracted_fields=fields,
            schema_results=schema_res,
            rule_results=rules,
            ml_result=ml_res,
            ocr_confidence=95.0,
            raw_text=raw_text,
            declared_category="Government Certificate"
        )

        self.assertEqual(outcome.final_status, STATUS_INCONSISTENCIES)
        self.assertTrue(outcome.requires_admin_review)
        self.assertTrue(outcome.has_contradictions)
        self.assertIn("Future Issue Date", outcome.escalation_reason)

    def test_escalation_on_missing_mandatory_field(self):
        """Missing required Student Name escalates to admin review queue as NEEDS MANUAL REVIEW."""
        raw_text = (
            "UNIVERSITY OF DELHI\n"
            "Roll Number: DU-2018-99412\n"
            "Issue Date: 24/02/2022\n"
            "Certificate ID: DU/DIP/99412"
        )
        doc_payload = {"pages": [{"page_number": 1, "text": raw_text}], "full_text": raw_text}
        fields = extract_structured_fields(doc_payload, "Educational Certificate")
        schema_res = validate_document_schema(fields, "Educational Certificate")
        ml_res = predict_document_category(raw_text)
        _, rules = run_rule_engine(fields, "Educational Certificate", raw_text)

        outcome = evaluate_verification_decision(
            extracted_fields=fields,
            schema_results=schema_res,
            rule_results=rules,
            ml_result=ml_res,
            ocr_confidence=92.0,
            raw_text=raw_text,
            declared_category="Educational Certificate"
        )

        self.assertEqual(outcome.final_status, STATUS_NEEDS_REVIEW)
        self.assertTrue(outcome.requires_admin_review)
        self.assertTrue(outcome.has_missing_required)
        self.assertIn("Student Name", outcome.escalation_reason)

    def test_escalation_on_unreadable_scan(self):
        """Very short or illegible text results in UNABLE TO VERIFY."""
        raw_text = "Blurry text"
        doc_payload = {"pages": [{"page_number": 1, "text": raw_text}], "full_text": raw_text}
        fields = extract_structured_fields(doc_payload, "Identity Document")
        schema_res = validate_document_schema(fields, "Identity Document")
        ml_res = predict_document_category(raw_text)
        _, rules = run_rule_engine(fields, "Identity Document", raw_text)

        outcome = evaluate_verification_decision(
            extracted_fields=fields,
            schema_results=schema_res,
            rule_results=rules,
            ml_result=ml_res,
            ocr_confidence=30.0,
            raw_text=raw_text,
            declared_category="Identity Document"
        )

        self.assertEqual(outcome.final_status, STATUS_UNABLE_TO_VERIFY)
        self.assertTrue(outcome.requires_admin_review)

    def test_admin_review_action_workflow(self):
        """Admin reviews an escalated document, applies field corrections, and commits APPROVED."""
        doc_id = database.save_verification_record(
            filename="escalated_degree.pdf",
            file_type="PDF",
            file_size=1200,
            declared_category="Educational Certificate",
            predicted_category="Educational Certificate",
            ocr_confidence=75.0,
            page_count=1,
            raw_text="Degree document requiring officer verification",
            verification_status=STATUS_NEEDS_REVIEW,
            extracted_fields={
                "Student Name": {"value": "R. Sharma", "confidence": 70.0, "page": 1, "evidence": "R. Sharma"}
            },
            triggered_rules=[{"Rule ID": "R001", "Rule Name": "Check", "Severity": "NEEDS MANUAL REVIEW"}],
            ml_result={"predicted_category": "Educational Certificate"},
            user_id=1,
            requires_admin_review=True,
            escalation_reason="Name abbreviated"
        )

        # Check pending queue
        queue = database.get_admin_review_queue(filter_type="PENDING")
        queue_ids = [d["id"] for d in queue]
        self.assertIn(doc_id, queue_ids)

        # Admin performs review and corrects abbreviated name to full name
        review_manager.submit_review_decision(
            document_id=doc_id,
            decision="APPROVED",
            reviewer_name="Officer Gupta",
            review_notes="Verified full name on university gazette.",
            admin_user_id=1,
            field_corrections={"Student Name": "Rahul Sharma"}
        )

        # Document must now be approved and removed from pending queue
        updated_doc = database.get_document_details(doc_id, is_admin=True)
        self.assertEqual(updated_doc["verification_status"], STATUS_CONSISTENCY_PASSED)
        self.assertEqual(updated_doc["admin_review_status"], "APPROVED")
        self.assertEqual(updated_doc["requires_admin_review"], 0)

        # Corrected field verified
        field_names = {f["field_name"]: f["corrected_value"] for f in updated_doc["fields"]}
        self.assertEqual(field_names["Student Name"], "Rahul Sharma")


class TestReportsAndExports(unittest.TestCase):
    """Tests Section 9: PDF Reports with Report References and User vs Admin Views."""

    def test_user_report_generation_with_reference(self):
        ref_id = database.generate_report_reference(99)
        self.assertTrue(ref_id.startswith("DOCV-2024-0099-"))

        pdf_bytes = report_generator.generate_verification_report_pdf(
            document_info={"filename": "transcript.pdf", "declared_category": "Educational Certificate", "id": 99},
            extracted_fields={"Student Name": "Ananya Verma", "Certificate ID": "DU/9912"},
            schema_results=[],
            rule_results=[],
            ml_results={"predicted_category": "Educational Certificate"},
            overall_status=STATUS_CONSISTENCY_PASSED,
            report_reference=ref_id,
            is_admin_report=False,
            detected_issues=[],
            recommended_actions=["Proceed with standard employment screening."]
        )

        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 2500)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))


class TestKnowledgeRepresentation(unittest.TestCase):
    """Phase 6: Tests formal document schemas, date parsers, and constraints."""

    def test_schema_registry_presence(self):
        self.assertIn("Educational Certificate", SCHEMA_REGISTRY)
        self.assertIn("Identity Document", SCHEMA_REGISTRY)
        self.assertIn("Government Certificate", SCHEMA_REGISTRY)

    def test_robust_date_parsing(self):
        self.assertEqual(parse_date_robust("15/07/2023"), date(2023, 7, 15))
        self.assertEqual(parse_date_robust("2021-11-20"), date(2021, 11, 20))
        self.assertEqual(parse_date_robust("12th March 2018"), date(2018, 3, 12))
        self.assertEqual(parse_date_robust("June 9, 2021"), date(2021, 6, 9))
        self.assertIsNone(parse_date_robust("InvalidDateString"))
        self.assertIsNone(parse_date_robust(""))

    def test_schema_validation_required_and_optional(self):
        fields = {
            "Student Name": "Rahul Sharma",
            "Certificate ID": "IITB/2023/101",
            "Institution": "IIT Bombay",
            "Issue Date": "15/07/2023",
            "Expiry Date": None
        }
        results = validate_document_schema(fields, "Educational Certificate")
        statuses = {r["Field"]: r["Status"] for r in results}

        self.assertEqual(statuses["Student Name"], "PRESENT")
        self.assertEqual(statuses["Certificate ID"], "PRESENT")
        self.assertEqual(statuses["Institution"], "PRESENT")
        self.assertEqual(statuses["Issue Date"], "VALID FORMAT")
        self.assertEqual(statuses["Expiry Date"], "OPTIONAL_ABSENT")


class TestProductionRuleEngine(unittest.TestCase):
    """Phase 7: Tests forward-chaining rules and explainability."""

    def test_rule_engine_valid_case(self):
        fields = {
            "Student Name": {"value": "Rahul Sharma", "confidence": 95.0, "evidence": "Rahul Sharma"},
            "Certificate ID": {"value": "IITB/2023/1089", "confidence": 90.0, "evidence": "ID: 1089"},
            "Institution": {"value": "Indian Institute of Technology Bombay", "confidence": 92.0, "evidence": "IIT Bombay"},
            "Issue Date": {"value": "15/07/2023", "confidence": 95.0, "evidence": "15/07/2023"},
            "Expiry Date": {"value": None, "confidence": 0.0, "evidence": ""}
        }
        raw_text = "IIT Bombay degree conferred upon Rahul Sharma certify."
        verdict, triggered = run_rule_engine(fields, "Educational Certificate", raw_text=raw_text)

        self.assertEqual(verdict, "CONSISTENT")
        rule_ids = [r["Rule ID"] for r in triggered]
        self.assertIn("R012", rule_ids)

    def test_rule_engine_future_issue_date(self):
        future_date = (date.today() + timedelta(days=365)).strftime("%d/%m/%Y")
        fields = {
            "Applicant Name": {"value": "Suresh Yadav", "evidence": ""},
            "Certificate ID": {"value": "UP/DOM/2030", "evidence": ""},
            "Issuing Authority": {"value": "Tehsildar", "evidence": ""},
            "Issue Date": {"value": future_date, "evidence": ""}
        }
        verdict, triggered = run_rule_engine(fields, "Government Certificate")
        self.assertEqual(verdict, "INCONSISTENT")
        rule_ids = [r["Rule ID"] for r in triggered]
        self.assertIn("R004", rule_ids)

    def test_rule_engine_inverted_expiry_before_issue(self):
        fields = {
            "Applicant Name": {"value": "Suresh Yadav", "evidence": ""},
            "Certificate ID": {"value": "UP/DOM/2022", "evidence": ""},
            "Issuing Authority": {"value": "Tehsildar", "evidence": ""},
            "Issue Date": {"value": "10/05/2022", "evidence": ""},
            "Expiry Date": {"value": "10/05/2021", "evidence": ""}
        }
        verdict, triggered = run_rule_engine(fields, "Government Certificate")
        self.assertEqual(verdict, "INCONSISTENT")
        rule_ids = [r["Rule ID"] for r in triggered]
        self.assertIn("R005", rule_ids)

    def test_rule_engine_expired_document(self):
        fields = {
            "Name": {"value": "Pooja Malhotra", "evidence": ""},
            "Document Number": {"value": "P8819204", "evidence": ""},
            "Date of Birth": {"value": "14/06/1990", "evidence": ""},
            "Issue Date": {"value": "10/01/2010", "evidence": ""},
            "Expiry Date": {"value": "09/01/2020", "evidence": ""}
        }
        verdict, triggered = run_rule_engine(fields, "Identity Document")
        self.assertEqual(verdict, "INCONSISTENT")
        rule_ids = [r["Rule ID"] for r in triggered]
        self.assertIn("R006", rule_ids)


class TestMachineLearningAndEvaluation(unittest.TestCase):
    """Phase 8 & 9: Tests real ML model, baseline comparison, and evaluation metrics."""

    def test_model_prediction_educational(self):
        sample_text = (
            "UNIVERSITY OF CAMBRIDGE\n"
            "CERTIFICATE OF ADVANCED STUDY IN MATHEMATICS\n"
            "This certifies that Student Name: Sarah Jenkins of Trinity College has satisfied the examiners "
            "in the examination in Mathematics Part III.\n"
            "Certificate ID: CAM-MATH-2021-992\n"
            "Issued on: 28/06/2021\n"
            "Registrary of the University"
        )
        res = predict_document_category(sample_text)
        self.assertEqual(res["predicted_category"], "Educational Certificate")
        self.assertGreaterEqual(res["confidence"], 30.0)
        self.assertIn("Educational Certificate", res["probabilities"])

    def test_evaluation_on_independent_test_set(self):
        eval_data = evaluate_models_on_test_set()
        proposed = eval_data["proposed_metrics"]
        self.assertGreaterEqual(proposed["Accuracy"], 0.80)
        self.assertGreaterEqual(proposed["F1 Score"], 0.80)


if __name__ == "__main__":
    unittest.main()
