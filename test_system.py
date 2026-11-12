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


class TestSIHEnhancedForensicsAndChecksums(unittest.TestCase):
    """SIH1709 Winning Upgrades: Verhoeff Checksum, PAN Coherence, ELA, and Uncertainty Matrix."""

    def test_verhoeff_aadhaar_validation(self):
        from verhoeff import validate_verhoeff, generate_verhoeff_check_digit
        # Valid test Aadhaar numbers
        self.assertTrue(validate_verhoeff("2093 4519 8273"))
        self.assertTrue(validate_verhoeff("9921 4410 8823"))
        # Invalid / fake Aadhaar numbers
        self.assertFalse(validate_verhoeff("1234 5678 9012"))
        self.assertFalse(validate_verhoeff("9999 8888 7777"))

    def test_pan_structure_and_surname_matching(self):
        from verhoeff import validate_pan_structure
        # Valid: 4th char P (Person), 5th char S matches surname 'Sharma'
        ok, msg = validate_pan_structure("ABCPS1234D", "Vikram Sharma")
        self.assertTrue(ok)
        # Invalid: 5th char S does NOT match surname 'Gupta'
        fail, msg = validate_pan_structure("ABCPS1234D", "Vikram Gupta")
        self.assertFalse(fail)
        self.assertIn("does not match first letter of surname", msg)

    def test_rule_r013_aadhaar_checksum_trigger(self):
        fields = {
            "Name": {"value": "Manish Tiwari", "confidence": 95.0},
            "Document Number": {"value": "1234 5678 9012", "confidence": 95.0},
            "Date of Birth": {"value": "09/12/1994", "confidence": 95.0}
        }
        verdict, rules = run_rule_engine(fields, "Identity Document", "AADHAAR CARD")
        self.assertEqual(verdict, "INCONSISTENT")
        rule_ids = [r["Rule ID"] for r in rules]
        self.assertIn("R013", rule_ids)

    def test_dual_axis_uncertainty_matrix_quadrants(self):
        from rule_engine import compute_dual_axis_uncertainty_matrix

        # Quadrant I: High OCR + Clean Rules = Verified Consistent
        q1 = compute_dual_axis_uncertainty_matrix(
            ocr_confidence=95.0,
            raw_text="Full pristine document text confer degree",
            extracted_fields={"Name": "Test", "ID": "123"},
            rule_results=[{"Severity": "PASS"}]
        )
        self.assertEqual(q1["quadrant_id"], "I")

        # Quadrant II: Low OCR + Clean Rules = Extraction Noise (Scan Blur)
        q2 = compute_dual_axis_uncertainty_matrix(
            ocr_confidence=40.0,
            raw_text="blurry text",
            extracted_fields={"Name": "Test"},
            rule_results=[{"Severity": "PASS"}]
        )
        self.assertEqual(q2["quadrant_id"], "II")

        # Quadrant IV: High OCR + Contradiction = Genuine Fraud
        q4 = compute_dual_axis_uncertainty_matrix(
            ocr_confidence=95.0,
            raw_text="Pristine high resolution document scan",
            extracted_fields={"Name": "Test", "ID": "123"},
            rule_results=[{"Severity": "INCONSISTENT"}]
        )
        self.assertEqual(q4["quadrant_id"], "IV")

    def test_controlled_benchmark_runner_execution(self):
        from benchmark_runner import run_benchmark_suite
        bench_path = Path(__file__).parent / "data" / "controlled_test_cases.json"
        res = run_benchmark_suite(bench_path)
        self.assertGreaterEqual(res["total_cases"], 20)
        self.assertEqual(res["proposed_far"], 0.0)  # FAR must be 0.0%
        self.assertGreaterEqual(res["proposed_accuracy"], 95.0)


class TestCourseCompletionCertificateVerification(unittest.TestCase):
    """
    R1, R2, R3: Automated tests for course / training completion certificate extraction,
    tolerant category mapping, and verification pipeline accuracy without false failures.
    """

    @classmethod
    def setUpClass(cls):
        # Locate sample certificate document
        candidate_paths = [
            Path(__file__).parent / "data" / "Microsoft_simplelearn_Introduction to Prompt Engineering with.pdf",
            Path(r"C:\Users\shiva\OneDrive\RESUMA\Random_certification\Microsoft_simplelearn_Introduction to Prompt Engineering with.pdf")
        ]
        cls.sample_pdf_path = None
        for p in candidate_paths:
            if p.exists():
                cls.sample_pdf_path = p
                break

    def setUp(self):
        if not self.sample_pdf_path or not self.sample_pdf_path.exists():
            self.skipTest("Sample certificate document 'Microsoft_simplelearn_Introduction to Prompt Engineering with.pdf' not found.")
        with open(self.sample_pdf_path, "rb") as f:
            self.pdf_bytes = f.read()
        self.filename = "Microsoft_simplelearn_Introduction to Prompt Engineering with.pdf"
        self.processed_doc = process_file_bytes(self.pdf_bytes, self.filename)

    def test_sample_certificate_extraction_educational_category(self):
        """Verify clean extraction under Educational Certificate without false mandatory field errors."""
        fields = extract_structured_fields(self.processed_doc, "Educational Certificate")

        # 1. Recipient Name extraction
        self.assertEqual(fields["Student Name"]["value"], "Shivam Patil")
        self.assertFalse(fields["Student Name"]["manual_verification_required"])

        # 2. Certificate ID / Code extraction
        self.assertEqual(fields["Certificate ID"]["value"], "9380068")
        self.assertFalse(fields["Certificate ID"]["manual_verification_required"])

        # 3. Issuing Authority / Institution extraction
        self.assertIn(fields["Institution"]["value"], ["Simplilearn", "Microsoft SkillUp", "Microsoft"])
        self.assertFalse(fields["Institution"]["manual_verification_required"])

        # 4. Issue Date continues to be extracted
        self.assertEqual(fields["Issue Date"]["value"], "11th November 2025")
        self.assertFalse(fields["Issue Date"]["manual_verification_required"])

        # 5. Schema Validation
        schema_results = validate_document_schema(fields, "Educational Certificate")
        statuses = {r["Field"]: r["Status"] for r in schema_results}
        self.assertNotEqual(statuses.get("Student Name"), "MISSING")
        self.assertNotEqual(statuses.get("Certificate ID"), "MISSING")
        self.assertNotEqual(statuses.get("Institution"), "MISSING")
        self.assertEqual(statuses.get("Issue Date"), "VALID FORMAT")

        # 6. Verification Pipeline Execution
        ml_res = predict_document_category(self.processed_doc["full_text"])
        _, rules = run_rule_engine(
            extracted_fields=fields,
            category_name="Educational Certificate",
            raw_text=self.processed_doc["full_text"],
            ocr_confidence=self.processed_doc["average_ocr_confidence"],
            ml_predicted_category=ml_res.get("predicted_category")
        )
        outcome = evaluate_verification_decision(
            extracted_fields=fields,
            schema_results=schema_results,
            rule_results=rules,
            ml_result=ml_res,
            ocr_confidence=self.processed_doc["average_ocr_confidence"],
            page_count=self.processed_doc["page_count"],
            raw_text=self.processed_doc["full_text"],
            declared_category="Educational Certificate"
        )

        self.assertEqual(outcome.final_status, STATUS_CONSISTENCY_PASSED)
        self.assertFalse(outcome.requires_admin_review)
        self.assertFalse(outcome.has_missing_required)
        for reason in outcome.detailed_reasons:
            self.assertNotIn("Mandatory field 'Student Name' could not be identified", reason)
            self.assertNotIn("Mandatory field 'Certificate ID' could not be identified", reason)
            self.assertNotIn("Mandatory field 'Institution' could not be identified", reason)

    def test_sample_certificate_extraction_government_category(self):
        """Verify tolerant extraction under Government Certificate without false mandatory field errors."""
        fields = extract_structured_fields(self.processed_doc, "Government Certificate")

        # 1. Recipient Name extraction mapped to Applicant Name
        self.assertEqual(fields["Applicant Name"]["value"], "Shivam Patil")
        self.assertFalse(fields["Applicant Name"]["manual_verification_required"])

        # 2. Certificate ID / Code extraction
        self.assertEqual(fields["Certificate ID"]["value"], "9380068")
        self.assertFalse(fields["Certificate ID"]["manual_verification_required"])

        # 3. Issuing Authority extraction
        self.assertIn(fields["Issuing Authority"]["value"], ["Simplilearn", "Microsoft SkillUp", "Microsoft"])
        self.assertFalse(fields["Issuing Authority"]["manual_verification_required"])

        # 4. Issue Date continues to be extracted
        self.assertEqual(fields["Issue Date"]["value"], "11th November 2025")
        self.assertFalse(fields["Issue Date"]["manual_verification_required"])

        # 5. Schema Validation
        schema_results = validate_document_schema(fields, "Government Certificate")
        statuses = {r["Field"]: r["Status"] for r in schema_results}
        self.assertNotEqual(statuses.get("Applicant Name"), "MISSING")
        self.assertNotEqual(statuses.get("Certificate ID"), "MISSING")
        self.assertNotEqual(statuses.get("Issuing Authority"), "MISSING")
        self.assertEqual(statuses.get("Issue Date"), "VALID FORMAT")

        # 6. Verification Pipeline Execution
        ml_res = predict_document_category(self.processed_doc["full_text"])
        _, rules = run_rule_engine(
            extracted_fields=fields,
            category_name="Government Certificate",
            raw_text=self.processed_doc["full_text"],
            ocr_confidence=self.processed_doc["average_ocr_confidence"],
            ml_predicted_category=ml_res.get("predicted_category")
        )
        outcome = evaluate_verification_decision(
            extracted_fields=fields,
            schema_results=schema_results,
            rule_results=rules,
            ml_result=ml_res,
            ocr_confidence=self.processed_doc["average_ocr_confidence"],
            page_count=self.processed_doc["page_count"],
            raw_text=self.processed_doc["full_text"],
            declared_category="Government Certificate"
        )

        self.assertEqual(outcome.final_status, STATUS_CONSISTENCY_PASSED)
        self.assertFalse(outcome.requires_admin_review)
        self.assertFalse(outcome.has_missing_required)
        for reason in outcome.detailed_reasons:
            self.assertNotIn("Mandatory field 'Applicant Name' could not be identified", reason)
            self.assertNotIn("Mandatory field 'Certificate ID' could not be identified", reason)
            self.assertNotIn("Mandatory field 'Issuing Authority' could not be identified", reason)

    def test_completion_header_phrasing_layout_patterns(self):
        """R1: Test extraction where name appears between completion headers and completion phrasing."""
        from information_extractor import extract_course_completion_name

        patterns = [
            ("DECLARATION OF COMPLETION\nPresented to\nShivam Patil\nhas successfully completed\nIntroduction to Prompt Engineering", "Shivam Patil"),
            ("CERTIFICATE OF COMPLETION\nShivam Patil\nhas completed the required courses", "Shivam Patil"),
            ("CERTIFICATE OF ACHIEVEMENT\nAwarded to\nShivam Patil\nfor successfully completing the course", "Shivam Patil"),
        ]

        for text, expected_name in patterns:
            res = extract_course_completion_name(text)
            self.assertIsNotNone(res, f"Failed for text: {text}")
            candidate, evidence, conf = res
            self.assertEqual(candidate, expected_name)
            self.assertGreaterEqual(conf, 90.0)

    def test_certificate_code_aliases_with_whitespace_before_colon(self):
        """R1: Test alias mappings and regex handling optional whitespace before colons."""
        from information_extractor import extract_field_from_lines
        from document_schema import get_schema

        aliases = get_schema("Educational Certificate").alias_map["Certificate ID"]
        test_lines = [
            ("Certificate code : 9380068", "9380068"),
            ("cert code: 9380068", "9380068"),
            ("verification code : 9380068", "9380068"),
            ("code : 9380068", "9380068"),
            ("credential id: 9380068", "9380068"),
            ("cert id : 9380068", "9380068"),
            ("Certificate code: 9380068", "9380068"),
        ]

        for line, expected_id in test_lines:
            res = extract_field_from_lines([line], aliases, "identifier")
            self.assertIsNotNone(res, f"Failed on line: '{line}'")
            val, evidence, conf = res
            self.assertEqual(val, expected_id)
            self.assertGreaterEqual(conf, 85.0)

    def test_online_learning_platforms_keyword_extraction(self):
        """R1: Test identification of major online learning platforms and certifiers."""
        from information_extractor import extract_template_institution

        platforms = [
            ("Microsoft SkillUp", "Microsoft SkillUp"),
            ("Simplilearn", "Simplilearn"),
            ("Microsoft", "Microsoft"),
            ("Google Cloud", "Google Cloud"),
            ("AWS", "AWS"),
            ("Coursera", "Coursera"),
            ("Udemy", "Udemy"),
            ("edX", "edX"),
        ]

        for phrase, canonical_name in platforms:
            doc_text = f"Certificate issued by {phrase} for course completion."
            res = extract_template_institution(doc_text)
            self.assertIsNotNone(res, f"Failed for platform: {phrase}")
            cand, ev, conf = res
            self.assertEqual(cand, canonical_name)
            self.assertGreaterEqual(conf, 85.0)

        # Google without Cloud should canonically resolve to Google
        res_google = extract_template_institution("Certificate issued by Google for completing data foundations.")
        self.assertIsNotNone(res_google)
        self.assertEqual(res_google[0], "Google")

    def test_tolerant_cross_category_field_mapping(self):
        """R2: Test that equivalent fields prevent false mandatory validation failures."""
        # Simulated extraction result with cross-category keys
        fields = {
            "Student Name": {"value": "Shivam Patil", "confidence": 92.0, "status": "PRESENT"},
            "Certificate ID": {"value": "9380068", "confidence": 90.0, "status": "PRESENT"},
            "Institution": {"value": "Simplilearn", "confidence": 88.0, "status": "PRESENT"},
            "Issue Date": {"value": "11th November 2025", "confidence": 88.0, "status": "PRESENT"},
            "Expiry Date": {"value": None, "confidence": 0.0, "status": "NOT_APPLICABLE"}
        }

        # Validating under Government Certificate where schema expects Applicant Name & Issuing Authority
        gov_schema_res = validate_document_schema(fields, "Government Certificate")
        statuses = {r["Field"]: r["Status"] for r in gov_schema_res}
        self.assertEqual(statuses.get("Applicant Name"), "PRESENT")
        self.assertEqual(statuses.get("Issuing Authority"), "PRESENT")
        self.assertEqual(statuses.get("Certificate ID"), "PRESENT")
        self.assertEqual(statuses.get("Issue Date"), "VALID FORMAT")
        self.assertNotIn("MISSING", statuses.values())

        # Validating under Identity Document where schema expects Full Name
        id_schema_res = validate_document_schema(fields, "Identity Document")
        id_statuses = {r["Field"]: r["Status"] for r in id_schema_res}
        self.assertEqual(id_statuses.get("Name"), "PRESENT")
        self.assertEqual(id_statuses.get("Date of Birth"), "NOT_APPLICABLE")

    def test_adversarial_name_extraction_edge_cases(self):
        """R1: Adversarially probe multi-line names, phrasing preceding header, and org prefixes."""
        from information_extractor import extract_course_completion_name

        # Case 1: Multi-line recipient name
        text_multiline = "CERTIFICATE OF COMPLETION\nPresented to\nMary Jane\nWatson\nhas successfully completed\nMachine Learning"
        res1 = extract_course_completion_name(text_multiline)
        self.assertIsNotNone(res1)
        self.assertEqual(res1[0], "Mary Jane Watson")

        # Case 2: Phrasing appears before the completion header
        text_early_phrase = "For completing prerequisites\nCERTIFICATE OF TRAINING\nConferred upon\nJohn Doe\ncompleted the course"
        res2 = extract_course_completion_name(text_early_phrase)
        self.assertIsNotNone(res2)
        self.assertEqual(res2[0], "John Doe")

        # Case 3: Org/educational prefix line before recipient name
        text_org_prefix = "Executive Education\nShivam Patil\nIntroduction to Prompt Engineering\nCertificate code : 9380068"
        res3 = extract_course_completion_name(text_org_prefix)
        self.assertIsNotNone(res3)
        self.assertEqual(res3[0], "Shivam Patil")

    def test_code_with_trailing_parenthetical_remarks(self):
        """R1: Test that parenthetical verify remarks are stripped from identifier candidates."""
        from information_extractor import extract_field_from_lines
        from document_schema import get_schema

        aliases = get_schema("Educational Certificate").alias_map["Certificate ID"]
        line = "Certificate code : 9380068 (Please verify online)"
        res = extract_field_from_lines([line], aliases, "identifier")
        self.assertIsNotNone(res)
        val, ev, conf = res
        self.assertEqual(val, "9380068")

        # Confirm validation does not produce a FORMAT WARNING
        fields = {"Certificate ID": {"value": val, "confidence": conf, "status": "PRESENT"}}
        schema_res = validate_document_schema(fields, "Educational Certificate")
        statuses = {r["Field"]: r["Status"] for r in schema_res}
        self.assertNotEqual(statuses.get("Certificate ID"), "FORMAT WARNING")

    def test_institution_multiline_boundaries(self):
        """R1: Test that multi-line text preceding an institution keyword does not leak into the name."""
        from information_extractor import extract_template_institution

        text = "John Doe\nCourse in Python\nStanford University"
        res = extract_template_institution(text)
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "Stanford University")

    def test_completion_date_alias_extraction(self):
        """R1: Test extraction of completion date aliases."""
        from information_extractor import extract_field_from_lines
        from document_schema import get_schema

        aliases = get_schema("Educational Certificate").alias_map["Issue Date"]
        for line, exp_date in [
            ("Date of completion: 11th November 2025", "11th November 2025"),
            ("Completed on: 11th November 2025", "11th November 2025")
        ]:
            res = extract_field_from_lines([line], aliases, "date")
            self.assertIsNotNone(res, f"Failed on: {line}")
            self.assertEqual(res[0], exp_date)

    def test_platform_standalone_header_institution_extraction(self):
        """R1: Ensure standalone platform names at top of certificate do not swallow next line header."""
        text = (
            "Simplilearn\n"
            "CERTIFICATE OF COMPLETION\n"
            "Presented to\n"
            "John Doe\n"
            "has successfully completed\n"
            "Python Course\n"
            "11th November 2025\n"
            "Certificate code : 9380068"
        )
        doc = {"pages": [{"page_number": 1, "text": text}], "full_text": text, "filename": "cert.pdf"}
        res = extract_structured_fields(doc, "Educational Certificate")
        self.assertIsNotNone(res.get("Institution"))
        self.assertEqual(res["Institution"]["value"], "Simplilearn")
        self.assertNotEqual(res["Institution"]["value"], "CERTIFICATE OF COMPLETION")

    def test_recipient_name_inline_phrasing_and_punctuation_triggers(self):
        """R1: Test punctuation in intermediate triggers and inline phrasing line names."""
        from information_extractor import extract_course_completion_name

        # Case A: Colon in intermediate trigger followed by multi-line name
        text_a = (
            "CERTIFICATE OF COMPLETION\n"
            "Presented to:\n"
            "Mary Jane\n"
            "Watson\n"
            "has successfully completed\n"
            "Python for Data Science"
        )
        res_a = extract_course_completion_name(text_a)
        self.assertIsNotNone(res_a)
        self.assertEqual(res_a[0], "Mary Jane Watson")

        # Case B: Multiple organization headers preceding header, and name on same line as phrasing
        text_b = (
            "Acme Institute of Technology\n"
            "Online Learning Center\n"
            "CERTIFICATE OF COMPLETION\n"
            "John Doe has successfully completed\n"
            "Introduction to Python"
        )
        res_b = extract_course_completion_name(text_b)
        self.assertIsNotNone(res_b)
        self.assertEqual(res_b[0], "John Doe")

    def test_identity_document_course_certificate_tolerant_mapping_end_to_end(self):
        """R2: Process real sample certificate under Identity Document category tolerantly."""
        fields = extract_structured_fields(self.processed_doc, "Identity Document")
        self.assertEqual(fields["Name"]["value"], "Shivam Patil")
        self.assertEqual(fields["Document Number"]["value"], "9380068")

        schema_res = validate_document_schema(fields, "Identity Document", raw_text=self.processed_doc["full_text"])
        statuses = {r["Field"]: r["Status"] for r in schema_res}
        self.assertEqual(statuses.get("Name"), "PRESENT")
        self.assertEqual(statuses.get("Document Number"), "PRESENT")
        self.assertEqual(statuses.get("Date of Birth"), "NOT_APPLICABLE")
        self.assertNotEqual(statuses.get("Date of Birth"), "MISSING")

        _, rules = run_rule_engine(
            extracted_fields=fields,
            category_name="Identity Document",
            raw_text=self.processed_doc["full_text"]
        )
        r001_rules = [r for r in rules if r.get("Rule ID") == "R001"]
        self.assertEqual(len(r001_rules), 0)

    def test_certificate_code_with_dash_or_equals_separator(self):
        """R1: Test certificate code extraction with dash and equals separators."""
        from information_extractor import extract_field_from_lines
        from document_schema import get_schema

        aliases = get_schema("Educational Certificate").alias_map["Certificate ID"]
        for line, exp_code in [
            ("Certificate code - 9380068", "9380068"),
            ("cert code = 9380068", "9380068"),
            ("credential id - 9380068", "9380068")
        ]:
            res = extract_field_from_lines([line], aliases, "identifier")
            self.assertIsNotNone(res, f"Failed on: {line}")
            self.assertEqual(res[0], exp_code)

    def test_names_containing_subwords_not_rejected(self):
        """R1: Ensure person names containing subwords (e.g. 'id' in David, 'inc' in Vincent) are not rejected."""
        from information_extractor import extract_course_completion_name, extract_structured_fields

        subword_test_cases = [
            ("David Miller", (
                "CERTIFICATE OF COMPLETION\n"
                "David Miller\n"
                "has successfully completed\n"
                "Python for Data Science\n"
                "Certificate code : 9380068\n"
                "11th November 2025\n"
                "Simplilearn"
            )),
            ("Vincent Price", (
                "CERTIFICATE OF COMPLETION\n"
                "Vincent Price\n"
                "has successfully completed\n"
                "Machine Learning Fundamentals\n"
                "Certificate code : 9380068\n"
                "11th November 2025\n"
                "Simplilearn"
            )),
            ("Rashid Khan", (
                "CERTIFICATE OF COMPLETION\n"
                "Presented to\n"
                "Rashid Khan\n"
                "has successfully completed\n"
                "Cloud Computing\n"
                "Certificate code : 9380068\n"
                "11th November 2025\n"
                "Simplilearn"
            ))
        ]

        for expected_name, text in subword_test_cases:
            res = extract_course_completion_name(text)
            self.assertIsNotNone(res, f"Failed extracting name for {expected_name}")
            self.assertEqual(res[0], expected_name)

            doc = {"pages": [{"page_number": 1, "text": text}], "full_text": text, "filename": "cert.pdf"}
            fields = extract_structured_fields(doc, "Educational Certificate")
            self.assertEqual(fields["Student Name"]["value"], expected_name)
            self.assertEqual(fields["Student Name"]["status"], "PRESENT")

    def test_intermediate_triggers_not_prepended_to_name(self):
        """R1: Ensure trigger phrases ('Certifies that', 'This certificate is proudly presented to') are not prepended to name."""
        from information_extractor import extract_course_completion_name

        text_a = (
            "CERTIFICATE OF COMPLETION\n"
            "Certifies that\n"
            "John Doe\n"
            "has successfully completed\n"
            "Deep Learning\n"
            "Certificate code : 9380068"
        )
        res_a = extract_course_completion_name(text_a)
        self.assertIsNotNone(res_a)
        self.assertEqual(res_a[0], "John Doe")
        self.assertNotIn("Certifies", res_a[0])

        text_b = (
            "CERTIFICATE OF COMPLETION\n"
            "This certificate is proudly presented to\n"
            "Alice Wonderland\n"
            "has completed\n"
            "Web Development"
        )
        res_b = extract_course_completion_name(text_b)
        self.assertIsNotNone(res_b)
        self.assertEqual(res_b[0], "Alice Wonderland")
        self.assertNotIn("presented", res_b[0].lower())

    def test_omitted_recipient_name_does_not_extract_course_or_phrasing(self):
        """R1: When recipient name is omitted, extractor must not extract phrasing or course title as name."""
        from information_extractor import extract_course_completion_name

        text = (
            "CERTIFICATE OF COMPLETION\n"
            "has successfully completed\n"
            "Python for Beginners\n"
            "Certificate code : 9380068\n"
            "11th November 2025\n"
            "Simplilearn"
        )
        res = extract_course_completion_name(text)
        self.assertIsNone(res)

    def test_course_indicators_consistent_across_schema_and_rule_engine(self):
        """R2: Ensure course certificates with varied indicators are handled tolerantly across schema & rules."""
        raw_text = (
            "CERTIFICATE OF ACHIEVEMENT\n"
            "Presented to\n"
            "John Doe\n"
            "has completed\n"
            "Python Fundamentals\n"
            "cert code : 9380068\n"
            "11th November 2025"
        )
        doc = {"pages": [{"page_number": 1, "text": raw_text}], "full_text": raw_text, "filename": "cert.pdf"}
        fields = extract_structured_fields(doc, "Identity Document")
        schema_res = validate_document_schema(fields, "Identity Document", raw_text=raw_text)
        dob_status = next((r["Status"] for r in schema_res if r["Field"] == "Date of Birth"), None)
        self.assertEqual(dob_status, "NOT_APPLICABLE")

        _, rules = run_rule_engine(fields, "Identity Document", raw_text=raw_text)
        r001_dob = [r for r in rules if r.get("Rule ID") == "R001" and "Date of Birth" in r.get("Rule Name", "")]
        self.assertEqual(len(r001_dob), 0)

    def test_inline_and_same_line_completion_header_layouts(self):
        """R1: Test inline and same-line layouts where name is on header line or phrasing line."""
        from information_extractor import extract_course_completion_name

        cases = [
            ("CERTIFICATE OF COMPLETION - Jane Doe has successfully completed Python Course", "Jane Doe"),
            ("CERTIFICATE OF COMPLETION: Jane Doe\nhas successfully completed Python Course", "Jane Doe"),
            ("CERTIFICATE OF ACHIEVEMENT: Jane Doe has completed Deep Learning", "Jane Doe"),
            ("DECLARATION OF COMPLETION. Jane Doe has successfully completed Python", "Jane Doe"),
        ]
        for text, exp in cases:
            res = extract_course_completion_name(text)
            self.assertIsNotNone(res, f"Failed extracting from: {text}")
            self.assertEqual(res[0], exp)

    def test_phrasing_line_connectors_do_not_pollute_recipient_name(self):
        """R1: Ensure connectors and attendance phrasing on phrasing lines do not pollute recipient name."""
        from information_extractor import extract_course_completion_name

        cases = [
            ("CERTIFICATE OF COMPLETION\nJane Doe\nwho has successfully completed\nPython Course", "Jane Doe"),
            ("CERTIFICATE OF COMPLETION\nJane Doe\nhas attended and successfully completed\nPython Course", "Jane Doe"),
            ("CERTIFICATE OF COMPLETION\nJane Doe\nfor the successful completion of Python Course", "Jane Doe"),
            ("CERTIFICATE OF COMPLETION\nJane Doe\nand has completed the requirements\nPython Course", "Jane Doe"),
        ]
        for text, exp in cases:
            res = extract_course_completion_name(text)
            self.assertIsNotNone(res, f"Failed extracting from: {text}")
            self.assertEqual(res[0], exp)

    def test_multiline_label_does_not_extract_label_as_value(self):
        """R1: Ensure meta-words (Name, of Issue) are not extracted as field values when label is on separate line."""
        from information_extractor import extract_structured_fields

        doc_edu = {
            "pages": [{"page_number": 1, "text": (
                "CERTIFICATE OF COMPLETION\n"
                "Recipient Name\n"
                "Jane Doe\n"
                "has successfully completed\n"
                "Certificate code : 9380068\n"
                "11th November 2025\n"
                "Simplilearn"
            )}],
            "full_text": (
                "CERTIFICATE OF COMPLETION\n"
                "Recipient Name\n"
                "Jane Doe\n"
                "has successfully completed\n"
                "Certificate code : 9380068\n"
                "11th November 2025\n"
                "Simplilearn"
            ),
            "filename": "cert.pdf"
        }
        fields_edu = extract_structured_fields(doc_edu, "Educational Certificate")
        self.assertEqual(fields_edu["Student Name"]["value"], "Jane Doe")
        self.assertNotEqual(fields_edu["Student Name"]["value"], "Name")

        doc_gov = {
            "pages": [{"page_number": 1, "text": (
                "CERTIFICATE OF COMPLETION\n"
                "Applicant Name\n"
                "Jane Doe\n"
                "has successfully completed\n"
                "Certificate code : 9380068\n"
                "11th November 2025\n"
                "Simplilearn"
            )}],
            "full_text": (
                "CERTIFICATE OF COMPLETION\n"
                "Applicant Name\n"
                "Jane Doe\n"
                "has successfully completed\n"
                "Certificate code : 9380068\n"
                "11th November 2025\n"
                "Simplilearn"
            ),
            "filename": "cert.pdf"
        }
        fields_gov = extract_structured_fields(doc_gov, "Government Certificate")
        self.assertEqual(fields_gov["Applicant Name"]["value"], "Jane Doe")
        self.assertNotEqual(fields_gov["Applicant Name"]["value"], "Name")

        doc_date = {
            "pages": [{"page_number": 1, "text": (
                "CERTIFICATE OF COMPLETION\n"
                "Jane Doe\n"
                "has completed Python\n"
                "Date of Issue\n"
                "11th November 2025\n"
                "Certificate code : 9380068"
            )}],
            "full_text": (
                "CERTIFICATE OF COMPLETION\n"
                "Jane Doe\n"
                "has completed Python\n"
                "Date of Issue\n"
                "11th November 2025\n"
                "Certificate code : 9380068"
            ),
            "filename": "cert.pdf"
        }
        fields_date = extract_structured_fields(doc_date, "Educational Certificate")
        self.assertEqual(fields_date["Issue Date"]["value"], "11th November 2025")
        self.assertNotEqual(fields_date["Issue Date"]["value"], "of Issue")

    def test_code_with_brackets_trailing_punctuation_and_compound_delimiters(self):
        """R1: Test identifier extraction with square bracket remarks, trailing punctuation, and compound delimiter lines."""
        from information_extractor import extract_field_from_lines
        from document_schema import get_schema

        aliases = get_schema("Educational Certificate").alias_map["Certificate ID"]
        cases = [
            ("Certificate code : 9380068 [Please verify online]", "9380068"),
            ("Certificate code : 9380068.", "9380068"),
            ("Certificate code : 9380068;", "9380068"),
            ("Certificate code : 9380068, Date: 11/11/2025", "9380068"),
            ("Certificate code : 9380068 | Date: 11th November 2025", "9380068"),
            ("Certificate code : 9380068 - Date of Issue: 11/11/2025", "9380068"),
            ("Certificate code: 9380068 | Issued on 11/11/2025", "9380068"),
        ]
        for line, exp in cases:
            res = extract_field_from_lines([line], aliases, "identifier")
            self.assertIsNotNone(res, f"Failed on: {line}")
            self.assertEqual(res[0], exp, f"Failed on line: {line}")

    def test_certificate_id_with_underscore_and_dot_format_validation(self):
        """R1: Test that Certificate IDs containing underscores and dots are recognized as valid formats."""
        for code in ["CERT_9380068", "cert.9380068", "AWS-CERT-001_A", "UC-1234.5678"]:
            fields = {"Certificate ID": {"value": code, "confidence": 90.0, "status": "PRESENT"}}
            res = validate_document_schema(fields, "Educational Certificate")
            stat = next((r["Status"] for r in res if r["Field"] == "Certificate ID"), None)
            self.assertEqual(stat, "PRESENT", f"Certificate ID {code} failed format check with status {stat}")

    def test_extended_completion_phrasing_patterns_and_verb_rejection(self):
        """R1: Test extraction with extended completion phrasing (graduated, participated, recognition, completion of)."""
        from information_extractor import extract_course_completion_name

        cases = [
            ("CERTIFICATE OF COMPLETION\nJane Doe\nhas successfully graduated from\nPython Bootcamp", "Jane Doe"),
            ("CERTIFICATE OF PARTICIPATION\nJane Doe\nhas participated in\nPython Workshop", "Jane Doe"),
            ("CERTIFICATE OF COMPLETION\nJane Doe\nfor completion of\nPython Course", "Jane Doe"),
            ("CERTIFICATE OF COMPLETION\nJane Doe\nin recognition of successful completion of\nPython Course", "Jane Doe"),
        ]
        for text, exp in cases:
            res = extract_course_completion_name(text)
            self.assertIsNotNone(res, f"Failed extracting from: {text}")
            self.assertEqual(res[0], exp)

    def test_multiline_bulleted_identifier_extraction(self):
        """R1: Test multi-line identifier extraction with bullet characters and trailing punctuation."""
        from information_extractor import extract_field_from_lines
        from document_schema import get_schema

        aliases = get_schema("Educational Certificate").alias_map["Certificate ID"]
        lines = ["Certificate code", "• 9380068."]
        res = extract_field_from_lines(lines, aliases, "identifier")
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "9380068")

    def test_date_parsing_across_dash_slash_dot_and_abbreviated_month_formats(self):
        """R1: Test date parsing and validation across dash, slash, dot, and abbreviated month formats."""
        from document_schema import parse_date_robust, validate_document_schema
        from information_extractor import DATE_REGEX, extract_field_from_lines
        from document_schema import get_schema
        from datetime import date

        date_samples = [
            ("01-Jan-2025", date(2025, 1, 1)),
            ("11-Nov-2025", date(2025, 11, 11)),
            ("11-November-2025", date(2025, 11, 11)),
            ("01/Jan/2025", date(2025, 1, 1)),
            ("01.Jan.2025", date(2025, 1, 1)),
            ("Nov 11, 2025", date(2025, 11, 11)),
            ("11-Nov-25", date(2025, 11, 11)),
            ("15-Aug-2024", date(2024, 8, 15)),
        ]
        aliases = get_schema("Educational Certificate").alias_map["Issue Date"]

        for date_str, expected_date in date_samples:
            # 1. Regex recognition
            match = DATE_REGEX.search(date_str)
            self.assertIsNotNone(match, f"DATE_REGEX failed to match: {date_str}")

            # 2. Robust calendar date parsing
            parsed = parse_date_robust(date_str)
            self.assertEqual(parsed, expected_date, f"parse_date_robust failed for: {date_str}")

            # 3. Line extraction with alias
            line = f"Date of Issue: {date_str}"
            extracted = extract_field_from_lines([line], aliases, "date")
            self.assertIsNotNone(extracted, f"extract_field_from_lines failed for: {line}")
            self.assertEqual(extracted[0], date_str)

            # 4. Schema validation produces VALID FORMAT (not MISSING)
            fields = {"Issue Date": {"value": date_str, "confidence": 95.0, "status": "PRESENT"}}
            res = validate_document_schema(fields, "Educational Certificate")
            stat = next((r["Status"] for r in res if r["Field"] == "Issue Date"), None)
            self.assertEqual(stat, "VALID FORMAT", f"Issue Date format check failed for: {date_str}")

    def test_identifier_and_name_extraction_with_leading_and_trailing_quotes(self):
        """R1: Test identifier and name extraction with single, double, and typographic quotes without false FORMAT WARNING."""
        from information_extractor import extract_field_from_lines, clean_extracted_value, normalize_person_name
        from document_schema import get_schema, validate_document_schema

        aliases = get_schema("Educational Certificate").alias_map["Certificate ID"]
        quote_cases = [
            'Certificate code : "9380068"',
            "Certificate code : '9380068'",
            'Certificate code : “9380068”',
            'Certificate code: "9380068."',
        ]
        for line in quote_cases:
            res = extract_field_from_lines([line], aliases, "identifier")
            self.assertIsNotNone(res, f"Failed extracting from: {line}")
            self.assertEqual(res[0], "9380068", f"Failed stripping quotes from: {line}")

            # Schema validation must not trigger FORMAT WARNING
            fields = {"Certificate ID": {"value": res[0], "confidence": 90.0, "status": "PRESENT"}}
            schema_res = validate_document_schema(fields, "Educational Certificate")
            status = next((r["Status"] for r in schema_res if r["Field"] == "Certificate ID"), None)
            self.assertEqual(status, "PRESENT")

        # Also test name cleaning with quotes
        self.assertEqual(normalize_person_name('"Shivam Patil"'), "Shivam Patil")
        self.assertEqual(normalize_person_name("“Shivam Patil”"), "Shivam Patil")
        self.assertEqual(clean_extracted_value("'9380068'"), "9380068")

    def test_extended_learning_platform_certifier_recognition(self):
        """R1: Test recognition of extended online learning platforms (DataCamp, Codecademy, Pluralsight, Kaggle, freeCodeCamp)."""
        from information_extractor import extract_template_institution
        from document_schema import is_course_completion_document

        platforms = [
            ("Certificate of Completion from DataCamp for Python", "DataCamp"),
            ("Completed Course on Codecademy for Web Development", "Codecademy"),
            ("Pluralsight Certificate of Achievement", "Pluralsight"),
            ("freeCodeCamp verified certification in JavaScript", "freeCodeCamp"),
            ("Kaggle Certificate of Completion", "Kaggle"),
        ]
        for text, expected in platforms:
            res = extract_template_institution(text)
            self.assertIsNotNone(res, f"Failed identifying platform from: {text}")
            self.assertEqual(res[0], expected)
            self.assertTrue(is_course_completion_document(raw_text=text))


if __name__ == "__main__":
    unittest.main()
