"""
Production Rule Engine Module (SIH1709).
Implements an explainable forward-chaining verification system:
- Validates field presence, temporal ordering, and calendar validity
- Detects cross-field chronological contradictions
- Checks structural identifier syntax and domain attestation keywords
- Provides Rule ID, Condition, Severity, Explanation, Supporting Evidence, and Recommended Action
- Computes overall status: 'CONSISTENT', 'NEEDS MANUAL REVIEW', or 'INCONSISTENT'
"""

from datetime import date
from typing import Dict, List, Tuple, Any, Optional
import re
from document_schema import (
    get_schema, parse_date_robust, validate_document_schema,
    FIELD_EQUIVALENTS, is_course_completion_document
)
from verhoeff import validate_verhoeff, validate_pan_structure


def parse_date(value: Any) -> Optional[date]:
    """Parses date string across diverse formats. Backwards compatibility helper."""
    return parse_date_robust(str(value) if value else None)


def run_rule_engine(
    extracted_fields: Dict[str, Any],
    category_name: str = "Educational Certificate",
    raw_text: str = "",
    ocr_confidence: float = 85.0,
    ml_predicted_category: Optional[str] = None
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Executes production rules over extracted structured document fields and text.
    Accepts both rich nested dictionaries {field: {"value": ...}} and flat {field: value}.
    Returns:
        verdict: 'CONSISTENT' | 'NEEDS MANUAL REVIEW' | 'INCONSISTENT'
        triggered_rules: List of structured rule event records
    """
    today = date.today()
    schema = get_schema(category_name)
    triggered_rules: List[Dict[str, Any]] = []

    # Helper to get field value safely (handles flat dict, nested dict, or equivalent fields)
    def get_val(f_name: str) -> Optional[str]:
        field_meta = extracted_fields.get(f_name, {})
        if isinstance(field_meta, dict):
            val = field_meta.get("value")
        else:
            val = field_meta
        if val is not None and str(val).strip():
            return str(val).strip()
        for eq in FIELD_EQUIVALENTS.get(f_name, []):
            eq_meta = extracted_fields.get(eq, {})
            eq_val = eq_meta.get("value") if isinstance(eq_meta, dict) else eq_meta
            if eq_val is not None and str(eq_val).strip():
                return str(eq_val).strip()
        return None

    def get_evidence(f_name: str) -> str:
        field_meta = extracted_fields.get(f_name, {})
        if isinstance(field_meta, dict) and field_meta.get("evidence"):
            return str(field_meta.get("evidence"))
        for eq in FIELD_EQUIVALENTS.get(f_name, []):
            eq_meta = extracted_fields.get(eq, {})
            if isinstance(eq_meta, dict) and eq_meta.get("evidence"):
                return str(eq_meta.get("evidence"))
        if isinstance(field_meta, dict):
            return str(field_meta.get("evidence", "No evidence recorded."))
        return f"Direct value provided: '{field_meta}'"

    is_course_cert = is_course_completion_document(extracted_fields, raw_text)

    # -----------------------------------------------------------------------
    # RULE R001: Mandatory Field Extraction Check
    # -----------------------------------------------------------------------
    for field_name, constraint in schema.fields.items():
        val = get_val(field_name)
        is_participation = bool(re.search(r"(?i)\b(?:participation|participated|attendance|attended|workshop|webinar)\b", raw_text))
        if constraint.required and not val:
            if is_course_cert and is_participation and field_name in ["Certificate ID", "Document Number"]:
                continue
            if is_course_cert and field_name not in [
                "Student Name", "Applicant Name", "Name", "Full Name", "Certificate ID", "Document Number",
                "Institution", "Issuing Authority", "Issue Date"
            ]:
                continue
            triggered_rules.append({
                "Rule ID": "R001",
                "Rule Name": f"Mandatory Field Check: {field_name}",
                "Condition": f"Field '{field_name}' is REQUIRED but value is NULL or empty",
                "Severity": "NEEDS MANUAL REVIEW",
                "Explanation": (
                    f"Mandatory field '{field_name}' could not be automatically extracted from the document. "
                    "This may be caused by non-standard layout, poor scan clarity, or an incomplete document."
                ),
                "Evidence": get_evidence(field_name),
                "Recommended Action": f"Inspect document visually and manually enter or confirm '{field_name}'."
            })

    # -----------------------------------------------------------------------
    # RULE R002: Calendar Date Parsing & Format Integrity
    # -----------------------------------------------------------------------
    for field_name, constraint in schema.fields.items():
        if constraint.data_type == "date":
            val = get_val(field_name)
            if val:
                parsed = parse_date_robust(val)
                if not parsed:
                    triggered_rules.append({
                        "Rule ID": "R002",
                        "Rule Name": f"Date Format Validity: {field_name}",
                        "Condition": f"'{val}' could not be parsed to a valid Gregorian calendar date",
                        "Severity": "NEEDS MANUAL REVIEW",
                        "Explanation": (
                            f"The extracted string '{val}' for '{field_name}' is not in a recognized date format. "
                            "It may contain OCR noise or a malformed date."
                        ),
                        "Evidence": get_evidence(field_name),
                        "Recommended Action": "Verify the original document date string and reformat if necessary."
                    })

    # -----------------------------------------------------------------------
    # RULE R003: Date of Birth Cannot Be In The Future
    # -----------------------------------------------------------------------
    dob_val = get_val("Date of Birth")
    dob_parsed = parse_date_robust(dob_val) if dob_val else None
    if dob_parsed and dob_parsed > today:
        triggered_rules.append({
            "Rule ID": "R003",
            "Rule Name": "Future Date of Birth Contradiction",
            "Condition": f"Date of Birth ({dob_parsed}) > Current Date ({today})",
            "Severity": "INCONSISTENT",
            "Explanation": (
                f"The extracted Date of Birth ({dob_parsed.strftime('%d-%b-%Y')}) is in the future. "
                "This indicates an irreconcilable data conflict or falsification."
            ),
            "Evidence": f"Extracted DOB: '{dob_val}'. Evidence: {get_evidence('Date of Birth')}",
            "Recommended Action": "Reject document as inconsistent. Demand official birth certificate verification."
        })

    # -----------------------------------------------------------------------
    # RULE R004: Document Issue Date Future Check
    # -----------------------------------------------------------------------
    issue_val = get_val("Issue Date")
    issue_parsed = parse_date_robust(issue_val) if issue_val else None
    if issue_parsed and issue_parsed > today:
        triggered_rules.append({
            "Rule ID": "R004",
            "Rule Name": "Future Issue Date Contradiction",
            "Condition": f"Issue Date ({issue_parsed}) > Current Date ({today})",
            "Severity": "INCONSISTENT",
            "Explanation": (
                f"Document Issue Date ({issue_parsed.strftime('%d-%b-%Y')}) is post-dated beyond current calendar date. "
                "Official credentials cannot be legally issued in the future."
            ),
            "Evidence": f"Extracted Issue Date: '{issue_val}'. Evidence: {get_evidence('Issue Date')}",
            "Recommended Action": "Flag document for suspected post-dating or clerical date entry error."
        })

    # -----------------------------------------------------------------------
    # RULE R005: Expiry Date Chronology (Expiry must be >= Issue Date)
    # -----------------------------------------------------------------------
    expiry_val = get_val("Expiry Date")
    expiry_parsed = parse_date_robust(expiry_val) if expiry_val else None
    if issue_parsed and expiry_parsed and expiry_parsed < issue_parsed:
        triggered_rules.append({
            "Rule ID": "R005",
            "Rule Name": "Temporal Inversion: Expiry Precedes Issue",
            "Condition": f"Expiry Date ({expiry_parsed}) < Issue Date ({issue_parsed})",
            "Severity": "INCONSISTENT",
            "Explanation": (
                f"Document expiry date ({expiry_parsed.strftime('%d-%b-%Y')}) occurs before its issue date "
                f"({issue_parsed.strftime('%d-%b-%Y')}). This is a fatal temporal contradiction."
            ),
            "Evidence": f"Issue: '{issue_val}', Expiry: '{expiry_val}'",
            "Recommended Action": "Reject document consistency. Re-check issuer records for inverted dates."
        })

    # -----------------------------------------------------------------------
    # RULE R006: Document Expiration Check
    # -----------------------------------------------------------------------
    if expiry_parsed and expiry_parsed < today:
        # If it's expired but not temporally inverted
        if not (issue_parsed and expiry_parsed < issue_parsed):
            triggered_rules.append({
                "Rule ID": "R006",
                "Rule Name": "Document Expired Status",
                "Condition": f"Expiry Date ({expiry_parsed}) < Current Date ({today})",
                "Severity": "INCONSISTENT",
                "Explanation": (
                    f"The document expired on {expiry_parsed.strftime('%d-%b-%Y')}. "
                    "Expired documents are legally invalid for current official verification."
                ),
                "Evidence": f"Extracted Expiry Date: '{expiry_val}'",
                "Recommended Action": "Request an up-to-date, renewed credential from the applicant."
            })

    # -----------------------------------------------------------------------
    # RULE R007: Age at Issuance Consistency (DOB vs Issue Date)
    # -----------------------------------------------------------------------
    if dob_parsed and issue_parsed:
        if issue_parsed < dob_parsed:
            triggered_rules.append({
                "Rule ID": "R007",
                "Rule Name": "Chronological Inversion: Issued Before Birth",
                "Condition": f"Issue Date ({issue_parsed}) < Date of Birth ({dob_parsed})",
                "Severity": "INCONSISTENT",
                "Explanation": (
                    f"Document issue date ({issue_parsed.strftime('%d-%b-%Y')}) precedes the holder's "
                    f"date of birth ({dob_parsed.strftime('%d-%b-%Y')})."
                ),
                "Evidence": f"DOB: '{dob_val}', Issue: '{issue_val}'",
                "Recommended Action": "Immediate rejection. Check for identity impersonation or severe data corruption."
            })

    # -----------------------------------------------------------------------
    # RULE R008: Identifier Structural Format Validation
    # -----------------------------------------------------------------------
    id_field_name = "Certificate ID" if "Certificate ID" in schema.fields else "Document Number"
    id_val = get_val(id_field_name)
    if id_val and len(id_val) < 3:
        triggered_rules.append({
            "Rule ID": "R008",
            "Rule Name": "Identifier Truncation Warning",
            "Condition": f"Length of '{id_field_name}' ({len(id_val)}) < minimum 3 characters",
            "Severity": "NEEDS MANUAL REVIEW",
            "Explanation": f"The extracted identifier '{id_val}' appears truncated or abnormally short.",
            "Evidence": f"Value: '{id_val}', Evidence: {get_evidence(id_field_name)}",
            "Recommended Action": "Manually verify the full registration / certificate number on the physical document."
        })

    # -----------------------------------------------------------------------
    # RULE R009: Category Domain Attestation Keywords
    # -----------------------------------------------------------------------
    if raw_text and schema.required_keywords:
        text_lower = raw_text.lower()
        matched_keywords = [kw for kw in schema.required_keywords if kw in text_lower]
        course_kws = [
            "certificate", "completion", "achievement", "completed", "course",
            "training", "skillup", "prompt", "engineering", "copilot",
            "simplilearn", "microsoft", "google", "aws", "coursera", "udemy", "edx"
        ]
        is_course_doc = is_course_cert or any(kw in text_lower for kw in course_kws)
        if len(matched_keywords) == 0 and not is_course_doc:
            triggered_rules.append({
                "Rule ID": "R009",
                "Rule Name": "Missing Domain Attestation Phrases",
                "Condition": "None of the category-specific attestation keywords found in document text",
                "Severity": "NEEDS MANUAL REVIEW",
                "Explanation": (
                    f"The document text contains none of the standard legal/institutional attestation words "
                    f"expected for '{category_name}' (e.g. {', '.join(schema.required_keywords[:4])})."
                ),
                "Evidence": f"Document text length: {len(raw_text)} chars. Matched keywords: None.",
                "Recommended Action": "Verify whether the correct document category was selected at upload."
            })

    # -----------------------------------------------------------------------
    # RULE R010: OCR Confidence & Scan Legibility
    # -----------------------------------------------------------------------
    if ocr_confidence < 60.0:
        triggered_rules.append({
            "Rule ID": "R010",
            "Rule Name": "Degraded Scan OCR Quality Warning",
            "Condition": f"Average OCR confidence ({ocr_confidence:.1f}%) < threshold 60.0%",
            "Severity": "NEEDS MANUAL REVIEW",
            "Explanation": (
                f"Document text was extracted with low OCR confidence ({ocr_confidence:.1f}%). "
                "Field extraction errors or character misrecognitions (e.g., 8 vs B, 0 vs O) may have occurred."
            ),
            "Evidence": f"OCR Confidence Score: {ocr_confidence:.1f}%",
            "Recommended Action": "Review extracted values carefully against visual document preview."
        })

    # -----------------------------------------------------------------------
    # RULE R011: Document Category Misalignment (ML vs Declared)
    # -----------------------------------------------------------------------
    if ml_predicted_category and ml_predicted_category != "UNKNOWN":
        if ml_predicted_category != category_name:
            # Course/training certificates uploaded under educational or government categories
            # should not trigger false category discrepancy warnings
            if not (is_course_cert and category_name in ["Educational Certificate", "Government Certificate"]):
                triggered_rules.append({
                    "Rule ID": "R011",
                    "Rule Name": "Category Classification Discrepancy",
                    "Condition": f"ML Predicted Category ('{ml_predicted_category}') != Declared Category ('{category_name}')",
                    "Severity": "NEEDS MANUAL REVIEW",
                    "Explanation": (
                        f"The machine learning document classifier predicted this document as '{ml_predicted_category}', "
                        f"whereas the user declared it as '{category_name}'. The document layout may differ from standard."
                    ),
                    "Evidence": f"ML Predicted: '{ml_predicted_category}' vs Declared: '{category_name}'",
                    "Recommended Action": "Confirm whether the document was filed under the appropriate category."
                })

    # -----------------------------------------------------------------------
    # RULE R013: Aadhaar Verhoeff Dihedral (D5) Checksum Validation
    # -----------------------------------------------------------------------
    id_val_raw = get_val("Document Number") or get_val("Certificate ID") or ""
    all_text_to_check = f"{id_val_raw} {raw_text}"
    digits_only = re.sub(r"\D", "", id_val_raw)

    is_aadhaar = (
        "aadhaar" in raw_text.lower() or
        "uidai" in raw_text.lower() or
        "unique identification" in raw_text.lower() or
        (category_name == "Identity Document" and not is_course_cert)
    )

    if is_aadhaar and len(digits_only) == 12:
        if not validate_verhoeff(digits_only):
            triggered_rules.append({
                "Rule ID": "R013",
                "Rule Name": "Aadhaar Verhoeff Checksum Failure",
                "Condition": f"Aadhaar UID '{digits_only}' fails mathematical Verhoeff check digit validation",
                "Severity": "INCONSISTENT",
                "Explanation": (
                    f"The 12-digit Aadhaar number '{digits_only}' fails the official UIDAI Verhoeff dihedral checksum. "
                    "This indicates an algorithmically fabricated, modified, or mistyped identity number."
                ),
                "Evidence": f"Extracted UID: '{id_val_raw}'. Checksum validation: FAILED.",
                "Recommended Action": "Reject document as mathematically inconsistent. Request authentic Aadhaar card."
            })

    # -----------------------------------------------------------------------
    # RULE R014: Permanent Account Number (PAN) Structural & Surname Coherence
    # -----------------------------------------------------------------------
    clean_id_no_space = re.sub(r"\s+", "", id_val_raw).upper()
    pan_candidate = None
    if re.fullmatch(r"^[A-Z]{5}[0-9]{4}[A-Z]$", clean_id_no_space):
        pan_candidate = clean_id_no_space
    else:
        pan_match = re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", raw_text)
        if pan_match:
            pan_candidate = pan_match.group(0)

    if pan_candidate:
        holder_name = get_val("Name") or get_val("Applicant Name") or get_val("Student Name") or ""
        is_pan_valid, pan_msg = validate_pan_structure(pan_candidate, holder_name)
        if not is_pan_valid:
            triggered_rules.append({
                "Rule ID": "R014",
                "Rule Name": "PAN Holder Identity Coherence Inconsistency",
                "Condition": f"PAN '{pan_candidate}' failed validation: {pan_msg}",
                "Severity": "INCONSISTENT",
                "Explanation": (
                    f"PAN validation anomaly: {pan_msg}. In official PAN formatting, the 4th letter designates entity "
                    "type (P for Person), and the 5th letter must match the cardholder's surname initial."
                ),
                "Evidence": f"PAN: '{pan_candidate}', Cardholder: '{holder_name}'. {pan_msg}",
                "Recommended Action": "Demand original PAN card verification; cross-reference Income Tax department records."
            })

    # -----------------------------------------------------------------------
    # RULE R015: Academic Horizon & Course Duration Consistency
    # -----------------------------------------------------------------------
    if category_name == "Educational Certificate" and issue_parsed:
        # Check for enrollment/matriculation year inside certificate ID or text
        # e.g., IITB/2021/CS/1089 or Roll No: 2020-BTECH-042
        cert_id_val = get_val("Certificate ID") or ""
        year_matches = re.findall(r"\b(20[0-2][0-9]|19[8-9][0-9])\b", f"{cert_id_val} {raw_text}")
        if year_matches:
            found_years = [int(y) for y in year_matches]
            earliest_year = min(found_years)
            issue_year = issue_parsed.year
            # If enrollment year is indicated and is equal to or after issue year
            if "bachelor" in raw_text.lower() or "b.tech" in raw_text.lower() or "degree" in raw_text.lower():
                if issue_year < earliest_year:
                    triggered_rules.append({
                        "Rule ID": "R015",
                        "Rule Name": "Academic Horizon Inconsistency: Award Precedes Enrollment",
                        "Condition": f"Degree issue year ({issue_year}) < Enrollment/Batch Year ({earliest_year})",
                        "Severity": "INCONSISTENT",
                        "Explanation": (
                            f"Degree confers in year {issue_year}, which precedes the recorded academic batch or "
                            f"enrollment year {earliest_year}. An official degree cannot be completed before admission."
                        ),
                        "Evidence": f"Issue Date Year: {issue_year}, Batch Year detected: {earliest_year}",
                        "Recommended Action": "Flag degree record for verification with registrar's convocation rolls."
                    })

    # -----------------------------------------------------------------------
    # Compute Final System Verdict
    # -----------------------------------------------------------------------
    has_inconsistent = any(r["Severity"] == "INCONSISTENT" for r in triggered_rules)
    has_manual_review = any(r["Severity"] == "NEEDS MANUAL REVIEW" for r in triggered_rules)

    if has_inconsistent:
        verdict = "INCONSISTENT"
    elif has_manual_review:
        verdict = "NEEDS MANUAL REVIEW"
    else:
        verdict = "CONSISTENT"
        triggered_rules.append({
            "Rule ID": "R012",
            "Rule Name": "All Consistency Production Rules Satisfied",
            "Condition": "All required fields present, dates chronologically consistent, formats valid",
            "Severity": "PASS",
            "Explanation": "No contradictory dates, missing mandatory values, or format anomalies detected.",
            "Evidence": "Complete validation pass across all schema constraints.",
            "Recommended Action": "Document is internally consistent. Proceed to standard workflow verification."
        })

    return verdict, triggered_rules


def compute_dual_axis_uncertainty_matrix(
    ocr_confidence: float,
    raw_text: str,
    extracted_fields: Dict[str, Any],
    rule_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes coordinates for the Dual-Axis Uncertainty vs Inconsistency Matrix.
    Distinguishes extraction noise (scan blur/lighting) from genuine fraud/contradictions.
    Returns:
        {
            "extraction_quality_score": float (0-100),
            "logical_consistency_score": float (0-100),
            "quadrant_id": "I" | "II" | "III" | "IV",
            "quadrant_name": str,
            "quadrant_description": str,
            "action_advice": str
        }
    """
    # 1. Calculate Extraction Quality Score (X-Axis: 0 to 100)
    ocr_factor = min(100.0, max(0.0, float(ocr_confidence)))
    text_len = len(raw_text.strip()) if raw_text else 0
    text_factor = min(100.0, (text_len / 250.0) * 100.0)

    # Check non-empty extracted fields
    if extracted_fields:
        extracted_count = sum(
            1 for v in extracted_fields.values()
            if (v.get("value") if isinstance(v, dict) else v) is not None
        )
        field_factor = (extracted_count / len(extracted_fields)) * 100.0
    else:
        field_factor = 50.0

    extraction_quality = round((ocr_factor * 0.50) + (text_factor * 0.25) + (field_factor * 0.25), 1)

    # 2. Calculate Logical Consistency Score (Y-Axis: 0 to 100)
    consistency_score = 100.0
    for r in rule_results:
        sev = r.get("Severity", "")
        if sev == "INCONSISTENT":
            consistency_score -= 50.0
        elif sev == "NEEDS MANUAL REVIEW":
            consistency_score -= 15.0

    logical_consistency = round(max(0.0, min(100.0, consistency_score)), 1)

    # 3. Determine Quadrant
    if extraction_quality >= 60.0 and logical_consistency >= 70.0:
        quadrant_id = "I"
        quadrant_name = "QUADRANT I: VERIFIED CONSISTENT"
        desc = "High extraction confidence and consistent logical/chronological relationships."
        advice = "Auto-finalized. Safe for downstream administrative processing."
    elif extraction_quality < 60.0 and logical_consistency >= 70.0:
        quadrant_id = "II"
        quadrant_name = "QUADRANT II: EXTRACTION NOISE (Scan Ambiguity)"
        desc = "Logical constraints pass, but OCR legibility or document clarity is degraded."
        advice = "Request clean high-resolution scan or prompt officer for visual field verification."
    elif extraction_quality < 60.0 and logical_consistency < 70.0:
        quadrant_id = "III"
        quadrant_name = "QUADRANT III: UNABLE TO VERIFY"
        desc = "Severe document degradation combined with missing or unresolvable fields."
        advice = "Reject automated parsing; demand physical original credential submission."
    else:
        quadrant_id = "IV"
        quadrant_name = "QUADRANT IV: GENUINE CONTRADICTION (Suspected Fraud)"
        desc = "High extraction clarity, but explicit logical, temporal, or checksum violations detected."
        advice = "Escalate to administrative investigation; flag irreconcilable data conflict."

    return {
        "extraction_quality_score": extraction_quality,
        "logical_consistency_score": logical_consistency,
        "quadrant_id": quadrant_id,
        "quadrant_name": quadrant_name,
        "quadrant_description": desc,
        "action_advice": advice
    }

