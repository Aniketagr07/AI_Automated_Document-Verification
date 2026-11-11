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
from document_schema import get_schema, parse_date_robust, validate_document_schema


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

    # Helper to get field value safely (handles flat dict or nested dict)
    def get_val(f_name: str) -> Optional[str]:
        field_meta = extracted_fields.get(f_name, {})
        if isinstance(field_meta, dict):
            val = field_meta.get("value")
        else:
            val = field_meta
        return str(val).strip() if val is not None and str(val).strip() else None

    def get_evidence(f_name: str) -> str:
        field_meta = extracted_fields.get(f_name, {})
        if isinstance(field_meta, dict):
            return str(field_meta.get("evidence", "No evidence recorded."))
        return f"Direct value provided: '{field_meta}'"

    # -----------------------------------------------------------------------
    # RULE R001: Mandatory Field Extraction Check
    # -----------------------------------------------------------------------
    for field_name, constraint in schema.fields.items():
        val = get_val(field_name)
        if constraint.required and not val:
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
        if len(matched_keywords) == 0:
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
