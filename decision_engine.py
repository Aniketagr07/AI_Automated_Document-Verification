"""
Explainable Decision Engine Module for DocVerify (SIH1709).
Fuses multi-modal verification findings:
- OCR legibility & extraction confidence
- Schema constraints & missing fields
- Forward-chaining production rule triggers
- Supervised ML document category predictions
- Human reviewer interventions and audit overrides

Produces one of the 4 strict final statuses:
1. CONSISTENCY CHECKS PASSED
2. NEEDS MANUAL REVIEW
3. INCONSISTENCIES DETECTED
4. UNABLE TO VERIFY

Enforces AI-First automated finalization for ordinary consistent cases,
and intelligent selective escalation to the admin review queue.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from constants import (
    STATUS_CONSISTENCY_PASSED,
    STATUS_NEEDS_REVIEW,
    STATUS_INCONSISTENCIES,
    STATUS_UNABLE_TO_VERIFY,
    STATUS_PASSED,
    STATUS_REVIEW,
    STATUS_INCONSISTENT,
    STATUS_UNABLE,
    STANDARD_STATUSES,
    AUTHENTICITY_DISCLAIMER
)
from rule_engine import compute_dual_axis_uncertainty_matrix


@dataclass
class DecisionOutcome:
    """Encapsulates the final explainable verification determination."""
    final_status: str
    simple_explanation: str
    detailed_reasons: List[str]
    requires_admin_review: bool
    escalation_reason: Optional[str] = None
    has_contradictions: bool = False
    has_missing_required: bool = False
    is_ocr_degraded: bool = False
    is_category_misaligned: bool = False
    recommended_actions: List[str] = field(default_factory=list)
    authenticity_disclaimer: str = AUTHENTICITY_DISCLAIMER
    extraction_quality_score: float = 85.0
    logical_consistency_score: float = 100.0
    uncertainty_quadrant: str = "QUADRANT I: VERIFIED CONSISTENT"
    uncertainty_advice: str = ""


def evaluate_verification_decision(
    extracted_fields: Dict[str, Any],
    schema_results: List[Dict[str, str]],
    rule_results: List[Dict[str, Any]],
    ml_result: Dict[str, Any],
    ocr_confidence: float = 85.0,
    page_count: int = 1,
    raw_text: str = "",
    declared_category: str = "Educational Certificate",
    admin_review: Optional[Dict[str, Any]] = None
) -> DecisionOutcome:
    """
    Fuses all verification signals into an explainable final verdict.
    Distinguishes extraction uncertainty from deterministic contradiction.
    """
    detailed_reasons: List[str] = []
    recommended_actions: List[str] = []

    matrix = compute_dual_axis_uncertainty_matrix(
        ocr_confidence=ocr_confidence,
        raw_text=raw_text,
        extracted_fields=extracted_fields,
        rule_results=rule_results
    )

    def _with_matrix(outcome: DecisionOutcome) -> DecisionOutcome:
        outcome.extraction_quality_score = matrix["extraction_quality_score"]
        outcome.logical_consistency_score = matrix["logical_consistency_score"]
        outcome.uncertainty_quadrant = matrix["quadrant_name"]
        outcome.uncertainty_advice = matrix["action_advice"]
        return outcome

    # -----------------------------------------------------------------------
    # 1. Check if an Authorized Human Admin Decision Already Overrides
    # -----------------------------------------------------------------------
    if admin_review and admin_review.get("decision"):
        decision = admin_review["decision"]
        reviewer = admin_review.get("reviewer_name", "Authorized Officer")
        notes = admin_review.get("review_notes", "No specific reviewer notes recorded.")

        if decision == "APPROVED":
            return _with_matrix(DecisionOutcome(
                final_status=STATUS_CONSISTENCY_PASSED,
                simple_explanation=(
                    f"Document consistency was manually reviewed and APPROVED by {reviewer}. "
                    f"Notes: {notes}"
                ),
                detailed_reasons=[f"Reviewed and approved by {reviewer}", f"Reviewer notes: {notes}"],
                requires_admin_review=False,
                escalation_reason=None,
                recommended_actions=["Proceed with standard application processing."],
                authenticity_disclaimer=AUTHENTICITY_DISCLAIMER
            ))
        elif decision == "REJECTED":
            return _with_matrix(DecisionOutcome(
                final_status=STATUS_INCONSISTENCIES,
                simple_explanation=(
                    f"Document was REJECTED during manual review by {reviewer}. "
                    f"Reason: {notes}"
                ),
                detailed_reasons=[f"Reviewer rejection by {reviewer}: {notes}"],
                requires_admin_review=False,
                escalation_reason=f"Rejected by reviewer {reviewer}",
                has_contradictions=True,
                recommended_actions=["Notify applicant of documented inconsistencies and request resubmission."],
                authenticity_disclaimer=AUTHENTICITY_DISCLAIMER
            ))
        elif decision == "UNRESOLVED":
            return _with_matrix(DecisionOutcome(
                final_status=STATUS_UNABLE_TO_VERIFY,
                simple_explanation=(
                    f"Document marked UNRESOLVED by {reviewer}. "
                    f"Could not reach conclusive determination. Notes: {notes}"
                ),
                detailed_reasons=[f"Marked unresolved by {reviewer}: {notes}"],
                requires_admin_review=True,
                escalation_reason=f"Unresolved ambiguity flagged by {reviewer}",
                recommended_actions=["Request physical original credential or issuing authority confirmation."],
                authenticity_disclaimer=AUTHENTICITY_DISCLAIMER
            ))
        elif decision == "INFO_REQUESTED":
            return _with_matrix(DecisionOutcome(
                final_status=STATUS_NEEDS_REVIEW,
                simple_explanation=(
                    f"Additional information requested by {reviewer}. "
                    f"Details: {notes}"
                ),
                detailed_reasons=[f"Information request by {reviewer}: {notes}"],
                requires_admin_review=True,
                escalation_reason=f"Pending additional information from applicant.",
                recommended_actions=["Applicant must supply clearer scan or additional identity proof."],
                authenticity_disclaimer=AUTHENTICITY_DISCLAIMER
            ))

    # -----------------------------------------------------------------------
    # 2. Automated Evaluation: Check for Unprocessable / Unsupported Document
    # -----------------------------------------------------------------------
    text_clean = raw_text.strip() if raw_text else ""
    if len(text_clean) < 30:
        return _with_matrix(DecisionOutcome(
            final_status=STATUS_UNABLE_TO_VERIFY,
            simple_explanation=(
                "The system was unable to read sufficient text from this document. "
                "The file may be blank, corrupted, heavily obscured, or in an unsupported format."
            ),
            detailed_reasons=["Document text length is under 30 characters.", "OCR yielded insufficient readable characters."],
            requires_admin_review=True,
            escalation_reason="Document text could not be extracted (illegible or blank scan).",
            is_ocr_degraded=True,
            recommended_actions=[
                "Upload a higher-resolution scan (300 DPI recommended).",
                "Ensure the document is right-side up, well-lit, and uncropped."
            ],
            authenticity_disclaimer=AUTHENTICITY_DISCLAIMER
        ))

    # -----------------------------------------------------------------------
    # 3. Check for Fatal Contradictions (Production Rules with INCONSISTENT severity)
    # -----------------------------------------------------------------------
    inconsistent_rules = [r for r in rule_results if r.get("Severity") == "INCONSISTENT"]
    if inconsistent_rules:
        reasons = [f"[{r.get('Rule ID')}] {r.get('Explanation')}" for r in inconsistent_rules]
        primary_msg = inconsistent_rules[0].get("Explanation")
        rule_name = inconsistent_rules[0].get("Rule Name") or inconsistent_rules[0].get("Rule ID", "Contradiction")
        actions = [r.get("Recommended Action") for r in inconsistent_rules if r.get("Recommended Action")]

        return _with_matrix(DecisionOutcome(
            final_status=STATUS_INCONSISTENCIES,
            simple_explanation=f"Contradictory or invalid information detected: {primary_msg}",
            detailed_reasons=reasons,
            requires_admin_review=True,
            escalation_reason=f"Critical contradiction detected ({rule_name}): {primary_msg}",
            has_contradictions=True,
            recommended_actions=actions or ["Flag document for review and verify issuing authority records."],
            authenticity_disclaimer=AUTHENTICITY_DISCLAIMER
        ))

    # -----------------------------------------------------------------------
    # 4. Check for Missing Mandatory Fields (Schema MISSING)
    # -----------------------------------------------------------------------
    missing_fields = [
        r["Field"] for r in schema_results if r.get("Status") == "MISSING"
    ]
    if missing_fields:
        missing_str = ", ".join(missing_fields)
        return _with_matrix(DecisionOutcome(
            final_status=STATUS_NEEDS_REVIEW,
            simple_explanation=(
                f"Required information is missing from the document: {missing_str}. "
                "Manual review is needed to verify whether this information was omitted or obscured."
            ),
            detailed_reasons=[f"Mandatory field '{f}' could not be identified." for f in missing_fields],
            requires_admin_review=True,
            escalation_reason=f"Missing mandatory fields: {missing_str}",
            has_missing_required=True,
            recommended_actions=[
                f"Verify physical presence of {missing_str} on document.",
                "Reviewer should manually enter or confirm the missing values."
            ],
            authenticity_disclaimer=AUTHENTICITY_DISCLAIMER
        ))

    # -----------------------------------------------------------------------
    # 5. Check for Degraded Scan Quality / Low OCR Confidence
    # -----------------------------------------------------------------------
    if ocr_confidence < 60.0:
        return _with_matrix(DecisionOutcome(
            final_status=STATUS_NEEDS_REVIEW,
            simple_explanation=(
                f"Document text was recognized with low confidence ({ocr_confidence:.1f}%). "
                "Characters may be misread due to blur, noise, or faint printing."
            ),
            detailed_reasons=[f"Average OCR confidence is {ocr_confidence:.1f}%, below the 60.0% threshold."],
            requires_admin_review=True,
            escalation_reason=f"Degraded scan clarity (OCR confidence {ocr_confidence:.1f}%).",
            is_ocr_degraded=True,
            recommended_actions=[
                "Inspect visually against original document.",
                "Verify high-risk characters (0 vs O, 8 vs B, 1 vs I)."
            ],
            authenticity_disclaimer=AUTHENTICITY_DISCLAIMER
        ))

    # -----------------------------------------------------------------------
    # 6. Check for Category Misalignment (ML vs User Declaration)
    # -----------------------------------------------------------------------
    predicted_cat = ml_result.get("predicted_category")
    ml_confidence = ml_result.get("confidence", 0.0)

    # Escalate if ML strongly identifies a mismatch (e.g. user uploaded Receipt/Invoice for an Educational Certificate)
    if (
        predicted_cat
        and predicted_cat != "UNKNOWN"
        and predicted_cat != declared_category
        and ml_confidence >= 65.0
    ):
        return _with_matrix(DecisionOutcome(
            final_status=STATUS_NEEDS_REVIEW,
            simple_explanation=(
                f"The document layout matches '{predicted_cat}' rather than the declared category '{declared_category}'. "
                "Please verify if the correct category was selected."
            ),
            detailed_reasons=[
                f"ML model categorized document as '{predicted_cat}' with {ml_confidence:.1f}% confidence.",
                f"Declared category was '{declared_category}'."
            ],
            requires_admin_review=True,
            escalation_reason=f"Category mismatch: Declared '{declared_category}' vs Detected '{predicted_cat}' ({ml_confidence:.1f}%).",
            is_category_misaligned=True,
            recommended_actions=[
                "Confirm if applicant selected the wrong category at upload.",
                "Re-file document under the appropriate category."
            ],
            authenticity_disclaimer=AUTHENTICITY_DISCLAIMER
        ))

    # -----------------------------------------------------------------------
    # 7. Check for Non-Fatal Review Warnings (R002, R008, R009)
    # -----------------------------------------------------------------------
    review_rules = [r for r in rule_results if r.get("Severity") == "NEEDS MANUAL REVIEW"]
    if review_rules:
        reasons = [f"[{r.get('Rule ID')}] {r.get('Explanation')}" for r in review_rules]
        return _with_matrix(DecisionOutcome(
            final_status=STATUS_NEEDS_REVIEW,
            simple_explanation=(
                "The document passed primary checks but contains formatting variations or unconfirmed items. "
                "Queued for brief officer verification."
            ),
            detailed_reasons=reasons,
            requires_admin_review=True,
            escalation_reason=f"Unresolved formatting or syntax flags ({len(review_rules)} items).",
            recommended_actions=["Officer should review flagged fields and approve if acceptable."],
            authenticity_disclaimer=AUTHENTICITY_DISCLAIMER
        ))

    # -----------------------------------------------------------------------
    # 8. All Checks Satisfied -> Clean Automatic Approval!
    # -----------------------------------------------------------------------
    return _with_matrix(DecisionOutcome(
        final_status=STATUS_CONSISTENCY_PASSED,
        simple_explanation=(
            "All automated consistency checks passed successfully. "
            "Required fields are present, valid, and chronologically coherent."
        ),
        detailed_reasons=[
            "All mandatory schema fields extracted with acceptable confidence.",
            "Dates and chronological relationships are consistent.",
            "Identifier structures meet category syntax requirements.",
            "No logical contradictions or expiration issues detected."
        ],
        requires_admin_review=False,
        escalation_reason=None,
        recommended_actions=[
            "Document is internally consistent. Ready for downstream application processing."
        ],
        authenticity_disclaimer=AUTHENTICITY_DISCLAIMER
    ))
