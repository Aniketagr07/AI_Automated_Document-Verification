"""
Review Manager Module for Human-in-the-Loop Document Verification (SIH1709).
Coordinates manual corrections, reviewer approvals, rejection reasons,
audit trail maintenance, and human oversight controls.
"""

from typing import Dict, List, Any, Optional
import database


def submit_field_correction(
    document_id: int,
    field_name: str,
    original_value: str,
    corrected_value: str,
    reviewer_name: str = "Reviewer"
) -> bool:
    """
    Records a manual correction to an extracted field.
    Ensures human corrections are saved to audit history and database.
    """
    if not corrected_value or corrected_value.strip() == (original_value or "").strip():
        return False

    return database.update_field_value(
        document_id=document_id,
        field_name=field_name,
        new_value=corrected_value.strip(),
        reviewer_name=reviewer_name
    )


def submit_review_decision(
    document_id: int,
    decision: str,
    reviewer_name: str,
    review_notes: str = "",
    admin_user_id: int = 1,
    field_corrections: Optional[Dict[str, str]] = None
) -> bool:
    """
    Records an official administrative review verdict:
    - 'APPROVED' (Verified consistent and acceptable)
    - 'REJECTED' (Irreconcilable inconsistency or non-authentic)
    - 'INFO_REQUESTED' (Requires applicant to provide additional document)
    - 'UNRESOLVED' (Could not reach conclusive determination)
    - 'FLAGGED' (Backward compatibility alias for INFO_REQUESTED)
    """
    if decision == "FLAGGED":
        decision = "INFO_REQUESTED"

    valid_decisions = ["APPROVED", "REJECTED", "INFO_REQUESTED", "UNRESOLVED"]
    if decision not in valid_decisions:
        raise ValueError(f"Invalid decision: '{decision}'. Must be one of {valid_decisions}")

    return database.record_admin_review_decision(
        document_id=document_id,
        admin_user_id=admin_user_id,
        admin_name=reviewer_name if reviewer_name else "Verification Officer",
        decision=decision,
        notes=review_notes.strip(),
        field_corrections=field_corrections
    )


def get_audit_trail(document_id: int) -> List[Dict[str, Any]]:
    """Retrieves full historical audit trail for a given document."""
    doc_details = database.get_document_details(document_id, is_admin=True)
    if not doc_details:
        return []
    return doc_details.get("audit", [])


def purge_document_data(document_id: int, actor: str = "USER", user_id: Optional[int] = None) -> bool:
    """Invokes secure GDPR/privacy hard-delete for a document."""
    return database.delete_document_record(document_id, actor=actor, user_id=user_id)
