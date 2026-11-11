"""
PDF Report Generator Module (SIH1709).
Generates official, professional verification and consistency review PDF reports
using PyMuPDF (fitz) with clean layout, typography, status badges, evidence tables,
and the required legal authenticity disclaimers.
Supports both user-facing verification certificates and detailed admin technical audit reports.
Runs locally without third-party external paid APIs.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import pymupdf

from constants import (
    STATUS_CONSISTENCY_PASSED,
    STATUS_NEEDS_REVIEW,
    STATUS_INCONSISTENCIES,
    STATUS_UNABLE_TO_VERIFY,
    STATUS_PASSED,
    STATUS_REVIEW,
    STATUS_INCONSISTENT,
    STATUS_UNABLE,
    STANDARD_STATUSES
)


def generate_verification_report_pdf(
    document_info: Optional[Dict[str, Any]] = None,
    extracted_fields: Optional[Dict[str, Any]] = None,
    schema_results: Optional[List[Dict[str, str]]] = None,
    rule_results: Optional[List[Dict[str, Any]]] = None,
    ml_results: Optional[Dict[str, Any]] = None,
    overall_status: Optional[str] = None,
    reviewer_notes: Optional[str] = None,
    report_reference: Optional[str] = None,
    is_admin_report: bool = False,
    detected_issues: Optional[List[str]] = None,
    recommended_actions: Optional[List[str]] = None
) -> bytes:
    """
    Constructs a structured, professional PDF verification report.
    Returns binary PDF bytes. Robust against missing fields or empty data.
    """
    doc_info = document_info or {}
    fields = extracted_fields or {}
    schema_res = schema_results or []
    rules = rule_results or []
    ml_res = ml_results or {}
    status_str = (overall_status or STATUS_NEEDS_REVIEW).strip()
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # Standard A4 (points)

    margin_x = 40
    curr_y = 35

    # Palette
    DARK_BLUE = (0.08, 0.22, 0.45)
    LIGHT_BG = (0.96, 0.97, 0.99)
    TEXT_MAIN = (0.12, 0.15, 0.20)
    TEXT_MUTED = (0.40, 0.45, 0.52)
    BORDER_COLOR = (0.82, 0.85, 0.90)

    # Status color mapping
    normalized_status = status_str.upper().replace("VALID", STATUS_PASSED).replace("CONSISTENT", STATUS_PASSED)
    if "PASS" in normalized_status:
        STATUS_COLOR = (0.10, 0.55, 0.30)  # Green
        STATUS_BG = (0.90, 0.97, 0.92)
        display_status = STATUS_PASSED
    elif "REVIEW" in normalized_status:
        STATUS_COLOR = (0.80, 0.50, 0.05)  # Amber
        STATUS_BG = (0.99, 0.96, 0.88)
        display_status = STATUS_REVIEW
    elif "UNABLE" in normalized_status:
        STATUS_COLOR = (0.45, 0.50, 0.58)  # Slate
        STATUS_BG = (0.93, 0.94, 0.96)
        display_status = STATUS_UNABLE
    else:
        STATUS_COLOR = (0.85, 0.20, 0.20)  # Red
        STATUS_BG = (0.99, 0.92, 0.92)
        display_status = STATUS_INCONSISTENCIES

    ref_code = report_reference or f"DOCV-2024-{doc_info.get('id', 1):04d}"

    # 1. Header Banner
    banner_title = "OFFICIAL DOCUMENT CONSISTENCY AUDIT REPORT" if is_admin_report else "DOCUMENT CONSISTENCY REVIEW CERTIFICATE"
    page.draw_rect(pymupdf.Rect(margin_x, curr_y, 595 - margin_x, curr_y + 60), color=None, fill=DARK_BLUE)
    page.insert_text((margin_x + 15, curr_y + 24), banner_title, fontsize=12, fontname="helv", color=(1, 1, 1))
    sub_title = (
        f"DocVerify Technical Audit Dossier (SIH1709) | Report Ref: {ref_code}"
        if is_admin_report else
        f"DocVerify Intelligent Document Verification Platform | Report Ref: {ref_code}"
    )
    page.insert_text(
        (margin_x + 15, curr_y + 44),
        sub_title,
        fontsize=8.5, fontname="helv", color=(0.85, 0.90, 0.98)
    )
    curr_y += 70

    # 2. Document Summary Box
    summary_box_height = 80
    page.draw_rect(pymupdf.Rect(margin_x, curr_y, 595 - margin_x, curr_y + summary_box_height), color=BORDER_COLOR, fill=LIGHT_BG)

    filename = doc_info.get("filename", "Unknown Document")
    doc_category = doc_info.get("declared_category", "Official Document")
    report_time = datetime.now().strftime("%d-%b-%Y %H:%M:%S")

    page.insert_text((margin_x + 15, curr_y + 18), f"Document File: {filename}", fontsize=9.5, fontname="helv", color=TEXT_MAIN)
    page.insert_text((margin_x + 15, curr_y + 34), f"Document Category: {doc_category}", fontsize=9, fontname="helv", color=TEXT_MAIN)
    page.insert_text((margin_x + 15, curr_y + 50), f"Audit Date & Time: {report_time}", fontsize=8.5, fontname="helv", color=TEXT_MUTED)
    page.insert_text((margin_x + 15, curr_y + 66), f"Report Reference ID: {ref_code}", fontsize=8.5, fontname="helv", color=TEXT_MUTED)

    # Status Badge
    badge_x = 595 - margin_x - 180
    page.draw_rect(pymupdf.Rect(badge_x, curr_y + 15, badge_x + 165, curr_y + 55), color=STATUS_COLOR, fill=STATUS_BG, width=1.2)
    page.insert_text((badge_x + 8, curr_y + 32), "VERIFICATION STATUS:", fontsize=7, fontname="helv", color=TEXT_MUTED)
    page.insert_text((badge_x + 8, curr_y + 45), display_status[:24], fontsize=8.5, fontname="helv", color=STATUS_COLOR)

    curr_y += summary_box_height + 15

    # 3. Extracted Information Section
    page.insert_text((margin_x, curr_y + 10), "1. Extracted Information & Verified Fields", fontsize=10.5, fontname="helv", color=DARK_BLUE)
    page.draw_line(pymupdf.Point(margin_x, curr_y + 14), pymupdf.Point(595 - margin_x, curr_y + 14), color=DARK_BLUE, width=1)
    curr_y += 22

    if fields:
        for field_name, meta in fields.items():
            if isinstance(meta, dict):
                val = meta.get("corrected_value") or meta.get("value") or "[NOT DETECTED]"
                conf = meta.get("confidence", 0.0)
                page_n = meta.get("page", 1)
            else:
                val = str(meta) if meta else "[NOT DETECTED]"
                conf = 90.0
                page_n = 1

            field_line = f"• {field_name}: {val}"
            page.insert_text((margin_x + 10, curr_y + 8), field_line, fontsize=8.5, fontname="helv", color=TEXT_MAIN)

            if is_admin_report:
                meta_str = f"Conf: {conf}% | Pg {page_n}"
                page.insert_text((595 - margin_x - 120, curr_y + 8), meta_str, fontsize=7.5, fontname="helv", color=TEXT_MUTED)

            curr_y += 16
    else:
        page.insert_text((margin_x + 10, curr_y + 8), "• No field extraction records available for this dossier.", fontsize=8.5, fontname="helv", color=TEXT_MUTED)
        curr_y += 16

    curr_y += 10

    # 4. Detected Issues & Findings Section
    page.insert_text((margin_x, curr_y + 10), "2. Consistency Findings & Detected Issues", fontsize=10.5, fontname="helv", color=DARK_BLUE)
    page.draw_line(pymupdf.Point(margin_x, curr_y + 14), pymupdf.Point(595 - margin_x, curr_y + 14), color=DARK_BLUE, width=1)
    curr_y += 22

    issues = detected_issues or []
    if not issues:
        # Extract issues from rule results if not explicitly provided
        for r in rules:
            if r.get("Severity") in ["INCONSISTENT", "NEEDS MANUAL REVIEW", "INVALID"]:
                issues.append(f"{r.get('Rule Name')}: {r.get('Explanation')}")

    if not issues or display_status == STATUS_PASSED:
        page.insert_text(
            (margin_x + 10, curr_y + 8),
            "✔ No logical contradictions, temporal inversions, or missing mandatory fields were identified.",
            fontsize=8.5, fontname="helv", color=(0.10, 0.55, 0.30)
        )
        curr_y += 18
    else:
        for issue in issues[:4]:
            page.insert_text((margin_x + 10, curr_y + 8), f"⚠️ {issue[:100]}", fontsize=8, fontname="helv", color=STATUS_COLOR)
            curr_y += 16

    curr_y += 10

    # 5. Recommended Actions Section
    page.insert_text((margin_x, curr_y + 10), "3. Recommended Actions", fontsize=10.5, fontname="helv", color=DARK_BLUE)
    page.draw_line(pymupdf.Point(margin_x, curr_y + 14), pymupdf.Point(595 - margin_x, curr_y + 14), color=DARK_BLUE, width=1)
    curr_y += 22

    actions = recommended_actions or []
    if not actions:
        if display_status == STATUS_PASSED:
            actions = ["Document is internally consistent. Ready for downstream workflow processing."]
        elif display_status in [STATUS_INCONSISTENCIES, STATUS_INCONSISTENT]:
            actions = ["Review the contradictory fields noted above. Re-upload an authentic, unexpired document."]
        elif display_status in [STATUS_UNABLE, STATUS_UNABLE_TO_VERIFY]:
            actions = ["Upload a clearer high-resolution scan (300 DPI) right-side up without glare."]
        else:
            actions = ["Document is awaiting brief officer confirmation of flagged items."]

    for action in actions[:3]:
        page.insert_text((margin_x + 10, curr_y + 8), f"→ {action[:105]}", fontsize=8.5, fontname="helv", color=TEXT_MAIN)
        curr_y += 16

    curr_y += 10

    # 6. Admin Technical Notes (Only if Admin Report or Reviewer Notes exist)
    if is_admin_report or reviewer_notes:
        page.insert_text((margin_x, curr_y + 10), "4. Administrative Review Notes & Audit Signoff", fontsize=10.5, fontname="helv", color=DARK_BLUE)
        page.draw_line(pymupdf.Point(margin_x, curr_y + 14), pymupdf.Point(595 - margin_x, curr_y + 14), color=DARK_BLUE, width=1)
        curr_y += 22

        r_notes = reviewer_notes or "Document processed under automated AI consistency pipeline."
        page.insert_text((margin_x + 10, curr_y + 8), f"Reviewer Decision Notes: {r_notes[:110]}", fontsize=8, fontname="helv", color=TEXT_MUTED)
        curr_y += 22

    # 7. Official Legal Authenticity Disclaimer Box
    disclaimer_height = 55
    page.draw_rect(pymupdf.Rect(margin_x, 842 - margin_x - disclaimer_height, 595 - margin_x, 842 - margin_x), color=BORDER_COLOR, fill=LIGHT_BG)
    page.insert_text(
        (margin_x + 10, 842 - margin_x - disclaimer_height + 15),
        "LEGAL AUTHENTICITY DISCLAIMER & REGULATORY NOTICE",
        fontsize=7.5, fontname="helv", color=DARK_BLUE
    )
    disclaimer_text = (
        "This report certifies automated consistency and internal logical coherence only. "
        "It does NOT establish legal authenticity or guarantee absence of counterfeit documents. "
        "Official verification requires querying issuer databases, digital PKI signatures, or cryptographically sealed QR codes."
    )
    page.insert_textbox(
        pymupdf.Rect(margin_x + 10, 842 - margin_x - disclaimer_height + 20, 595 - margin_x - 10, 842 - margin_x - 5),
        disclaimer_text,
        fontsize=6.5, fontname="helv", color=TEXT_MUTED
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes
