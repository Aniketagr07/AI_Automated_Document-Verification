"""
Information Extraction Module for Automated Document Verification System (SIH1709).
Implements multi-strategy, layout-tolerant field extraction:
- Anchor-value proximity parsing (same line, adjacent lines, tab/table alignments)
- Sentence-template semantic extraction ("This is to certify that [Name]...")
- Format-specific regex recognition (Dates, Document IDs, Roll Numbers)
- Tracks source page, surrounding supporting text (evidence), confidence score,
  and manual review verification flags for every field.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from document_schema import get_schema, parse_date_robust, DocumentSchema

# High-precision patterns for specific structured entities
DATE_REGEX = re.compile(
    r"\b(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|"
    r"\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{4}|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE
)

# Common identification patterns (Aadhaar 12-digit, PAN 10-char, Passport, Registration codes)
ID_PATTERNS = [
    re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),                  # Indian PAN Card format
    re.compile(r"\b\d{4}\s+\d{4}\s+\d{4}\b"),                  # Aadhaar 12 digits with spaces
    re.compile(r"\b\d{12}\b"),                                 # Aadhaar 12 digits continuous
    re.compile(r"\b[A-Z][0-9]{7}\b"),                          # Indian Passport format
    re.compile(r"\b[A-Z0-9]{2,5}[-/][0-9]{3,8}(?:[-/][0-9]{2,4})?\b"), # Standard roll/cert serials
]


def clean_extracted_value(val: str) -> str:
    """Strips leading/trailing punctuation, bullets, colons, and excessive spaces."""
    cleaned = re.sub(r"^[\s:\-–—|=•]+", "", val)
    cleaned = re.sub(r"[\s:\-–—|=•]+$", "", cleaned)
    return cleaned.strip()


def normalize_person_name(name: str) -> str:
    """
    Normalises a raw name candidate:
    - Strips honorary titles (Mr., Ms., Mrs., Dr., Prof., Shri., Smt.)
    - Strips trailing relational conjunctions (son of, s/o, has completed…)
    - Title-cases all-uppercase names
    - Collapses runs of whitespace
    """
    # Remove titles at start
    name = re.sub(
        r"^(?:Mr\.?|Mrs\.?|Ms\.?|Miss\.?|Dr\.?|Prof\.?|Shri\.?|Smt\.?|Er\.?)\s+",
        "", name, flags=re.IGNORECASE
    ).strip()
    # Remove trailing relational suffixes
    name = re.split(
        r"\s+(?:son\s+of|daughter\s+of|s/o|d/o|w/o|has\s+completed|has\s+passed|bearing|residing)",
        name, flags=re.IGNORECASE, maxsplit=1
    )[0].strip()
    # Collapse extra whitespace
    name = re.sub(r"\s{2,}", " ", name)
    # Title-case if the name is fully uppercase (OCR artifact)
    if name.isupper() and len(name.split()) <= 5:
        name = name.title()
    return name


def extract_field_from_lines(
    lines: List[str],
    aliases: List[str],
    data_type: str = "string"
) -> Optional[Tuple[str, str, float]]:
    """
    Scans document lines for aliases and retrieves candidate value,
    evidence line, and match confidence.
    Prioritizes specific/longer aliases across all lines first.
    """
    sorted_aliases = sorted(aliases, key=len, reverse=True)

    stop_phrases = [
        r"\s+has\s+completed\b",
        r"\s+has\s+passed\b",
        r"\s+bearing\b",
        r"\s+dated\b",
        r"\s+issued\s+on\b",
        r"\s+son\s+of\b",
        r"\s+daughter\s+of\b",
        r"\s+s/o\b",
        r"\s+d/o\b",
        r"\s+w/o\b",
        r"\s+having\b",
        r"\s+residing\s+at\b"
    ]

    # Strategy 1: Search highest priority aliases across all lines first
    for alias in sorted_aliases:
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean:
                continue

            pattern = r"(?i)\b" + re.escape(alias) + r"\b\s*(?::|=|-|–|—)?\s*(.+)"
            match = re.search(pattern, line_clean)
            if match:
                candidate = clean_extracted_value(match.group(1))
                # Truncate at common inline prose stop-phrases
                for stop_p in stop_phrases:
                    candidate = re.split(stop_p, candidate, flags=re.IGNORECASE)[0].strip()

                if candidate and len(candidate) >= 2:
                    confidence = 90.0 if any(sep in line_clean for sep in [":", "=", "-"]) else 80.0
                    if data_type == "date" and parse_date_robust(candidate):
                        confidence = 95.0
                    return candidate, line_clean, confidence

    # Strategy 2: Label on one line, Value on the immediately succeeding line(s)
    for alias in sorted_aliases:
        for i, line in enumerate(lines[:-1]):
            line_clean = line.strip()
            # Allow trailing punctuation on the label line
            if re.fullmatch(r"(?i)\b" + re.escape(alias) + r"\b\s*(?::|=|-|\.)?", line_clean):
                # Check up to 2 lines below for the value
                for offset in (1, 2):
                    if i + offset < len(lines):
                        next_line = lines[i + offset].strip()
                        if next_line and len(next_line) >= 2:
                            evidence = f"{line_clean} -> {next_line}"
                            conf = 85.0
                            if data_type == "date" and parse_date_robust(next_line):
                                conf = 92.0
                            return next_line, evidence, conf

    # Strategy 3: Value on a line, label in parentheses on the NEXT line
    # e.g.  "Peter Parker\n(Student Name)"
    for alias in sorted_aliases:
        for i, line in enumerate(lines[1:], start=1):
            line_clean = line.strip()
            if re.search(r"(?i)\(" + re.escape(alias) + r"\)", line_clean):
                prev_line = lines[i - 1].strip()
                if prev_line and len(prev_line) >= 2:
                    evidence = f"{prev_line} <- ({alias})"
                    return prev_line, evidence, 82.0

    return None


def extract_template_student_name(text: str) -> Optional[Tuple[str, str, float]]:
    """
    Extracts student name from traditional academic certificate templates.
    Handles:
    - Name on the SAME line after the trigger phrase
    - Name on the NEXT non-empty line after the trigger (most common in real certs)
    - Various trigger phrase forms with trailing punctuation (. : -)
    """
    # Trigger phrases — allow trailing . : - before or instead of the name
    # Also allow \s* between words to handle OCR word-merging (THISISTOCERTIFY → THIS IS TO CERTIFY)
    trigger_patterns = [
        r"(?i)this\s*is\s*to\s*certify\s*that[.:\-]?",
        r"(?i)this\s*certifies?\s*that[.:\-]?",
        r"(?i)proudly\s*presented\s*to[.:\-]?",
        r"(?i)presented\s*to[.:\-]?",
        r"(?i)awarded\s*to[.:\-]?",
        r"(?i)conferred\s*(?:upon|on)[.:\-]?",
        r"(?i)this\s*certificate\s*is\s*(?:presented|awarded)\s*to[.:\-]?",
        r"(?i)acknowledges?\s*that[.:\-]?",
        r"(?i)certif(?:ied|ies)\s*that[.:\-]?",
        r"(?i)is\s*hereby\s*granted\s*to[.:\-]?",
    ]

    bad_kw = ["university", "college", "school", "board", "institute",
              "bachelor", "master", "doctor", "degree", "diploma"]

    # Strategy A: Name on the SAME line after the trigger
    inline_stop = (
        r"(?:\s+(?:s/o|d/o|w/o|son\s+of|daughter\s+of|has\s+successfully|"
        r"has\s+completed|has\s+passed|bearing|enrolment|enrolled|residing)|\n|,)"
    )
    for trigger in trigger_patterns:
        pattern = trigger + r"\s+([A-Za-z][A-Za-z\s\.\'-]{1,58}?)" + inline_stop
        m = re.search(pattern, text)
        if m:
            candidate = normalize_person_name(clean_extracted_value(m.group(1)))
            if 2 < len(candidate) < 60 and not any(kw in candidate.lower() for kw in bad_kw):
                return candidate, m.group(0).strip(), 93.0

    # Strategy B: Name on the NEXT non-empty line after the trigger
    lines = text.splitlines()
    for trigger in trigger_patterns:
        for i, line in enumerate(lines):
            if re.search(trigger, line.strip()):
                for j in range(i + 1, min(i + 4, len(lines))):
                    candidate_line = lines[j].strip()
                    if not candidate_line:
                        continue
                    # Skip lines that look like degree/section text
                    if re.search(
                        r"(?i)^(?:in\s+the|has\s+completed|has\s+passed|bachelor|master|"
                        r"doctor|degree|diploma|university|college|school|board|certificate|"
                        r"awarded|conferred|successfully|requirements)",
                        candidate_line
                    ):
                        break
                    # Accept if it looks like a person name (1-5 alpha words)
                    words = candidate_line.split()
                    if 1 <= len(words) <= 5 and all(
                        re.match(r"^[A-Za-z][A-Za-z\.\'-]*$", w) for w in words
                    ):
                        candidate = normalize_person_name(candidate_line)
                        if 2 < len(candidate) < 60 and not any(kw in candidate.lower() for kw in bad_kw):
                            evidence = f"{line.strip()} -> {candidate_line}"
                            return candidate, evidence, 90.0
                    break  # Only check the first non-empty line

    return None


def extract_template_institution(text: str) -> Optional[Tuple[str, str, float]]:
    """Extracts institution name from certificate headers or signatures."""
    inst_keywords = [
        r"(?i)([A-Za-z\s\.\'-]+(?:University|Institute\s+of\s+Technology|College\s+of\s+Engineering|Polytechnic|Academy|Board\s+of\s+Education))",
        r"(?i)(Indian\s+Institute\s+of\s+[A-Za-z\s]+)",
        r"(?i)(National\s+Institute\s+of\s+[A-Za-z\s]+)"
    ]
    for kw in inst_keywords:
        match = re.search(kw, text)
        if match:
            candidate = clean_extracted_value(match.group(1))
            if 4 < len(candidate) < 80:
                return candidate, match.group(0).strip(), 88.0
    return None


def extract_structured_fields(
    processed_doc: Dict[str, Any],
    category_name: str
) -> Dict[str, Dict[str, Any]]:
    """
    Executes multi-strategy field extraction over multi-page document data.
    Returns dictionary mapping field names to detailed metadata:
    {
        "value": str,
        "page": int,
        "evidence": str,
        "confidence": float,
        "status": "PRESENT" | "MISSING" | "UNCERTAIN",
        "manual_verification_required": bool
    }
    """
    schema = get_schema(category_name)
    pages = processed_doc.get("pages", [])
    full_text = processed_doc.get("full_text", "")

    extracted_records: Dict[str, Dict[str, Any]] = {}

    for field_name, constraint in schema.fields.items():
        aliases = schema.alias_map.get(field_name, [field_name])
        found_val = None
        found_page = 1
        found_evidence = ""
        found_conf = 0.0

        # Pass 1: Page-by-page anchor line scan
        for page_data in pages:
            page_num = page_data.get("page_number", 1)
            lines = [l for l in page_data.get("text", "").splitlines() if l.strip()]

            res = extract_field_from_lines(lines, aliases, constraint.data_type)
            if res:
                found_val, found_evidence, found_conf = res
                found_page = page_num
                break

        # Pass 2: Semantic template extraction if field is still missing
        if not found_val:
            if field_name in ["Student Name", "Name", "Applicant Name"]:
                res = extract_template_student_name(full_text)
                if res:
                    found_val, found_evidence, found_conf = res
                    found_page = 1

            elif field_name in ["Institution", "Issuing Authority"]:
                res = extract_template_institution(full_text)
                if res:
                    found_val, found_evidence, found_conf = res
                    found_page = 1

            elif field_name in ["Certificate ID", "Document Number"]:
                # Check for standalone ID patterns in full text
                for id_pat in ID_PATTERNS:
                    m = id_pat.search(full_text)
                    if m:
                        found_val = m.group(0).strip()
                        found_evidence = f"Pattern match: {found_val}"
                        found_conf = 75.0
                        found_page = 1
                        break

            elif constraint.data_type == "date":
                # Fallback: scan for any explicit date mention with proximity to alias
                for alias in aliases:
                    idx = full_text.lower().find(alias.lower())
                    if idx != -1:
                        snippet = full_text[idx:idx + 120]
                        date_match = DATE_REGEX.search(snippet)
                        if date_match:
                            found_val = date_match.group(0).strip()
                            found_evidence = f"{snippet.strip()}"
                            found_conf = 78.0
                            found_page = 1
                            break

                # Last-resort for Issue Date: find the first standalone date in the document
                # that hasn't been claimed by another date field yet
                if not found_val and field_name == "Issue Date":
                    all_dates = DATE_REGEX.findall(full_text)
                    if all_dates:
                        # Pick last date found (issue date often appears near bottom)
                        found_val = all_dates[-1].strip()
                        found_evidence = f"Standalone date fallback: {found_val}"
                        found_conf = 55.0
                        found_page = 1

        # Post-processing and status categorization
        if found_val:
            # Validate format consistency
            if constraint.data_type == "date":
                parsed = parse_date_robust(found_val)
                if not parsed:
                    found_conf = min(found_conf, 45.0)

            # Manual verification requirement flag
            manual_req = (found_conf < 75.0) or (constraint.required and found_conf < 85.0)
            status = "PRESENT" if found_conf >= 70.0 else "UNCERTAIN"

            extracted_records[field_name] = {
                "value": found_val,
                "page": found_page,
                "evidence": found_evidence,
                "confidence": round(found_conf, 1),
                "status": status,
                "manual_verification_required": manual_req
            }
        else:
            # Required fields that are absent → MISSING
            # Optional fields that are absent → NOT_APPLICABLE (not a failure)
            if constraint.required:
                field_status = "MISSING"
                field_evidence = "Field anchor and value pattern not identified in document text."
            else:
                field_status = "NOT_APPLICABLE"
                field_evidence = "Optional field — not present in this document."
            extracted_records[field_name] = {
                "value": None,
                "page": 1,
                "evidence": field_evidence,
                "confidence": 0.0,
                "status": field_status,
                "manual_verification_required": constraint.required
            }

    return extracted_records
