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
import unicodedata
from typing import Dict, List, Any, Optional, Tuple
from document_schema import get_schema, parse_date_robust, DocumentSchema, FIELD_EQUIVALENTS, is_course_completion_document

# High-precision patterns for specific structured entities
DATE_REGEX = re.compile(
    r"\b(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|"
    r"\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|"
    r"\d{1,2}(?:st|nd|rd|th)?[-/\.\s]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-/\.\s,]+\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-/\.\s]+\d{1,2}(?:st|nd|rd|th)?(?:,)?[-/\.\s]+\d{2,4})\b",
    re.IGNORECASE
)

# Common identification patterns (Aadhaar 12-digit, PAN 10-char, Passport, Registration codes, Certificate codes)
ID_PATTERNS = [
    re.compile(r"(?i)\b(?:certificate\s+code|cert\s+code|verification\s+code|credential\s+id|cert\s+id|certificate\s+id)\s*(?::|=|-|–|—)?\s*[\"\'“”‘’]?([A-Za-z0-9\-\/\._]+)[\"\'“”‘’]?\b"),
    re.compile(r"(?i)\bcode\s*(?::|=|-|–|—)\s*[\"\'“”‘’]?([A-Za-z0-9\-\/\._]+)[\"\'“”‘’]?\b"),
    re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),                  # Indian PAN Card format
    re.compile(r"\b\d{4}\s+\d{4}\s+\d{4}\b"),                  # Aadhaar 12 digits with spaces
    re.compile(r"\b\d{12}\b"),                                 # Aadhaar 12 digits continuous
    re.compile(r"\b[A-Z][0-9]{7}\b"),                          # Indian Passport format
    re.compile(r"\b[A-Z0-9]{2,5}[-/][0-9]{3,8}(?:[-/][0-9]{2,4})?\b"), # Standard roll/cert serials
    re.compile(r"(?i)\b(?:code\s*(?::|=|-|–|—)\s*)[\"\'“”‘’]?(\d{5,12})[\"\'“”‘’]?\b"),
]

COMPLETION_TRIGGER_BODY = (
    r"(?:this\s+certificate\s+is\s+proudly\s+awarded\s+to|"
    r"this\s+certificate\s+is\s+proudly\s+presented\s+to|"
    r"this\s+certificate\s+is\s+awarded\s+to|"
    r"this\s+certificate\s+is\s+presented\s+to|"
    r"this\s+is\s+to\s+certify\s+that|"
    r"hereby\s+certifies\s+that|"
    r"this\s+certifies\s+that|"
    r"certifies\s+that|"
    r"is\s+hereby\s+awarded\s+to|"
    r"is\s+hereby\s+granted\s+to|"
    r"is\s+proudly\s+presented\s+to|"
    r"is\s+proudly\s+awarded\s+to|"
    r"is\s+presented\s+to|"
    r"is\s+awarded\s+to|"
    r"proudly\s+presented\s+to|"
    r"proudly\s+awarded\s+to|"
    r"presented\s+to|"
    r"awarded\s+to|"
    r"conferred\s+upon|"
    r"conferred\s+on|"
    r"given\s+to|"
    r"granted\s+to|"
    r"in\s+recognition\s+of\s+the|"
    r"in\s+recognition\s+of|"
    r"hereby\s+recognizes\s+that|"
    r"recognizes\s+that|"
    r"proudly\s+honors|"
    r"honors)"
)

HEADER_OR_TRIGGER_REGEX = re.compile(
    r"(?i)^(?:DECLARATION\s+OF\s+COMPLETION|CERTIFICATE\s+OF\s+COMPLETION|"
    r"CERTIFICATE\s+OF\s+ACHIEVEMENT|CERTIFICATE\s+OF\s+ACCOMPLISHMENT|"
    r"CERTIFICATE\s+OF\s+PARTICIPATION|CERTIFICATE\s+OF\s+TRAINING|"
    r"STATEMENT\s+OF\s+ACCOMPLISHMENT|COURSE\s+COMPLETION\s+CERTIFICATE|"
    r"TRAINING\s+COMPLETION\s+CERTIFICATE|CERTIFICATE\s+OF\s+EXCELLENCE|"
    r"CERTIFICATE\s+OF\s+MERIT|CERTIFICATE\s+OF\s+RECOGNITION|"
    r"CERTIFICATE\s+OF\s+ATTENDANCE|CERTIFICATE\s+OF\s+APPRECIATION|"
    r"CERTIFICATE\s+OF\s+SPECIALIZATION|COURSE\s+CERTIFICATE|"
    r"TRAINING\s+CERTIFICATE|COMPLETION\s+CERTIFICATE|"
    rf"{COMPLETION_TRIGGER_BODY}|"
    r"has\s+(?:successfully\s+)?graduated|has\s+(?:successfully\s+)?participated|"
    r"has\s+successfully\s+completed|has\s+completed|for\s+successfully\s+completing|"
    r"for\s+completing|completed\s+the\s+course).*"
)

# Individual disallow keywords for person name candidates (checked on whole-word boundaries)
BAD_NAME_WORDS = [
    "university", "college", "school", "board", "institute", "academy",
    "bachelor", "master", "doctor", "degree", "diploma",
    "certificate", "completion", "achievement", "accomplishment",
    "declaration", "training", "course", "courses", "skillup", "simplilearn",
    "microsoft", "google", "aws", "coursera", "udemy", "edx", "udacity",
    "cisco", "ibm", "oracle", "linkedin",
    "introduction", "advanced", "fundamentals", "engineering",
    "education", "learning", "online", "program", "programme", "track",
    "bootcamp", "specialization", "professional", "executive", "verified",
    "honor", "honour", "global", "center", "centre", "division", "department",
    "organization", "organisation", "foundation", "skills", "skill",
    "technologies", "technology", "solutions", "services", "system", "systems",
    "development", "knowledge", "network", "corporation", "corp", "inc", "llc", "ltd",
    "credential", "credentials", "id", "code", "grade", "score", "marks",
    "presented", "certify", "certified", "awarded", "conferred", "granted",
    "recipient", "participant", "candidate", "student", "applicant",
    "graduated", "graduate", "graduation", "participated", "participation",
    "attended", "attendance", "successfully", "completed", "completing",
    "passing", "passed", "requirements", "finished", "recognition",
]
BAD_NAME_WORD_REGEX = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in BAD_NAME_WORDS) + r")\b", re.IGNORECASE)

# Field and label keywords that must never be falsely treated as candidate field values
LABEL_WORDS = {
    "name", "student name", "applicant name", "recipient name", "candidate name",
    "participant name", "full name", "holder name", "cardholder name",
    "number", "no", "no.", "id", "code", "date", "issue date", "date of issue",
    "authority", "issuing authority", "institution", "organization", "platform",
    "of", "the", "for", "signature", "certified", "certify", "holder"
}

# Online learning platforms and certifiers keyword definitions
ONLINE_LEARNING_PLATFORMS = [
    (re.compile(r"(?i)\b(?:microsoft\s*skillup)\b"), "Microsoft SkillUp"),
    (re.compile(r"(?i)\b(?:simplilearn|simplelearn)\b"), "Simplilearn"),
    (re.compile(r"(?i)\b(?:microsoft)\b"), "Microsoft"),
    (re.compile(r"(?i)\b(?:google\s*cloud)\b"), "Google Cloud"),
    (re.compile(r"(?i)\b(?:google)\b"), "Google"),
    (re.compile(r"(?i)\b(?:aws|amazon\s*web\s*services)\b"), "AWS"),
    (re.compile(r"(?i)\b(?:coursera)\b"), "Coursera"),
    (re.compile(r"(?i)\b(?:udemy)\b"), "Udemy"),
    (re.compile(r"(?i)\b(?:edx)\b"), "edX"),
    (re.compile(r"(?i)\b(?:udacity)\b"), "Udacity"),
    (re.compile(r"(?i)\b(?:datacamp)\b"), "DataCamp"),
    (re.compile(r"(?i)\b(?:codecademy)\b"), "Codecademy"),
    (re.compile(r"(?i)\b(?:pluralsight)\b"), "Pluralsight"),
    (re.compile(r"(?i)\b(?:freecodecamp)\b"), "freeCodeCamp"),
    (re.compile(r"(?i)\b(?:kaggle)\b"), "Kaggle"),
    (re.compile(r"(?i)\b(?:cisco)\b"), "Cisco"),
    (re.compile(r"(?i)\b(?:ibm)\b"), "IBM"),
    (re.compile(r"(?i)\b(?:linkedin\s*learning)\b"), "LinkedIn Learning"),
    (re.compile(r"(?i)\b(?:oracle)\b"), "Oracle"),
]


def clean_extracted_value(val: str) -> str:
    """Strips leading/trailing punctuation, quotes, bullets, colons, and excessive spaces."""
    val = unicodedata.normalize("NFKC", str(val))
    cleaned = re.sub(r"^[\s:\-–—|=•\.,;\"'“”‘’]+", "", val)
    cleaned = re.sub(r"[\s:\-–—|=•\.,;\"'“”‘’]+$", "", cleaned)
    return cleaned.strip()


def normalize_person_name(name: str) -> str:
    """
    Normalises a raw name candidate:
    - Normalizes unicode (converting mathematical script/bold fonts to standard characters)
    - Strips honorary titles (Mr., Ms., Mrs., Dr., Prof., Shri., Smt.)
    - Strips leading trigger phrases (This is to certify that, Presented to, Certifies that...)
    - Strips leading label prefixes (Recipient Name, Participant Name, Student Name...)
    - Strips trailing relational conjunctions (son of, s/o, has completed…)
    - Title-cases all-uppercase names
    - Collapses runs of whitespace
    """
    name = unicodedata.normalize("NFKC", str(name)).strip()
    # Strip leading/trailing punctuation and quotes
    name = re.sub(r"^[\s:\-–—|=•\.,;\"'“”‘’]+", "", name)
    name = re.sub(r"[\s:\-–—|=•\.,;\"'“”‘’]+$", "", name).strip()
    # Remove titles at start
    name = re.sub(
        r"^(?:Mr\.?|Mrs\.?|Ms\.?|Miss\.?|Dr\.?|Prof\.?|Shri\.?|Smt\.?|Er\.?)\s+",
        "", name, flags=re.IGNORECASE
    ).strip()
    # Remove leading completion trigger phrases if prepended
    name = re.sub(
        rf"^{COMPLETION_TRIGGER_BODY}\s*[:.\-–—]?\s*",
        "", name, flags=re.IGNORECASE
    ).strip()
    # Remove leading role labels if prepended
    name = re.sub(
        r"^(?:recipient\s+name|participant\s+name|student\s+name|candidate\s+name|applicant\s+name|name)\s*[:.\-–—]?\s*",
        "", name, flags=re.IGNORECASE
    ).strip()
    # Remove trailing relational suffixes and completion phrasing
    name = re.split(
        r"\s+(?:son\s+of|daughter\s+of|s/o|d/o|w/o|has\s+successfully\s+completed|has\s+completed|have\s+completed|has\s+passed|has\s+(?:successfully\s+)?participated(?:\s+in)?|participated\s+in|has\s+attended|for\s+attending|for\s+participating|bearing|residing|for\s+successfully\s+completing|for\s+completing)",
        name, flags=re.IGNORECASE, maxsplit=1
    )[0].strip()
    # Strip any trailing punctuation and quotes
    name = re.sub(r"[\s:\-–—|=•\.,;\"'“”‘’]+$", "", name).strip()
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
    Handles optional whitespace before colons (e.g. "Certificate code : 9380068").
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
    for raw_alias in sorted_aliases:
        alias = unicodedata.normalize("NFKC", raw_alias).strip()
        # Clean trailing colon or punctuation from alias for flexible boundary matching
        alias_clean = re.sub(r"[\s:=–—\-]+$", "", alias)
        if not alias_clean:
            alias_clean = alias

        # Skip broad categorical noun aliases in inline scan if no explicit separator exists
        # e.g. "Lovely Professional University Conducted by..." shouldn't treat "university" as a prefix label
        is_broad_noun = alias_clean.lower() in {
            "university", "college", "school", "institute", "board", "academy",
            "platform", "authority", "organization", "issuer", "certifier"
        }

        for i, line in enumerate(lines):
            line_clean = unicodedata.normalize("NFKC", line).strip()
            if not line_clean:
                continue

            has_sep = any(sep in line_clean for sep in [":", "=", " - ", " – ", " — "])
            if is_broad_noun and not has_sep:
                continue

            # Handles alias followed by optional whitespace, optional colon/dash, whitespace, and value
            pattern = r"(?i)\b" + re.escape(alias_clean) + r"\b\s*(?::|=|-|–|—)?\s*(.+)"
            match = re.search(pattern, line_clean)
            if match:
                candidate = clean_extracted_value(match.group(1))
                # Truncate at common inline prose stop-phrases
                for stop_p in stop_phrases:
                    candidate = re.split(stop_p, candidate, flags=re.IGNORECASE)[0].strip()

                if candidate and len(candidate) >= 2:
                    if candidate.lower().strip() in LABEL_WORDS:
                        continue
                    if data_type == "identifier":
                        candidate = re.sub(r"\s*(?:\([^)]*\)|\[[^\]]*\])", "", candidate).strip()
                        candidate = re.split(r"\s*[\|,;]\s*", candidate)[0].strip()
                        candidate = re.split(r"\s+[-–—]\s+(?:date|dated|issue|issued|valid|status)\b", candidate, flags=re.IGNORECASE)[0].strip()
                        candidate = clean_extracted_value(candidate)
                    if data_type == "date":
                        if not parse_date_robust(candidate):
                            continue
                        confidence = 95.0
                    else:
                        confidence = 90.0 if any(sep in line_clean for sep in [":", "=", "-"]) else 80.0
                    return candidate, line_clean, confidence

    # Strategy 2: Label on one line, Value on the immediately succeeding line(s)
    for raw_alias in sorted_aliases:
        alias = unicodedata.normalize("NFKC", raw_alias).strip()
        alias_clean = re.sub(r"[\s:=–—\-]+$", "", alias)
        if not alias_clean:
            alias_clean = alias

        for i, line in enumerate(lines[:-1]):
            line_clean = unicodedata.normalize("NFKC", line).strip()
            # Allow trailing punctuation on the label line
            if re.fullmatch(r"(?i)\b" + re.escape(alias_clean) + r"\b\s*(?::|=|-|\.)?", line_clean):
                # Check up to 2 lines below for the value
                for offset in (1, 2):
                    if i + offset < len(lines):
                        next_line = unicodedata.normalize("NFKC", lines[i + offset]).strip()
                        if next_line and len(next_line) >= 2 and not HEADER_OR_TRIGGER_REGEX.match(next_line):
                            val_to_use = clean_extracted_value(next_line)
                            if val_to_use.lower().strip() in LABEL_WORDS:
                                continue
                            if data_type == "identifier":
                                val_to_use = re.sub(r"\s*(?:\([^)]*\)|\[[^\]]*\])", "", val_to_use).strip()
                                val_to_use = re.split(r"\s*[\|,;]\s*", val_to_use)[0].strip()
                                val_to_use = re.split(r"\s+[-–—]\s+(?:date|dated|issue|issued|valid|status)\b", val_to_use, flags=re.IGNORECASE)[0].strip()
                                val_to_use = clean_extracted_value(val_to_use)
                            if data_type == "date":
                                if not parse_date_robust(val_to_use):
                                    continue
                                conf = 92.0
                            else:
                                conf = 85.0
                            evidence = f"{line_clean} -> {next_line}"
                            return val_to_use, evidence, conf

    # Strategy 3: Value on a line, label in parentheses on the NEXT line
    # e.g.  "Peter Parker\n(Student Name)"
    for raw_alias in sorted_aliases:
        alias = unicodedata.normalize("NFKC", raw_alias).strip()
        alias_clean = re.sub(r"[\s:=–—\-]+$", "", alias)
        if not alias_clean:
            alias_clean = alias

        for i, line in enumerate(lines[1:], start=1):
            line_clean = unicodedata.normalize("NFKC", line).strip()
            if re.search(r"(?i)\(" + re.escape(alias_clean) + r"\)", line_clean):
                prev_line = clean_extracted_value(unicodedata.normalize("NFKC", lines[i - 1]).strip())
                if prev_line and len(prev_line) >= 2 and not HEADER_OR_TRIGGER_REGEX.match(prev_line):
                    if prev_line.lower().strip() in LABEL_WORDS:
                        continue
                    if data_type == "identifier":
                        prev_line = re.sub(r"\s*(?:\([^)]*\)|\[[^\]]*\])", "", prev_line).strip()
                        prev_line = re.split(r"\s*[\|,;]\s*", prev_line)[0].strip()
                        prev_line = clean_extracted_value(prev_line)
                    if data_type == "date" and not parse_date_robust(prev_line):
                        continue
                    evidence = f"{prev_line} <- ({alias_clean})"
                    return prev_line, evidence, 82.0

    return None


def extract_course_completion_name(text: str) -> Optional[Tuple[str, str, float]]:
    """
    Extracts recipient name from online course / training completion certificates.
    Supports:
    1. Layout pattern where the recipient's name appears between the completion header
       (e.g., "DECLARATION OF COMPLETION", "CERTIFICATE OF COMPLETION", "CERTIFICATE OF ACHIEVEMENT")
       and the completion phrasing (e.g., "has successfully completed" or "has completed"),
       whether on multiple lines, inline on the same line, or sharing header/phrasing lines.
    2. Layout pattern where the recipient name appears as the initial header line before
       course title / certification code (e.g. Simplilearn/Microsoft certificates).
    """
    text_norm = unicodedata.normalize("NFKC", text)
    lines = [l.strip() for l in text_norm.splitlines() if l.strip()]
    if not lines:
        return None

    header_pattern = re.compile(
        r"(?i)(?:DECLARATION\s+OF\s+COMPLETION|CERTIFICATE\s+OF\s+COMPLETION|"
        r"CERTIFICATE\s+OF\s+ACHIEVEMENT|CERTIFICATE\s+OF\s+ACCOMPLISHMENT|"
        r"CERTIFICATE\s+OF\s+PARTICIPATION|CERTIFICATE\s+OF\s+TRAINING|"
        r"STATEMENT\s+OF\s+ACCOMPLISHMENT|COURSE\s+COMPLETION\s+CERTIFICATE|"
        r"TRAINING\s+COMPLETION\s+CERTIFICATE|CERTIFICATE\s+OF\s+EXCELLENCE|"
        r"CERTIFICATE\s+OF\s+MERIT|CERTIFICATE\s+OF\s+RECOGNITION|"
        r"CERTIFICATE\s+OF\s+ATTENDANCE|CERTIFICATE\s+OF\s+APPRECIATION|"
        r"CERTIFICATE\s+OF\s+SPECIALIZATION|COURSE\s+CERTIFICATE|"
        r"TRAINING\s+CERTIFICATE|COMPLETION\s+CERTIFICATE)"
    )

    phrasing_pattern = re.compile(
        r"(?i)(?:has\s+successfully\s+completed|have\s+successfully\s+completed|"
        r"has\s+attended\s+and\s+(?:successfully\s+)?completed|"
        r"has\s+attended\b|for\s+attending\b|"
        r"has\s+(?:successfully\s+)?participated\s+in|for\s+participating\s+in|"
        r"has\s+(?:successfully\s+)?graduated(?:\s+from)?|"
        r"in\s+recognition\s+of\s+(?:the\s+)?(?:successful\s+)?completion\s+of|"
        r"for\s+(?:the\s+)?(?:successful\s+)?completion\s+of|"
        r"has\s+completed|have\s+completed|for\s+successfully\s+completing|"
        r"for\s+completing|successfully\s+completed|completed\s+the\s+course|"
        r"completed\s+the\s+training|completed\s+the\s+program|completed\s+the\s+programme|"
        r"has\s+fulfilled\s+(?:all\s+)?(?:the\s+)?requirements|"
        r"has\s+satisfied\s+(?:all\s+)?(?:the\s+)?requirements|"
        r"has\s+met\s+(?:all\s+)?(?:the\s+)?requirements|"
        r"completed\s+coursework|"
        r"for\s+completing\s+the|has\s+successfully\s+finished|"
        r"has\s+finished|has\s+attended\s+and\s+completed|"
        r"has\s+successfully\s+passed|for\s+successfully\s+passing|for\s+passing\s+the|"
        r"has\s+(?:successfully\s+)?cleared|has\s+(?:successfully\s+)?mastered)"
    )

    intermediate_triggers = re.compile(
        rf"(?i)^(?:{COMPLETION_TRIGGER_BODY}|"
        r"recipient\s+name|participant\s+name|student\s+name|candidate\s+name|applicant\s+name|name)\s*[:.\-–—]?\s*$"
    )
    trigger_prefix_cleaner = re.compile(
        rf"^(?:{COMPLETION_TRIGGER_BODY}|"
        r"recipient\s+name|participant\s+name|student\s+name|candidate\s+name|applicant\s+name|name)\s*[:.\-–—]?\s*",
        re.IGNORECASE
    )

    # Strategy 1: Between completion header and completion phrasing
    header_idx = -1
    header_match_obj = None
    phrasing_idx = -1
    phrasing_match_obj = None

    for i, line in enumerate(lines):
        if header_idx == -1:
            m_h = header_pattern.search(line)
            if m_h:
                header_idx = i
                header_match_obj = m_h
                # Check if phrasing is ALSO on the same header line
                m_p = phrasing_pattern.search(line[m_h.end():])
                if m_p:
                    phrasing_idx = i
                    phrasing_match_obj = m_p
                    break
        else:
            m_p = phrasing_pattern.search(line)
            if m_p:
                phrasing_idx = i
                phrasing_match_obj = m_p
                break

    if header_idx != -1 and phrasing_idx != -1:
        intermediate_lines = []

        if header_idx == phrasing_idx:
            # Header and phrasing on the same line: extract candidate between them
            header_end = header_match_obj.end()
            line_str = lines[header_idx]
            m_p_abs = phrasing_pattern.search(line_str, pos=header_end)
            if m_p_abs:
                between = line_str[header_end:m_p_abs.start()].strip()
                cleaned_between = trigger_prefix_cleaner.sub("", between).strip()
                cleaned_between = clean_extracted_value(cleaned_between)
                if cleaned_between:
                    intermediate_lines.append(cleaned_between)
        else:
            # Header on header_idx, phrasing on phrasing_idx
            # Check if header line itself had text after the header
            header_end = header_match_obj.end()
            header_suffix = lines[header_idx][header_end:].strip()
            if header_suffix:
                cleaned_h = trigger_prefix_cleaner.sub("", header_suffix).strip()
                cleaned_h = clean_extracted_value(cleaned_h)
                if cleaned_h and not intermediate_triggers.match(cleaned_h):
                    intermediate_lines.append(cleaned_h)

            for idx in range(header_idx + 1, phrasing_idx):
                l_str = lines[idx]
                if intermediate_triggers.match(l_str):
                    continue
                cleaned_l = trigger_prefix_cleaner.sub("", l_str).strip()
                cleaned_l = clean_extracted_value(cleaned_l)
                if cleaned_l:
                    intermediate_lines.append(cleaned_l)

            # Only check phrasing_prefix if intermediate_lines is empty
            if not intermediate_lines:
                phrasing_match = phrasing_match_obj
                phrasing_prefix = lines[phrasing_idx][:phrasing_match.start()].strip() if phrasing_match else ""
                if phrasing_prefix and not intermediate_triggers.match(phrasing_prefix):
                    cleaned_prefix = trigger_prefix_cleaner.sub("", phrasing_prefix).strip()
                    cleaned_prefix = clean_extracted_value(cleaned_prefix)
                    if cleaned_prefix and cleaned_prefix.lower() not in {"who", "and", "that", "which", "has"}:
                        intermediate_lines.append(cleaned_prefix)

        if intermediate_lines:
            joined_candidate = normalize_person_name(" ".join(intermediate_lines))
            joined_words = joined_candidate.split()
            if 1 <= len(joined_words) <= 5 and all(re.match(r"^[A-Za-z][A-Za-z\.\'-]*$", w) for w in joined_words):
                if not BAD_NAME_WORD_REGEX.search(joined_candidate):
                    evidence = f"{lines[header_idx]} -> {joined_candidate} -> {lines[phrasing_idx]}"
                    return joined_candidate, evidence, 94.0

        for line in intermediate_lines:
            candidate = normalize_person_name(line)
            words = candidate.split()
            if 1 <= len(words) <= 5 and all(re.match(r"^[A-Za-z][A-Za-z\.\'-]*$", w) for w in words):
                if not BAD_NAME_WORD_REGEX.search(candidate):
                    evidence = f"{lines[header_idx]} -> {candidate} -> {lines[phrasing_idx]}"
                    return candidate, evidence, 94.0

    # Strategy 1b: Regex on full text for header ... [Name] ... phrasing
    block_match = re.search(
        r"(?i)(?:DECLARATION\s+OF\s+COMPLETION|CERTIFICATE\s+OF\s+COMPLETION|"
        r"CERTIFICATE\s+OF\s+ACHIEVEMENT|CERTIFICATE\s+OF\s+ACCOMPLISHMENT|"
        r"CERTIFICATE\s+OF\s+PARTICIPATION|CERTIFICATE\s+OF\s+TRAINING|"
        r"STATEMENT\s+OF\s+ACCOMPLISHMENT|COURSE\s+COMPLETION\s+CERTIFICATE|"
        r"TRAINING\s+COMPLETION\s+CERTIFICATE|CERTIFICATE\s+OF\s+EXCELLENCE|"
        r"CERTIFICATE\s+OF\s+MERIT|CERTIFICATE\s+OF\s+RECOGNITION)\s*(?:\n+|[\s:\-–—\.]+)\s*"
        r"(?:(?:this\s+is\s+to\s+certify\s+that|presented\s+to|awarded\s+to|proudly\s+presented\s+to|conferred\s+upon|conferred\s+on)\s*(?:\n+|[\s:\-–—\.]+)\s*)?"
        r"([A-Za-z][A-Za-z\s\.\'-]{1,50}?)\s*(?:\n+|[\s:\-–—\.]+)\s*"
        r"(?:has\s+successfully\s+completed|has\s+completed|for\s+successfully\s+completing|completed\s+the\s+course)",
        text_norm
    )
    if block_match:
        candidate = normalize_person_name(clean_extracted_value(block_match.group(1)))
        if 2 < len(candidate) < 60 and not BAD_NAME_WORD_REGEX.search(candidate):
            return candidate, block_match.group(0).strip()[:100], 95.0

    # Strategy 2: First line(s) before course title / certificate code
    # e.g. "Shivam Patil\nIntroduction to Prompt Engineering with\nGitHub Copilot\n..."
    course_indicators = re.compile(
        r"(?i)^(?:Introduction\s+to|Advanced\s+|Mastering\s+|Fundamentals\s+of|"
        r"Course\s+on|Training\s+on|Specialization|Professional\s+Certificate|"
        r"Deep\s+Learning|Machine\s+Learning|Prompt\s+Engineering|GitHub\s+Copilot|"
        r"Data\s+Science|Cloud|AWS|Microsoft|Google|Python|Java|DevOps|"
        r".*?\b(?:course|training|engineering|programming|analytics|computing|intelligence|developer|foundations)\b)"
    )
    cert_indicators = re.compile(r"(?i)\b(?:certificate\s+code|cert\s+code|verification\s+code|code\s*:|credential\s+id)\b")

    for i in range(min(3, len(lines))):
        line = lines[i]
        if (
            HEADER_OR_TRIGGER_REGEX.match(line)
            or phrasing_pattern.search(line)
            or intermediate_triggers.match(line)
            or course_indicators.search(line)
            or cert_indicators.search(line)
        ):
            continue
        candidate = normalize_person_name(line)
        if BAD_NAME_WORD_REGEX.search(candidate):
            continue
        words = candidate.split()
        if not words or words[0].lower() in {"has", "have", "is", "was", "for", "in", "with", "by", "to", "at", "from", "of", "who", "which", "that"}:
            continue
        if 2 <= len(words) <= 5 and all(re.match(r"^[A-Za-z][A-Za-z\.\'-]*$", w) for w in words):
            has_succeeding_course = (i + 1 < len(lines) and bool(course_indicators.search(lines[i + 1])))
            next_is_course_or_date = False
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                next_is_course_or_date = (
                    bool(course_indicators.search(next_line))
                    or bool(DATE_REGEX.search(next_line))
                    or bool(cert_indicators.search(next_line))
                )
            if has_succeeding_course or (next_is_course_or_date and bool(cert_indicators.search(text_norm))):
                evidence = f"Recipient Name: {candidate}"
                return candidate, evidence, 92.0

    return None


def extract_template_student_name(text: str) -> Optional[Tuple[str, str, float]]:
    """
    Extracts student name from traditional academic certificate templates.
    Handles:
    - Name on the SAME line after the trigger phrase
    - Name on the NEXT non-empty line after the trigger (most common in real certs)
    - Various trigger phrase forms with trailing punctuation (. : -)
    """
    text_norm = unicodedata.normalize("NFKC", text)

    # First attempt course completion specific extraction
    course_cand = extract_course_completion_name(text_norm)
    if course_cand:
        return course_cand

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

    # Strategy A: Name on the SAME line after the trigger
    inline_stop = (
        r"(?:\s+(?:s/o|d/o|w/o|son\s+of|daughter\s+of|has\s+successfully|"
        r"has\s+(?:successfully\s+)?participated(?:\s+in)?|participated\s+in|"
        r"has\s+attended|for\s+attending|for\s+participating|"
        r"has\s+completed|has\s+passed|bearing|enrolment|enrolled|residing)|\n|,)"
    )
    for trigger in trigger_patterns:
        pattern = trigger + r"\s+([A-Za-z][A-Za-z\s\.\'-]{1,58}?)" + inline_stop
        m = re.search(pattern, text_norm)
        if m:
            candidate = normalize_person_name(clean_extracted_value(m.group(1)))
            if 2 < len(candidate) < 60 and not BAD_NAME_WORD_REGEX.search(candidate):
                return candidate, m.group(0).strip(), 93.0

    # Strategy B: Name on the NEXT non-empty line after the trigger
    lines = text_norm.splitlines()
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
                        if 2 < len(candidate) < 60 and not BAD_NAME_WORD_REGEX.search(candidate):
                            evidence = f"{line.strip()} -> {candidate_line}"
                            return candidate, evidence, 90.0
                    break  # Only check the first non-empty line

    return None


def extract_template_institution(text: str, filename: str = "") -> Optional[Tuple[str, str, float]]:
    """
    Extracts institution name or issuing authority from certificate headers,
    signatures, online learning platform certifiers, or document filename.
    """
    text_norm = unicodedata.normalize("NFKC", text)

    # Strategy 1: Academic / university patterns in document text (spaces only, no newlines across words)
    inst_keywords = [
        r"\b([A-Z][A-Za-z]+(?:[ \t]+[A-Z][A-Za-z]+){0,4}[ \t]+(?:University|Institute(?:[ \t]+of[ \t]+[A-Za-z]+)?|College|Academy|Board))\b",
        r"(?i)\b(?:conducted\s+by|organized\s+by)\s+([A-Z][A-Za-z0-9]+(?:[ \t]+[A-Z][A-Za-z0-9]+){0,3})",
        r"(?i)\b(Government\s+of\s+India|Skill\s+India)\b",
        r"(?i)\b([A-Za-z][A-Za-z \t\.\'-]+(?:University|Institute\s+of\s+Technology|College\s+of\s+Engineering|Polytechnic|Academy|Board\s+of\s+Education))\b",
        r"(?i)\b(Indian\s+Institute\s+of\s+[A-Za-z \t]+)\b",
        r"(?i)\b(National\s+Institute\s+of\s+[A-Za-z \t]+)\b"
    ]
    for kw in inst_keywords:
        match = re.search(kw, text_norm)
        if match:
            raw_cand = match.group(1) if match.lastindex else match.group(0)
            candidate = clean_extracted_value(re.sub(r"(?i)^(?:held\s+at|at|by|in)\s+", "", raw_cand))
            if 3 < len(candidate) < 80:
                return candidate, match.group(0).strip(), 88.0

    # Strategy 2: Online learning platforms and certifiers in document text
    for pat, name in ONLINE_LEARNING_PLATFORMS:
        m = pat.search(text_norm)
        if m:
            return name, f"Identified learning platform: {m.group(0).strip()}", 90.0

    # Strategy 3: Check document filename for learning platforms / certifiers
    if filename:
        clean_fn = unicodedata.normalize("NFKC", filename).replace("_", " ").replace("-", " ")
        for pat, name in ONLINE_LEARNING_PLATFORMS:
            m = pat.search(clean_fn)
            if m:
                return name, f"Identified from document source: {name} ({filename})", 88.0

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
    full_text = unicodedata.normalize("NFKC", processed_doc.get("full_text", ""))
    filename = processed_doc.get("filename", "")

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
            raw_page_text = unicodedata.normalize("NFKC", page_data.get("text", ""))
            lines = [l for l in raw_page_text.splitlines() if l.strip()]

            res = extract_field_from_lines(lines, aliases, constraint.data_type)
            if res:
                found_val, found_evidence, found_conf = res
                found_page = page_num
                break

        # Pass 2: Semantic template extraction if field is still missing
        if not found_val:
            if field_name in ["Student Name", "Name", "Applicant Name", "Full Name"]:
                res = extract_course_completion_name(full_text)
                if not res:
                    res = extract_template_student_name(full_text)
                if res:
                    found_val, found_evidence, found_conf = res
                    found_page = 1

            elif field_name in ["Institution", "Issuing Authority"]:
                res = extract_template_institution(full_text, filename=filename)
                if res:
                    found_val, found_evidence, found_conf = res
                    found_page = 1

            elif field_name in ["Certificate ID", "Document Number"]:
                # Check for standalone ID patterns in full text
                for id_pat in ID_PATTERNS:
                    m = id_pat.search(full_text)
                    if m:
                        raw_id = m.group(1).strip() if m.lastindex else m.group(0).strip()
                        cand_id = clean_extracted_value(raw_id)
                        # Ensure candidate is not a date or date fragment (e.g. 11/2025 from 21/11/2025)
                        if parse_date_robust(cand_id) or re.search(r"\b\d{1,2}[/-]" + re.escape(cand_id), full_text) or re.match(r"^\d{1,2}[/-]\d{4}$", cand_id):
                            continue
                        found_val = cand_id
                        found_evidence = f"Pattern match: {m.group(0).strip()}"
                        found_conf = 88.0
                        found_page = 1
                        break

            elif constraint.data_type == "date":
                # Fallback: scan for any explicit date mention with proximity to alias
                for alias in aliases:
                    alias_norm = unicodedata.normalize("NFKC", alias).strip()
                    alias_clean = re.sub(r"[\s:=–—\-]+$", "", alias_norm)
                    if not alias_clean:
                        alias_clean = alias_norm
                    idx = full_text.lower().find(alias_clean.lower())
                    if idx != -1:
                        snippet = full_text[idx:idx + 120]
                        date_match = DATE_REGEX.search(snippet)
                        if date_match:
                            found_val = date_match.group(0).strip()
                            found_evidence = f"{snippet.strip()}"
                            found_conf = 85.0
                            found_page = 1
                            break

                # Last-resort for Issue Date: find the first standalone date in the document
                # that hasn't been claimed by another date field yet
                if not found_val and field_name == "Issue Date":
                    all_dates = DATE_REGEX.findall(full_text)
                    if all_dates:
                        # Pick last date found (issue date often appears near bottom)
                        candidate_date = all_dates[-1].strip()
                        parsed_d = parse_date_robust(candidate_date)
                        if parsed_d:
                            found_val = candidate_date
                            found_evidence = f"Document date: {found_val}"
                            found_conf = 88.0
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

    # Pass 3: Tolerant cross-category field alignment
    # If a field is missing in schema, populate from extracted equivalent field if available
    for field_name, record in extracted_records.items():
        if record.get("status") == "MISSING":
            for eq_name in FIELD_EQUIVALENTS.get(field_name, []):
                eq_record = extracted_records.get(eq_name)
                if eq_record and eq_record.get("value"):
                    extracted_records[field_name] = {
                        "value": eq_record["value"],
                        "page": eq_record.get("page", 1),
                        "evidence": eq_record.get("evidence", ""),
                        "confidence": eq_record.get("confidence", 85.0),
                        "status": "PRESENT",
                        "manual_verification_required": False
                    }
                    break

    # Pass 4: Tolerant non-applicable alignment for course/training completion documents
    if is_course_completion_document(extracted_records, full_text):
        for field_name, record in extracted_records.items():
            if record.get("status") == "MISSING" and field_name not in [
                "Student Name", "Applicant Name", "Name", "Full Name", "Certificate ID", "Document Number",
                "Institution", "Issuing Authority", "Issue Date"
            ]:
                record["status"] = "NOT_APPLICABLE"
                record["manual_verification_required"] = False
                record["evidence"] = f"Field '{field_name}' is not applicable for course completion certificate."

    return extracted_records
