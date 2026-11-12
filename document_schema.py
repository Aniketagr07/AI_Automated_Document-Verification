"""
Knowledge Representation Module (SIH1709).
Defines formal, extensible schemas for official documents including:
- Data types, required vs optional fields
- Identifier pattern constraints (Aadhaar, PAN, Passport, Roll/Registration No)
- Date constraints and cross-field relationship rules
- Extensibility for adding new document categories dynamically
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime, date
import re

# Standard date formats supported across Indian and International documents
DATE_FORMATS = [
    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
    "%Y-%m-%d", "%Y/%m/%d",
    "%d/%m/%y", "%d-%m-%y",
    "%d %B %Y", "%d %b %Y",
    "%d-%b-%Y", "%d-%B-%Y",
    "%d/%b/%Y", "%d/%B/%Y",
    "%d.%b.%Y", "%d.%B.%Y",
    "%d-%b-%y", "%d-%B-%y",
    "%d/%b/%y", "%d.%b.%y",
    "%B %d, %Y", "%b %d, %Y",
    "%d %B, %Y", "%d %b, %Y",
    "%B %d %Y", "%b %d %Y",
    "%b %d, %y", "%B %d, %y",
    "%Y-%b-%d", "%Y-%B-%d",
    "%Y"  # Year only (e.g. in some graduation certificates)
]


def parse_date_robust(value: Optional[str]) -> Optional[date]:
    """Parses a date string across diverse formats with clean whitespace and quote stripping."""
    if not value or not isinstance(value, str):
        return None
    cleaned = value.strip().strip("\"'“”‘’").replace(",", ", ").replace("  ", " ")
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    # Try regex fallback for formats like 15th August 2022
    cleaned_ord = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", cleaned, flags=re.IGNORECASE)
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned_ord, fmt).date()
        except ValueError:
            continue
    return None


def validate_date(value: Any) -> bool:
    """Validates if value can be parsed as a supported calendar date."""
    if not value:
        return False
    return parse_date_robust(str(value)) is not None


@dataclass
class FieldConstraint:
    """Defines formal constraints on an individual document field."""
    field_name: str
    data_type: str  # 'string', 'date', 'identifier', 'number'
    required: bool = True
    min_length: int = 1
    max_length: int = 255
    regex_pattern: Optional[str] = None
    description: str = ""
    example: str = ""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class CrossFieldRule:
    """Formal knowledge representation of relational rules between fields."""
    rule_id: str
    rule_name: str
    field_a: str
    field_b: str
    relationship: str  # 'A_LESS_THAN_OR_EQUAL_B', 'A_GREATER_THAN_B', 'AGE_AT_EVENT_VALID'
    description: str


@dataclass
class DocumentSchema:
    """Complete Knowledge Representation Schema for a document category."""
    category_id: str
    category_name: str
    description: str
    fields: Dict[str, FieldConstraint]
    cross_field_rules: List[CrossFieldRule] = field(default_factory=list)
    required_keywords: List[str] = field(default_factory=list)
    alias_map: Dict[str, List[str]] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.fields[key]

    def __contains__(self, key: str) -> bool:
        return key in self.fields

    def get(self, key: str, default: Any = None) -> Any:
        return self.fields.get(key, default)

    def keys(self):
        return self.fields.keys()

    def values(self):
        return self.fields.values()

    def items(self):
        return self.fields.items()


# ===========================================================================
# PRE-BUILT SCHEMAS FOR OFFICIAL DOCUMENT TYPES
# ===========================================================================

EDUCATIONAL_CERTIFICATE_SCHEMA = DocumentSchema(
    category_id="educational_certificate",
    category_name="Educational Certificate",
    description="Degrees, Diplomas, Marksheets, and Academic Certificates.",
    fields={
        "Student Name": FieldConstraint(
            field_name="Student Name",
            data_type="string",
            required=True,
            min_length=2,
            regex_pattern=r"^[A-Za-z\s\.\'-]+$",
            description="Full legal name of the student or graduate."
        ),
        "Certificate ID": FieldConstraint(
            field_name="Certificate ID",
            data_type="identifier",
            required=True,
            min_length=3,
            regex_pattern=r"^[A-Za-z0-9\-\/\._]+$",
            description="Unique certificate, serial, or roll/registration number."
        ),
        "Institution": FieldConstraint(
            field_name="Institution",
            data_type="string",
            required=True,
            min_length=3,
            description="Name of the university, board, college, or institute."
        ),
        "Issue Date": FieldConstraint(
            field_name="Issue Date",
            data_type="date",
            required=True,
            description="Date on which the degree/certificate was awarded."
        ),
        "Expiry Date": FieldConstraint(
            field_name="Expiry Date",
            data_type="date",
            required=False,
            description="Optional expiration date for provisional or vocational certifications."
        )
    },
    cross_field_rules=[
        CrossFieldRule(
            rule_id="CFR-EDU-01",
            rule_name="Award Date Consistency",
            field_a="Issue Date",
            field_b="Expiry Date",
            relationship="A_LESS_THAN_OR_EQUAL_B",
            description="Certificate issue date must be on or before its expiry date."
        )
    ],
    required_keywords=[
        "certify", "degree", "diploma", "examination", "conferred",
        "passed", "university", "institute", "college", "board", "grade", "marks",
        "certificate", "completion", "achievement", "completed", "course", "training",
        "simplilearn", "microsoft", "google", "aws", "coursera", "udemy", "edx"
    ],
    alias_map={
        "Student Name": [
            "student name", "name of student", "student's name", "candidate name",
            "name of the candidate", "certifies that", "certified that", "awarded to",
            "this is to certify that", "son of", "daughter of", "name",
            "applicant name", "recipient name", "recipient", "participant name", "presented to"
        ],
        "Certificate ID": [
            "certificate id", "certificate number", "certificate no", "certificate no.",
            "registration number", "registration no", "registration no.", "roll number",
            "roll no", "roll no.", "enrolment number", "enrollment no", "serial number",
            "cert no", "hall ticket no", "folio no",
            "certificate code", "cert code", "verification code", "code :", "code", "credential id", "cert id"
        ],
        "Institution": [
            "institution name", "name of institution", "university name", "college name",
            "school name", "board of education", "institute of", "university", "college",
            "school", "board", "academy",
            "issuing authority", "organization", "issuer", "learning platform", "platform", "certifier"
        ],
        "Issue Date": [
            "date of issue", "issue date", "issued on", "date issued", "date of award",
            "awarded on", "dated", "convocation date", "date",
            "date of completion", "completion date", "completed on",
            # OCR misread variants (e.g. "Issue" → "ksue", "lssue", "lssue")
            "date of ksue", "date of lssue", "date of isue", "date of issure",
            "date of issuance", "issuance date", "date awarded", "date of certification"
        ],
        "Expiry Date": [
            "date of expiry", "expiry date", "expiration date", "valid until",
            "valid upto", "valid through", "validity", "valid till"
        ]
    }
)


IDENTITY_DOCUMENT_SCHEMA = DocumentSchema(
    category_id="identity_document",
    category_name="Identity Document",
    description="Aadhaar, Passport, Driving License, Voter ID, PAN card.",
    fields={
        "Name": FieldConstraint(
            field_name="Name",
            data_type="string",
            required=True,
            min_length=2,
            regex_pattern=r"^[A-Za-z\s\.\'-]+$",
            description="Full legal name of the identity holder."
        ),
        "Document Number": FieldConstraint(
            field_name="Document Number",
            data_type="identifier",
            required=True,
            min_length=4,
            description="Official identity number (Aadhaar 12-digit, PAN 10-char, Passport, etc.)."
        ),
        "Date of Birth": FieldConstraint(
            field_name="Date of Birth",
            data_type="date",
            required=True,
            description="Birth date of the identity holder."
        ),
        "Issue Date": FieldConstraint(
            field_name="Issue Date",
            data_type="date",
            required=False,
            description="Date when document or card was issued."
        ),
        "Expiry Date": FieldConstraint(
            field_name="Expiry Date",
            data_type="date",
            required=False,
            description="Expiration date of passport, license, or identity credential."
        )
    },
    cross_field_rules=[
        CrossFieldRule(
            rule_id="CFR-ID-01",
            rule_name="Birth Prior to Issue",
            field_a="Date of Birth",
            field_b="Issue Date",
            relationship="A_LESS_THAN_B",
            description="Date of birth must strictly precede document issue date."
        ),
        CrossFieldRule(
            rule_id="CFR-ID-02",
            rule_name="Validity Horizon",
            field_a="Issue Date",
            field_b="Expiry Date",
            relationship="A_LESS_THAN_OR_EQUAL_B",
            description="Identity document issue date must be on or before expiry date."
        )
    ],
    required_keywords=[
        "identity", "government", "india", "republic", "holder", "card",
        "birth", "father", "address", "signature", "authority", "permanent account"
    ],
    alias_map={
        "Name": [
            "full name", "name of holder", "holder name", "cardholder name",
            "name", "given name", "surname", "beneficiary",
            "applicant name", "student name", "recipient name", "recipient", "participant name", "presented to", "awarded to"
        ],
        "Document Number": [
            "document number", "document no", "document no.", "identification number",
            "id number", "card number", "pan", "pan number", "aadhaar number",
            "uid", "passport no", "passport number", "license no", "epic no",
            "certificate id", "certificate code", "cert code", "verification code", "code :", "code", "credential id", "cert id"
        ],
        "Date of Birth": [
            "date of birth", "birth date", "dob", "d.o.b.", "year of birth", "born on"
        ],
        "Issue Date": [
            "date of issue", "issue date", "issued on", "date of issue / issuance", "dated", "date",
            "date of completion", "completion date", "completed on"
        ],
        "Expiry Date": [
            "date of expiry", "expiry date", "expiration date", "valid until",
            "valid upto", "date of expiration"
        ]
    }
)


GOVERNMENT_CERTIFICATE_SCHEMA = DocumentSchema(
    category_id="government_certificate",
    category_name="Government Certificate",
    description="Caste, Domicile, Income, Birth, and Official Notified Certificates.",
    fields={
        "Applicant Name": FieldConstraint(
            field_name="Applicant Name",
            data_type="string",
            required=True,
            min_length=2,
            regex_pattern=r"^[A-Za-z\s\.\'-]+$",
            description="Legal name of the applicant or recipient."
        ),
        "Certificate ID": FieldConstraint(
            field_name="Certificate ID",
            data_type="identifier",
            required=True,
            min_length=3,
            description="Official tracking / application / certificate registration number."
        ),
        "Issuing Authority": FieldConstraint(
            field_name="Issuing Authority",
            data_type="string",
            required=True,
            min_length=3,
            description="Government department, officer designation, or municipal body."
        ),
        "Issue Date": FieldConstraint(
            field_name="Issue Date",
            data_type="date",
            required=True,
            description="Official date of attestation or gazetted issuance."
        ),
        "Expiry Date": FieldConstraint(
            field_name="Expiry Date",
            data_type="date",
            required=False,
            description="Validity expiration date (for income, creamy layer, etc.)."
        )
    },
    cross_field_rules=[
        CrossFieldRule(
            rule_id="CFR-GOV-01",
            rule_name="Government Validity Chronology",
            field_a="Issue Date",
            field_b="Expiry Date",
            relationship="A_LESS_THAN_OR_EQUAL_B",
            description="Government certificate issue date must precede expiry date."
        )
    ],
    required_keywords=[
        "government", "revenue", "tehsildar", "magistrate", "district", "collector",
        "certificate", "applicant", "certify", "attestation", "official", "state",
        "completion", "course", "training"
    ],
    alias_map={
        "Applicant Name": [
            "applicant name", "name of applicant", "applicant", "beneficiary name",
            "name of the applicant", "certifies that", "certified that", "name",
            "student name", "recipient name", "recipient", "participant name", "presented to", "awarded to"
        ],
        "Certificate ID": [
            "certificate id", "certificate number", "certificate no", "certificate no.",
            "application number", "application no", "registration number", "reference number",
            "ack no", "barcode number", "token no",
            "certificate code", "cert code", "verification code", "code :", "code", "credential id", "cert id"
        ],
        "Issuing Authority": [
            "issuing authority", "issuing department", "issued by", "department name",
            "authority", "competent authority", "office of", "tehsildar", "district magistrate",
            "sub divisional magistrate", "revenue officer",
            "institution", "organization", "issuer", "learning platform", "platform", "certifier"
        ],
        "Issue Date": [
            "date of issue", "issue date", "issued on", "date issued", "dated", "date",
            "date of completion", "completion date", "completed on"
        ],
        "Expiry Date": [
            "date of expiry", "expiry date", "expiration date", "valid until",
            "valid upto", "valid for financial year"
        ]
    }
)


# Schema Registry for flexible knowledge management & extensibility
SCHEMA_REGISTRY: Dict[str, DocumentSchema] = {
    "Educational Certificate": EDUCATIONAL_CERTIFICATE_SCHEMA,
    "Identity Document": IDENTITY_DOCUMENT_SCHEMA,
    "Government Certificate": GOVERNMENT_CERTIFICATE_SCHEMA
}

DOCUMENT_SCHEMAS = SCHEMA_REGISTRY  # Backward compatibility alias


def register_custom_schema(schema: DocumentSchema) -> None:
    """Enables extending the system with new document categories at runtime."""
    SCHEMA_REGISTRY[schema.category_name] = schema


def get_schema(category_name: str) -> DocumentSchema:
    """Fetches schema by category name with sensible fallback."""
    if category_name in SCHEMA_REGISTRY:
        return SCHEMA_REGISTRY[category_name]
    for key, val in SCHEMA_REGISTRY.items():
        if key.lower() == category_name.lower():
            return val
    raise ValueError(f"Schema not found for category: '{category_name}'")


FIELD_EQUIVALENTS: Dict[str, List[str]] = {
    "Student Name": ["Applicant Name", "Name", "Recipient Name", "Full Name"],
    "Applicant Name": ["Student Name", "Name", "Recipient Name", "Full Name"],
    "Name": ["Student Name", "Applicant Name", "Recipient Name", "Full Name"],
    "Full Name": ["Student Name", "Applicant Name", "Name", "Recipient Name"],
    "Institution": ["Issuing Authority", "Organization", "Issuer"],
    "Issuing Authority": ["Institution", "Organization", "Issuer"],
    "Certificate ID": ["Document Number", "Certificate Code", "Code", "Credential ID"],
    "Document Number": ["Certificate ID", "Certificate Code", "Code", "Credential ID"],
}


def is_course_completion_document(
    fields_dict: Optional[Dict[str, Any]] = None,
    raw_text: str = ""
) -> bool:
    """
    Checks if a document exhibits course or training completion characteristics.
    Inspects raw text, filenames, and extracted field values against completion indicators.
    """
    course_indicators = [
        "completion", "completed", "achievement", "accomplishment", "participation",
        "course", "courses", "training", "prompt engineering",
        "copilot", "simplilearn", "simplelearn", "microsoft skillup", "skillup",
        "coursera", "udemy", "edx", "udacity", "cisco", "datacamp", "codecademy",
        "pluralsight", "freecodecamp", "kaggle",
        "google cloud", "google", "aws", "amazon web services", "microsoft",
        "skill boost", "skills build", "skill builder",
        "certificate code", "cert code", "verification code", "credential id", "code :",
        "declaration of completion", "certificate of completion", "certificate of achievement",
        "statement of accomplishment", "course certificate", "training certificate"
    ]
    text_lower = (raw_text or "").lower()
    if any(k in text_lower for k in course_indicators):
        return True
    if fields_dict:
        for v in fields_dict.values():
            v_val = v.get("value") if isinstance(v, dict) else v
            v_ev = v.get("evidence") if isinstance(v, dict) else ""
            v_str = (str(v_val or "") + " " + str(v_ev or "")).lower()
            if any(k in v_str for k in course_indicators):
                return True
    return False


def validate_document_schema(
    fields_dict: Dict[str, Any],
    category_name: str,
    raw_text: str = ""
) -> List[Dict[str, str]]:
    """
    Validates extracted field values against the formal knowledge representation schema.
    Returns list of validation records with Field, Status, and Explanation.
    Supports tolerant category mapping for course and training completion certificates.
    """
    schema = get_schema(category_name)
    results = []

    # Check if document exhibits course/training completion characteristics
    is_course_cert = is_course_completion_document(fields_dict, raw_text)

    for field_name, constraint in schema.fields.items():
        field_info = fields_dict.get(field_name, {})
        # Support both raw string or metadata dict from extractor
        if isinstance(field_info, dict):
            val = field_info.get("value")
        else:
            val = field_info

        val_str = str(val).strip() if val is not None else ""

        # Tolerant category mapping: check equivalent field names across categories
        if not val_str:
            for eq_field in FIELD_EQUIVALENTS.get(field_name, []):
                eq_info = fields_dict.get(eq_field, {})
                eq_val = eq_info.get("value") if isinstance(eq_info, dict) else eq_info
                if eq_val:
                    val_str = str(eq_val).strip()
                    val = eq_val
                    break

        # Tolerant handling: non-applicable fields for course completion certificates
        is_participation = bool(re.search(r"(?i)\b(?:participation|participated|attendance|attended|workshop|webinar)\b", raw_text))
        if constraint.required and not val_str and is_course_cert:
            if is_participation and field_name in ["Certificate ID", "Document Number"]:
                results.append({
                    "Field": field_name,
                    "Status": "NOT_APPLICABLE",
                    "Explanation": f"Field '{field_name}' is not mandatory for participation or workshop credentials."
                })
                continue
            if field_name not in [
                "Student Name", "Applicant Name", "Name", "Full Name", "Certificate ID", "Document Number",
                "Institution", "Issuing Authority", "Issue Date"
            ]:
                results.append({
                    "Field": field_name,
                    "Status": "NOT_APPLICABLE",
                    "Explanation": f"Field '{field_name}' is not applicable for course completion certificate."
                })
                continue

        # Check required fields
        if constraint.required and not val_str:
            results.append({
                "Field": field_name,
                "Status": "MISSING",
                "Explanation": f"Mandatory field '{field_name}' is missing or could not be extracted."
            })
            continue

        # Optional field not provided
        if not val_str:
            results.append({
                "Field": field_name,
                "Status": "OPTIONAL_ABSENT",
                "Explanation": f"Optional field '{field_name}' is not present in document."
            })
            continue

        # Data type validation: Date
        if constraint.data_type == "date":
            parsed = parse_date_robust(val_str)
            if parsed is None:
                results.append({
                    "Field": field_name,
                    "Status": "INVALID FORMAT",
                    "Explanation": f"Date '{val_str}' does not conform to supported calendar formats."
                })
            else:
                results.append({
                    "Field": field_name,
                    "Status": "VALID FORMAT",
                    "Explanation": f"Valid date parsed: {parsed.strftime('%d-%b-%Y')}."
                })
        else:
            # String / Identifier validation
            if len(val_str) < constraint.min_length:
                results.append({
                    "Field": field_name,
                    "Status": "INVALID VALUE",
                    "Explanation": f"Extracted value '{val_str}' is shorter than minimum length {constraint.min_length}."
                })
            elif constraint.regex_pattern and not re.search(constraint.regex_pattern, val_str):
                results.append({
                    "Field": field_name,
                    "Status": "FORMAT WARNING",
                    "Explanation": f"Value '{val_str}' deviates from expected pattern for {field_name}."
                })
            else:
                results.append({
                    "Field": field_name,
                    "Status": "PRESENT",
                    "Explanation": f"Field '{field_name}' extracted with valid value: '{val_str}'."
                })

    return results
