
from datetime import datetime, date

from document_schema import (
    validate_document_schema,
    DATE_FORMATS
)


# -----------------------------------
# 1. DATE PARSING
# -----------------------------------

def parse_date(value):

    if not value or not isinstance(value, str):
        return None

    for fmt in DATE_FORMATS:

        try:
            return datetime.strptime(
                value.strip(), fmt
            ).date()

        except ValueError:
            continue

    return None


# -----------------------------------
# 2. RULE ENGINE
# -----------------------------------

def run_rule_engine(
    fields,
    document_type="Educational Certificate"
):

    today = date.today()

    schema_results = validate_document_schema(
        fields,
        document_type
    )

    triggered_rules = []

    # -----------------------------------
    # RULE 1: REQUIRED FIELD CHECK
    # -----------------------------------

    for result in schema_results:

        field = result["Field"]
        schema_status = result["Status"]

        if schema_status == "MISSING":

            triggered_rules.append({
                "Rule ID": "R1",
                "Rule Name": f"{field} Check",
                "Severity": "MANUAL REVIEW",
                "Explanation": result["Explanation"]
            })

        # -----------------------------------
        # RULE 2: INVALID FORMAT
        # -----------------------------------

        elif schema_status == "INVALID FORMAT":

            triggered_rules.append({
                "Rule ID": "R2",
                "Rule Name": f"{field} Format Check",
                "Severity": "MANUAL REVIEW",
                "Explanation": result["Explanation"]
            })

        # -----------------------------------
        # RULE 3: INVALID STRING VALUE
        # -----------------------------------

        elif schema_status == "INVALID VALUE":

            triggered_rules.append({
                "Rule ID": "R3",
                "Rule Name": f"{field} Value Check",
                "Severity": "MANUAL REVIEW",
                "Explanation": result["Explanation"]
            })

    # -----------------------------------
    # RULE 4: DATE OF BIRTH CHECK
    # -----------------------------------

    dob = parse_date(
        fields.get("Date of Birth")
    )

    if dob and dob > today:

        triggered_rules.append({
            "Rule ID": "R4",
            "Rule Name": "Date of Birth Check",
            "Severity": "INVALID",
            "Explanation": "Date of birth cannot be in the future."
        })

    # -----------------------------------
    # RULE 5: FUTURE ISSUE DATE
    # -----------------------------------

    issue_date = parse_date(
        fields.get("Issue Date")
    )

    if issue_date and issue_date > today:

        triggered_rules.append({
            "Rule ID": "R5",
            "Rule Name": "Future Issue Date",
            "Severity": "INVALID",
            "Explanation": "Issue date is in the future."
        })

    # -----------------------------------
    # RULE 6: EXPIRY DATE CONSISTENCY
    # -----------------------------------

    expiry_date = parse_date(
        fields.get("Expiry Date")
    )

    if (
        issue_date
        and expiry_date
        and expiry_date < issue_date
    ):

        triggered_rules.append({
            "Rule ID": "R6",
            "Rule Name": "Date Consistency",
            "Severity": "INVALID",
            "Explanation": (
                "Expiry date is earlier than issue date."
            )
        })

    # -----------------------------------
    # RULE 7: DOCUMENT EXPIRY
    # -----------------------------------

    elif expiry_date and expiry_date < today:

        triggered_rules.append({
            "Rule ID": "R7",
            "Rule Name": "Expiry Status",
            "Severity": "INVALID",
            "Explanation": "Document has expired."
        })

    # -----------------------------------
    # FINAL STATUS
    # -----------------------------------

    if any(
        rule["Severity"] == "INVALID"
        for rule in triggered_rules
    ):

        status = "INVALID"

    elif triggered_rules:

        status = "NEEDS MANUAL REVIEW"

    else:

        status = "VALID"

        triggered_rules.append({
            "Rule ID": "R8",
            "Rule Name": "All Checks Passed",
            "Severity": "PASS",
            "Explanation": (
                "All implemented consistency rules passed."
            )
        })

    return status, triggered_rules

