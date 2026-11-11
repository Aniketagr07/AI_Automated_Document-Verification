
from datetime import datetime


# =====================================
# DOCUMENT KNOWLEDGE REPRESENTATION
# =====================================

DOCUMENT_SCHEMAS = {

    "Educational Certificate": {

        "Student Name": {
            "data_type": "string",
            "required": True,
            "min_length": 2
        },

        "Certificate ID": {
            "data_type": "string",
            "required": True,
            "min_length": 3
        },

        "Institution": {
            "data_type": "string",
            "required": True,
            "min_length": 2
        },

        "Issue Date": {
            "data_type": "date",
            "required": True
        },

        "Expiry Date": {
            "data_type": "date",
            "required": False
        }
    },


    "Identity Document": {

        "Name": {
            "data_type": "string",
            "required": True,
            "min_length": 2
        },

        "Document Number": {
            "data_type": "string",
            "required": True,
            "min_length": 3
        },

        "Date of Birth": {
            "data_type": "date",
            "required": True
        },

        "Issue Date": {
            "data_type": "date",
            "required": False
        },

        "Expiry Date": {
            "data_type": "date",
            "required": False
        }
    },


    "Government Certificate": {

        "Applicant Name": {
            "data_type": "string",
            "required": True,
            "min_length": 2
        },

        "Certificate ID": {
            "data_type": "string",
            "required": True,
            "min_length": 3
        },

        "Issuing Authority": {
            "data_type": "string",
            "required": True,
            "min_length": 2
        },

        "Issue Date": {
            "data_type": "date",
            "required": True
        },

        "Expiry Date": {
            "data_type": "date",
            "required": False
        }
    }
}


# =====================================
# SUPPORTED DATE FORMATS
# =====================================

DATE_FORMATS = [

    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y-%m-%d",
    "%d.%m.%Y",

    "%d/%m/%y",
    "%d-%m-%y",

    "%B %d, %Y",
    "%b %d, %Y"
]


# =====================================
# DATE VALIDATION
# =====================================

def validate_date(value):

    if not isinstance(value, str) or not value.strip():
        return False

    for date_format in DATE_FORMATS:

        try:

            datetime.strptime(
                value.strip(),
                date_format
            )

            return True

        except ValueError:
            continue

    return False


# =====================================
# SCHEMA VALIDATION
# =====================================

def validate_document_schema(
    fields,
    document_type="Educational Certificate"
):

    results = []

    schema = DOCUMENT_SCHEMAS.get(document_type)

    if schema is None:

        raise ValueError(
            f"Unsupported document type: {document_type}"
        )

    for field, rules in schema.items():

        value = fields.get(field)

        # ---------------------------------
        # REQUIRED FIELD CHECK
        # ---------------------------------

        if rules["required"] and (
            value is None or not str(value).strip()
        ):

            results.append({

                "Field": field,
                "Status": "MISSING",
                "Explanation": f"{field} is required."

            })

            continue

        # ---------------------------------
        # OPTIONAL FIELD CHECK
        # ---------------------------------

        if value is None or not str(value).strip():

            results.append({

                "Field": field,
                "Status": "OPTIONAL",
                "Explanation": (
                    f"{field} is optional and was not provided."
                )

            })

            continue

        # ---------------------------------
        # DATE VALIDATION
        # ---------------------------------

        if rules["data_type"] == "date":

            if validate_date(value):

                results.append({

                    "Field": field,
                    "Status": "VALID FORMAT",
                    "Explanation": "Date format is supported."

                })

            else:

                results.append({

                    "Field": field,
                    "Status": "INVALID FORMAT",
                    "Explanation": (
                        "Date is invalid or uses an unsupported format."
                    )

                })

        # ---------------------------------
        # STRING VALIDATION
        # ---------------------------------

        else:

            min_length = rules.get("min_length", 1)

            if len(str(value).strip()) < min_length:

                results.append({

                    "Field": field,
                    "Status": "INVALID VALUE",
                    "Explanation": (
                        f"{field} contains too little information."
                    )

                })

            else:

                results.append({

                    "Field": field,
                    "Status": "PRESENT",
                    "Explanation": f"{field} has been extracted."

                })

    return results

