

import pymupdf
import pytesseract
import io
import re

from PIL import Image, ImageOps, ImageFilter


# -----------------------------------
# TESSERACT OCR CONFIGURATION
# -----------------------------------

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


OCR_CONFIG = "--oem 3 --psm 6"


# -----------------------------------
# IMAGE PREPROCESSING
# -----------------------------------

def preprocess_image(image):

    image = ImageOps.exif_transpose(image)

    image = image.convert("L")

    image = ImageOps.autocontrast(image)

    image = image.filter(
        ImageFilter.SHARPEN
    )

    return image


# -----------------------------------
# IMAGE OCR
# -----------------------------------

def extract_image_text(file_bytes):

    image = Image.open(
        io.BytesIO(file_bytes)
    )

    image = preprocess_image(image)

    return pytesseract.image_to_string(
        image,
        config=OCR_CONFIG
    )


# -----------------------------------
# PDF TEXT EXTRACTION + OCR
# -----------------------------------

def extract_pdf_text(file_bytes):

    text_parts = []

    with pymupdf.open(
        stream=file_bytes,
        filetype="pdf"
    ) as pdf:

        for page in pdf:

            page_text = page.get_text().strip()

            # Use OCR for pages with little or no selectable text
            if len(page_text) < 30:

                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False
                )

                image = Image.open(
                    io.BytesIO(
                        pix.tobytes("png")
                    )
                )

                image = preprocess_image(image)

                page_text = pytesseract.image_to_string(
                    image,
                    config=OCR_CONFIG
                )

            text_parts.append(page_text)

    return "\n".join(text_parts)


# -----------------------------------
# DOCUMENT FIELD ALIASES
# -----------------------------------

FIELD_ALIASES = {

    "Educational Certificate": {

        "Student Name": [
            "student name",
            "name of student",
            "student's name",
            "candidate name",
            "name"
        ],

        "Certificate ID": [
            "certificate id",
            "certificate number",
            "certificate no",
            "certificate no.",
            "registration number",
            "registration no",
            "roll number",
            "serial number"
        ],

        "Institution": [
            "institution name",
            "name of institution",
            "school name",
            "college name",
            "university name",
            "institution",
            "school",
            "college",
            "university"
        ],

        "Issue Date": [
            "date of issue",
            "issue date",
            "issued on",
            "date issued"
        ],

        "Expiry Date": [
            "date of expiry",
            "expiry date",
            "expiration date",
            "valid until",
            "valid upto"
        ]
    },


    "Identity Document": {

        "Name": [
            "full name",
            "name of holder",
            "holder name",
            "cardholder name",
            "name"
        ],

        "Document Number": [
            "document number",
            "document no",
            "document no.",
            "identification number",
            "id number",
            "card number"
        ],

        "Date of Birth": [
            "date of birth",
            "birth date",
            "dob"
        ],

        "Issue Date": [
            "date of issue",
            "issue date",
            "issued on",
            "date issued"
        ],

        "Expiry Date": [
            "date of expiry",
            "expiry date",
            "expiration date",
            "valid until",
            "valid upto"
        ]
    },


    "Government Certificate": {

        "Applicant Name": [
            "applicant name",
            "name of applicant",
            "applicant",
            "beneficiary name",
            "name"
        ],

        "Certificate ID": [
            "certificate id",
            "certificate number",
            "certificate no",
            "certificate no.",
            "application number",
            "registration number",
            "reference number"
        ],

        "Issuing Authority": [
            "issuing authority",
            "issuing department",
            "issued by",
            "department name",
            "authority",
            "issuer",
            "department"
        ],

        "Issue Date": [
            "date of issue",
            "issue date",
            "issued on",
            "date issued"
        ],

        "Expiry Date": [
            "date of expiry",
            "expiry date",
            "expiration date",
            "valid until",
            "valid upto"
        ]
    }
}


# -----------------------------------
# FIELD EXTRACTION
# -----------------------------------

def extract_fields(text, document_type="Educational Certificate"):

    if document_type not in FIELD_ALIASES:

        raise ValueError(
            f"Unsupported document type: {document_type}"
        )

    fields = {}

    # Normalize text
    text = text.replace("\r", "\n")

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    category_fields = FIELD_ALIASES[document_type]

    for field, aliases in category_fields.items():

        fields[field] = None

        # Match longer aliases first
        aliases = sorted(
            aliases,
            key=len,
            reverse=True
        )

        for line in lines:

            for alias in aliases:

                pattern = (
                    r"^\s*"
                    + re.escape(alias)
                    + r"\s*(?::|=|-)\s*(.*?)\s*$"
                )

                match = re.match(
                    pattern,
                    line,
                    flags=re.IGNORECASE
                )

                if match:

                    value = match.group(1).strip()

                    if value:

                        fields[field] = value
                        break

            if fields[field] is not None:
                break

    return fields


# -----------------------------------
# MAIN DOCUMENT PROCESSOR
# -----------------------------------

def process_document(
    uploaded_file,
    document_type="Educational Certificate"
):

    file_bytes = uploaded_file.getvalue()

    if uploaded_file.type == "application/pdf":

        extracted_text = extract_pdf_text(
            file_bytes
        )

    elif uploaded_file.type.startswith("image/"):

        extracted_text = extract_image_text(
            file_bytes
        )

    else:

        raise ValueError(
            "Unsupported document format."
        )

    fields = extract_fields(
        extracted_text,
        document_type
    )

    return extracted_text, fields

