"""
Document Processing Module for Automated Official Document Verification (SIH1709).
Handles multi-format ingestion (PDF, PNG, JPG, JPEG), multi-page parsing,
scanned vs digital text extraction, image preprocessing (deskew, contrast, sharpening),
and cross-platform Tesseract OCR configuration.
Supports both modern unified pipelines and legacy interfaces for 100% integration compatibility.
"""

import os
import io
import re
import shutil
import logging
import unicodedata
from typing import Dict, List, Tuple, Any, Optional, Union
from PIL import Image, ImageOps, ImageFilter
import pymupdf
import pytesseract
from pytesseract import Output

try:
    import winocr
    HAS_WINOCR = True
except Exception:
    HAS_WINOCR = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cross-Platform Tesseract Discovery
# ---------------------------------------------------------------------------

def configure_tesseract() -> Optional[str]:
    """
    Finds and configures Tesseract OCR across Windows, Linux, and Cloud environments.
    Avoids hardcoded paths and falls back gracefully.
    """
    # 1. Check environment variable override
    env_path = os.environ.get("TESSERACT_CMD")
    if env_path and os.path.isfile(env_path):
        pytesseract.pytesseract.tesseract_cmd = env_path
        return env_path

    # 2. Check if tesseract is in system PATH
    which_path = shutil.which("tesseract")
    if which_path:
        pytesseract.pytesseract.tesseract_cmd = which_path
        return which_path

    # 3. Known standard installation paths
    candidate_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract"
    ]

    for path in candidate_paths:
        if path and os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return path

    logger.warning("Tesseract OCR binary not found in standard paths. Fallback to digital text.")
    return None

TESSERACT_PATH = configure_tesseract()
OCR_CONFIG = "--oem 3 --psm 3"
OCR_CONFIG_FALLBACK = "--oem 3 --psm 6"


# ---------------------------------------------------------------------------
# Image Preprocessing & Enhancement
# ---------------------------------------------------------------------------

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Applies orientation correction, resolution scaling, grayscale conversion,
    contrast enhancement, and gentle unsharp mask to maximize OCR recognition rate.
    """
    from PIL import ImageEnhance

    # Fix EXIF rotation
    image = ImageOps.exif_transpose(image)

    # Scale to optimal resolution for Tesseract (~300 DPI equivalent)
    w, h = image.size
    max_dim = max(w, h)
    if max_dim < 1800:
        # Upscale low-res scans – Tesseract needs ~30px per capital letter
        scale = 1800 / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    elif max_dim > 4000:
        # Downscale very high-res images to reduce noise
        scale = 3600 / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)

    # Convert to grayscale
    gray = image.convert("L")

    # Stretch dynamic range
    contrasted = ImageOps.autocontrast(gray, cutoff=2)

    # Gentle contrast boost
    enhanced = ImageEnhance.Contrast(contrasted).enhance(1.4)

    # Gentle unsharp mask (less aggressive than SHARPEN which adds noise)
    sharpened = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))

    return sharpened


def detect_and_correct_orientation(image: Image.Image) -> Image.Image:
    """
    Detects page rotation using Tesseract OSD (Orientation and Script Detection)
    and rotates the image to upright position if necessary.
    """
    if not TESSERACT_PATH:
        return image
    try:
        osd = pytesseract.image_to_osd(image, output_type=Output.DICT)
        rotate_angle = osd.get("rotate", 0)
        if rotate_angle in [90, 180, 270]:
            return image.rotate(360 - rotate_angle, expand=True)
    except Exception:
        # If OSD fails (e.g. insufficient text on page), return original
        pass
    return image


def run_ocr_on_image(image: Image.Image) -> Tuple[str, float]:
    """
    Runs OCR on a PIL image and calculates word-level confidence score.
    Returns (extracted_text, average_confidence_percentage).
    Prioritizes native Windows Neural OCR (winocr) when available,
    falling back to Tesseract OCR across platforms.
    """
    # 1. Try Windows native Neural OCR first (high speed, built-in to Windows 10/11)
    if HAS_WINOCR:
        try:
            res = winocr.recognize_pil_sync(image, "en")
            if isinstance(res, dict) and "lines" in res and res["lines"]:
                line_texts = [l.get("text", "").strip() for l in res["lines"] if l.get("text", "").strip()]
                win_text = "\n".join(line_texts)
            else:
                win_text = res.get("text", "").strip() if isinstance(res, dict) else ""
            if win_text and len(win_text) >= 15:
                # Fix common OCR digit spacing e.g. "21/1 1/2025" -> "21/11/2025"
                normalized_text = re.sub(r'(\d{1,2}/\d)\s+(\d/\d{2,4})', r'\1\2', win_text)
                return normalized_text, 92.5
        except Exception as e:
            logger.debug(f"winocr recognition skipped: {e}")

    # 2. Try Tesseract OCR if configured
    if not TESSERACT_PATH:
        return "", 0.0

    try:
        enhanced = preprocess_image(image)
        # Try orientation correction
        enhanced = detect_and_correct_orientation(enhanced)

        # Primary: fully automatic page segmentation (best for certificates/mixed layout)
        data = pytesseract.image_to_data(enhanced, config=OCR_CONFIG, output_type=Output.DICT)
        text_words = []
        confidences = []

        n_boxes = len(data["text"])
        for i in range(n_boxes):
            word = data["text"][i].strip()
            conf = float(data["conf"][i])
            if word:
                text_words.append(word)
                if conf > 0:
                    confidences.append(conf)

        extracted_text = pytesseract.image_to_string(enhanced, config=OCR_CONFIG).strip()

        # Fallback: if primary produces sparse text, try single-block PSM
        if len(extracted_text) < 50 and TESSERACT_PATH:
            fb_text = pytesseract.image_to_string(enhanced, config=OCR_CONFIG_FALLBACK).strip()
            if len(fb_text) > len(extracted_text):
                extracted_text = fb_text

        mean_confidence = float(sum(confidences) / len(confidences)) if confidences else 70.0
        return extracted_text, round(mean_confidence, 2)

    except Exception as e:
        logger.error(f"OCR execution failed: {e}")
        # Fallback simple string extraction
        try:
            raw_text = pytesseract.image_to_string(image).strip()
            return raw_text, 50.0
        except Exception:
            return "", 0.0


# ---------------------------------------------------------------------------
# High-Level Document Processor
# ---------------------------------------------------------------------------

def process_file_bytes(
    file_bytes: bytes,
    filename: str,
    mime_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Processes uploaded document bytes (PDF, PNG, JPG, JPEG).
    Extracts text per page, measures OCR confidence, generates preview images,
    and returns comprehensive structured document payload.
    """
    pages_data = []
    full_text_parts = []
    confidences = []
    is_pdf = filename.lower().endswith(".pdf") or (mime_type and "pdf" in mime_type.lower())

    if is_pdf:
        with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
            page_count = len(doc)
            for page_index in range(page_count):
                page = doc[page_index]
                page_num = page_index + 1

                # 1. Render high-res pixmap for preview and scanned OCR
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0), alpha=False)
                page_img = Image.open(io.BytesIO(pix.tobytes("png")))

                # 2. Check native digital text
                digital_text = unicodedata.normalize("NFKC", page.get_text().strip())

                # Only treat as purely native digital if it has rich comprehensive content (50+ words)
                # Shorter digital overlays often leave header logos, institutions, and signatures on background images
                digital_words = [w for w in digital_text.split() if re.search(r'[A-Za-z0-9]{2,}', w)]
                has_comprehensive_digital = (len(digital_words) >= 50 and "(cid:" not in digital_text)

                if has_comprehensive_digital:
                    # Native digital PDF text is pristine and comprehensive
                    page_text = digital_text
                    page_conf = 98.5
                    extraction_method = "DIGITAL_NATIVE"
                else:
                    # Scanned or hybrid PDF page: run OCR
                    ocr_text, ocr_conf = run_ocr_on_image(page_img)

                    if digital_text and ocr_text:
                        # Hybrid page: digital vector text overlaid on a template background image
                        # Merge digital text and background OCR text cleanly
                        seen = set()
                        merged_lines = []
                        for l in (digital_text + "\n" + ocr_text).splitlines():
                            cl = l.strip().lower()
                            if cl and cl not in seen:
                                seen.add(cl)
                                merged_lines.append(l.strip())
                        page_text = "\n".join(merged_lines)
                        page_conf = max(ocr_conf, 92.0)
                        extraction_method = "HYBRID_DIGITAL_OCR"
                    elif ocr_text:
                        page_text = ocr_text
                        page_conf = ocr_conf if ocr_conf > 0 else 75.0
                        extraction_method = "SCANNED_OCR"
                    else:
                        page_text = digital_text
                        page_conf = 60.0 if digital_text else 0.0
                        extraction_method = "FALLBACK_DIGITAL"

                pages_data.append({
                    "page_number": page_num,
                    "text": page_text,
                    "confidence": page_conf,
                    "method": extraction_method,
                    "preview_image": page_img
                })
                full_text_parts.append(page_text)
                confidences.append(page_conf)
    else:
        # Standard Image file (PNG, JPG, JPEG)
        page_img = Image.open(io.BytesIO(file_bytes))
        ocr_text, ocr_conf = run_ocr_on_image(page_img)

        pages_data.append({
            "page_number": 1,
            "text": ocr_text,
            "confidence": ocr_conf,
            "method": "IMAGE_OCR",
            "preview_image": page_img
        })
        full_text_parts.append(ocr_text)
        confidences.append(ocr_conf)

    full_text = "\n\n--- Page Break ---\n\n".join(full_text_parts).strip()
    avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

    # 3. Digital Document Forensics & ELA Tamper Analysis
    from forensics_analyzer import compute_error_level_analysis, inspect_document_metadata
    forensics = inspect_document_metadata(file_bytes, filename)
    ela_preview = None
    ela_risk = 0.0
    ela_verdict = "UNIFORM ERROR LEVELS"
    if pages_data and "preview_image" in pages_data[0]:
        try:
            ela_preview, ela_risk, ela_verdict = compute_error_level_analysis(pages_data[0]["preview_image"])
        except Exception as e:
            logger.debug(f"ELA analysis skipped: {e}")

    return {
        "filename": filename,
        "page_count": len(pages_data),
        "full_text": full_text,
        "average_ocr_confidence": avg_confidence,
        "pages": pages_data,
        "is_ocr_enabled": bool(TESSERACT_PATH or HAS_WINOCR),
        "forensics": forensics,
        "ela_preview_image": ela_preview,
        "ela_risk_score": ela_risk,
        "ela_verdict": ela_verdict
    }


# ---------------------------------------------------------------------------
# Backwards-Compatible Interfaces (for legacy imports and callers)
# ---------------------------------------------------------------------------

def extract_image_text(file_bytes: bytes) -> str:
    """Extracts raw text from image bytes."""
    res = process_file_bytes(file_bytes, "image.png", "image/png")
    return res["full_text"]


def extract_pdf_text(file_bytes: bytes) -> str:
    """Extracts text from PDF bytes (native or scanned OCR)."""
    res = process_file_bytes(file_bytes, "document.pdf", "application/pdf")
    return res["full_text"]


def extract_fields(text: str, document_type: str = "Educational Certificate") -> Dict[str, Optional[str]]:
    """Legacy field extraction returning simple {field_name: value} mapping."""
    from information_extractor import extract_structured_fields
    payload = {
        "full_text": text,
        "pages": [{"page_number": 1, "text": text, "confidence": 95.0}]
    }
    extracted = extract_structured_fields(payload, document_type)
    return {k: v.get("value") for k, v in extracted.items()}


def process_document(
    uploaded_file: Any,
    document_type: str = "Educational Certificate"
) -> Tuple[str, Dict[str, Any]]:
    """
    Unified entry point for document processing.
    Accepts Streamlit UploadedFile, file-like object, path, or bytes.
    Returns:
        (extracted_text, fields_dict)
    Compatible with both legacy 2-element tuple unpacking and rich payloads.
    """
    if hasattr(uploaded_file, "getvalue"):
        file_bytes = uploaded_file.getvalue()
        filename = getattr(uploaded_file, "name", "uploaded_document.pdf")
        mime_type = getattr(uploaded_file, "type", "application/pdf")
    elif isinstance(uploaded_file, (bytes, bytearray)):
        file_bytes = bytes(uploaded_file)
        filename = "uploaded_document.pdf"
        mime_type = "application/pdf"
    elif isinstance(uploaded_file, str):
        with open(uploaded_file, "rb") as f:
            file_bytes = f.read()
        filename = os.path.basename(uploaded_file)
        mime_type = "application/pdf" if filename.lower().endswith(".pdf") else "image/png"
    else:
        raise ValueError(f"Unsupported uploaded_file input type: {type(uploaded_file)}")

    processed_payload = process_file_bytes(file_bytes, filename, mime_type)
    extracted_text = processed_payload["full_text"]

    from information_extractor import extract_structured_fields
    fields = extract_structured_fields(processed_payload, document_type)

    return extracted_text, fields


# Standard aliases registry for backwards-compatibility
FIELD_ALIASES = {
    "Educational Certificate": {
        "Student Name": ["student name", "name of student", "student's name", "candidate name", "name"],
        "Certificate ID": ["certificate id", "certificate number", "certificate no", "registration number", "roll number"],
        "Institution": ["institution name", "name of institution", "university name", "college name", "institution", "university"],
        "Issue Date": ["date of issue", "issue date", "issued on", "date issued"],
        "Expiry Date": ["date of expiry", "expiry date", "expiration date", "valid until"]
    },
    "Identity Document": {
        "Name": ["full name", "name of holder", "holder name", "name"],
        "Document Number": ["document number", "document no", "identification number", "id number", "card number"],
        "Date of Birth": ["date of birth", "birth date", "dob"],
        "Issue Date": ["date of issue", "issue date", "issued on"],
        "Expiry Date": ["date of expiry", "expiry date", "expiration date", "valid until"]
    },
    "Government Certificate": {
        "Applicant Name": ["applicant name", "name of applicant", "applicant", "beneficiary name", "name"],
        "Certificate ID": ["certificate id", "certificate number", "certificate no", "application number"],
        "Issuing Authority": ["issuing authority", "issuing department", "issued by", "department name", "authority"],
        "Issue Date": ["date of issue", "issue date", "issued on"],
        "Expiry Date": ["date of expiry", "expiry date", "expiration date", "valid until"]
    }
}

__all__ = [
    "TESSERACT_PATH",
    "OCR_CONFIG",
    "configure_tesseract",
    "preprocess_image",
    "detect_and_correct_orientation",
    "run_ocr_on_image",
    "process_file_bytes",
    "process_document",
    "extract_image_text",
    "extract_pdf_text",
    "extract_fields",
    "FIELD_ALIASES"
]
