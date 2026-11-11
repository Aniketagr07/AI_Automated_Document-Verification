"""
Digital Document Forensics Module for SIH1709.
Performs:
1. Error Level Analysis (ELA) for image splice/tamper detection
2. PDF Metadata Provenance Inspection (detecting Photoshop, Canva, GIMP alterations)
3. Visual Tamper Heatmap generation
"""

import io
from typing import Dict, Any, Tuple
from PIL import Image, ImageChops, ImageEnhance
import numpy as np


def compute_error_level_analysis(
    image: Image.Image,
    resave_quality: int = 90,
    amplification: float = 14.0
) -> Tuple[Image.Image, float, str]:
    """
    Computes Error Level Analysis (ELA) on an image to detect digital splicing.
    When an image is modified (e.g. changing roll numbers or names) and resaved,
    the modified areas have different compression levels than the original.
    
    Returns:
        (ela_image, tamper_risk_score_0_to_100, assessment_verdict)
    """
    img_rgb = image.convert("RGB")

    # Resave image at standard quality
    buffer = io.BytesIO()
    img_rgb.save(buffer, "JPEG", quality=resave_quality)
    buffer.seek(0)
    resaved_img = Image.open(buffer)

    # Calculate absolute difference
    diff = ImageChops.difference(img_rgb, resaved_img)
    diff_arr = np.array(diff, dtype=np.float32)

    max_diff = float(np.max(diff_arr))
    mean_diff = float(np.mean(diff_arr))
    variance = float(np.var(diff_arr))

    # Enhance difference image for human review
    enhanced_diff = ImageEnhance.Brightness(diff).enhance(amplification)

    # Heuristic tamper metric
    # High localized variance relative to mean indicates spliced / copy-pasted artifacts
    risk_score = min(100.0, round((max_diff / 255.0) * 45.0 + (mean_diff * 4.0) + (min(variance, 100.0) * 0.15), 1))

    if risk_score > 65.0:
        verdict = "SUSPICIOUS COMPRESSION VARIANCE (Possible Image Tampering / Splicing)"
    elif risk_score > 40.0:
        verdict = "MODERATE VARIANCE (Secondary JPEG Compression or Re-encoding)"
    else:
        verdict = "UNIFORM ERROR LEVELS (Consistent Compression Surface)"

    return enhanced_diff, risk_score, verdict


def inspect_document_metadata(doc_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Analyzes document authoring tool, creator metadata, and modification dates.
    Flags suspicious tools (e.g. Canva, Adobe Photoshop, GIMP, ILovePDF) for official certificates.
    """
    metadata = {}
    suspicious_flags = []
    is_pdf = filename.lower().endswith(".pdf")

    if is_pdf:
        try:
            import pymupdf
            with pymupdf.open(stream=doc_bytes, filetype="pdf") as doc:
                meta = doc.metadata or {}
                producer = meta.get("producer", "")
                creator = meta.get("creator", "")
                author = meta.get("author", "")
                mod_date = meta.get("modDate", "")
                creation_date = meta.get("creationDate", "")

                metadata = {
                    "Producer": producer or "Not Specified",
                    "Creator": creator or "Not Specified",
                    "Author": author or "Not Specified",
                    "Creation Date": creation_date or "Not Specified",
                    "Modification Date": mod_date or "Not Specified",
                    "Page Count": len(doc)
                }

                suspicious_tools = [
                    "canva", "photoshop", "gimp", "ilovepdf", "pdfescape",
                    "coreldraw", "paint.net", "inkscape"
                ]
                combined = f"{producer} {creator}".lower()
                for tool in suspicious_tools:
                    if tool in combined:
                        suspicious_flags.append(
                            f"Document generated or modified using graphic design tool '{tool.title()}', atypical for official university/government gazettes."
                        )
        except Exception as e:
            metadata["Error"] = f"Failed to parse PDF metadata: {str(e)}"
    else:
        # Check EXIF for images
        try:
            img = Image.open(io.BytesIO(doc_bytes))
            exif = img.getexif()
            if exif:
                software = exif.get(0x0131, "")
                if software:
                    metadata["Software"] = software
                    if any(t in software.lower() for t in ["photoshop", "canva", "gimp"]):
                        suspicious_flags.append(f"Image edited with '{software}'.")
        except Exception:
            pass

    return {
        "metadata": metadata,
        "is_suspicious": len(suspicious_flags) > 0,
        "suspicious_reasons": suspicious_flags
    }
