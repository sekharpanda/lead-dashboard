# ---------- BEGIN PATCH: robust OCR imports & helpers ----------
import os
import re
import io
import logging
from typing import Optional, List

import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import requests

logger = logging.getLogger("lead-dashboard-ocr")

# Try to import pytesseract + cv2 (best local OCR). If unavailable, we'll fall back to OCR.space.
USE_PYTESSERACT = False
USE_OPENCV = False
try:
    import pytesseract
    USE_PYTESSERACT = True
    logger.info("pytesseract available - will use local OCR where possible.")
except Exception:
    logger.warning("pytesseract not available. Will try OCR.space API fallback if configured.")

try:
    import cv2
    USE_OPENCV = True
except Exception:
    logger.warning("cv2 (OpenCV) not available - some preprocessing steps will be skipped for local OCR.")

# OCR.space API config (optional)
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "")  # set this in Streamlit Secrets or env
OCR_SPACE_URL = "https://api.ocr.space/parse/image"

# ---------- OCR HELPERS ----------
def parse_spend_string(s: str) -> float:
    if not isinstance(s, str):
        s = str(s)
    t = re.sub(r"[₹$€£,]|AED|INR|usd|rs\.?", "", s, flags=re.IGNORECASE)
    t = re.sub(r"[^0-9\.\-]", "", t)
    t = re.sub(r"\.(?=.*\.)", "", t)
    try:
        return float(t) if t != "" else 0.0
    except Exception:
        return 0.0

def preprocess_image_for_ocr(pil_img: Image.Image):
    """Convert to grayscale and apply simple thresholding. Works only if cv2 is available."""
    if not USE_OPENCV:
        return pil_img
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)
    blur = cv2.medianBlur(th, 3)
    return blur

def ocr_with_pytesseract(pil_img: Image.Image) -> str:
    """Run local OCR with pytesseract (requires pytesseract installed and Tesseract binary available)."""
    if not USE_PYTESSERACT:
        raise RuntimeError("pytesseract not available")
    proc = preprocess_image_for_ocr(pil_img)
    try:
        text = pytesseract.image_to_string(proc, lang='eng')
    except Exception as e:
        logger.exception("pytesseract OCR failed: %s", e)
        text = ""
    return text

def ocr_with_ocr_space(pil_img: Image.Image, api_key: str) -> str:
    """Call OCR.space API. Requires network and API key. Returns concatenated parsed text."""
    if not api_key:
        raise RuntimeError("OCR.space API key not provided")
    try:
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        buffered.seek(0)
        files = {"filename": ("image.png", buffered, "image/png")}
        data = {"apikey": api_key, "language": "eng", "isOverlayRequired": False}
        resp = requests.post(OCR_SPACE_URL, files=files, data=data, timeout=60)
        resp.raise_for_status()
        j = resp.json()
        parsed = []
        if j.get("IsErroredOnProcessing"):
            logger.warning("OCR.space error: %s", j.get("ErrorMessage"))
            return ""
        for result in j.get("ParsedResults", []):
            parsed.append(result.get("ParsedText", ""))
        return "\n".join(parsed)
    except Exception as e:
        logger.exception("OCR.space call failed: %s", e)
        return ""

def ocr_image_to_text(pil_img: Image.Image) -> str:
    """
    Unified OCR entry point:
     - prefer pytesseract (local) if available
     - else try OCR.space if API key present
     - else return empty string (OCR disabled)
    """
    text = ""
    if USE_PYTESSACT:
        try:
            text = ocr_with_pytesseract(pil_img)
            if text and len(text.strip()) > 10:
                return text
        except Exception:
            logger.warning("pytesseract attempt failed; falling back if possible.")

    if OCR_SPACE_API_KEY:
        text = ocr_with_ocr_space(pil_img, OCR_SPACE_API_KEY)
        if text and len(text.strip()) > 0:
            return text

    # final fallback: no OCR available
    return ""
# ---------- END PATCH ----------
