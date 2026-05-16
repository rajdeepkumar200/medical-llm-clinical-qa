"""Lightweight OCR utilities for lab-report uploads.

Supports common image formats via Tesseract and PDFs via PyMuPDF (with
Tesseract fallback for image-only PDF pages).

All imports are deferred so the app can still start if the OCR libraries
or the system Tesseract binary are missing — in that case
``extract_text_from_file`` returns an empty string and the caller falls
back to the previous "model can't see the image" behaviour.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Cap the amount of text we feed into the prompt so a long multi-page PDF
# doesn't blow past the model's context window.
MAX_OCR_CHARS = 4000

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".gif"}
PDF_EXTS = {".pdf"}


def _clean(text: str) -> str:
    """Collapse whitespace and trim to MAX_OCR_CHARS."""
    if not text:
        return ""
    # Normalise whitespace: keep newlines, collapse runs of spaces.
    lines = [" ".join(ln.split()) for ln in text.splitlines()]
    cleaned = "\n".join(ln for ln in lines if ln.strip())
    if len(cleaned) > MAX_OCR_CHARS:
        cleaned = cleaned[:MAX_OCR_CHARS] + "\n…[truncated]"
    return cleaned


def _ocr_image(path: str) -> str:
    """Run Tesseract on a single image. Returns "" on any failure."""
    try:
        import pytesseract  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover - depends on env
        logger.warning(f"OCR libs not available: {e}")
        return ""
    try:
        with Image.open(path) as img:
            # Convert exotic modes (RGBA, P, CMYK) to RGB for Tesseract.
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            text = pytesseract.image_to_string(img)
        return _clean(text)
    except Exception as e:
        logger.warning(f"Image OCR failed for {path}: {e}")
        return ""


def _ocr_pdf(path: str) -> str:
    """Extract text from a PDF. Uses embedded text first; falls back to
    Tesseract on rendered pages when a page has no extractable text.
    """
    try:
        import fitz  # type: ignore[import-not-found]  # PyMuPDF
    except Exception as e:  # pragma: no cover
        logger.warning(f"PyMuPDF not available: {e}")
        return ""

    parts: list[str] = []
    try:
        with fitz.open(path) as doc:
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text("text") or ""
                if len(page_text.strip()) < 20:
                    # Likely a scanned page — OCR the rendered image.
                    try:
                        import pytesseract  # type: ignore[import-not-found]
                        from PIL import Image  # type: ignore[import-not-found]
                        import io

                        pix = page.get_pixmap(dpi=200)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        page_text = pytesseract.image_to_string(img)
                    except Exception as e:
                        logger.warning(
                            f"PDF page {page_num} OCR fallback failed: {e}"
                        )
                parts.append(f"--- Page {page_num} ---\n{page_text.strip()}")
                if sum(len(p) for p in parts) > MAX_OCR_CHARS:
                    break
    except Exception as e:
        logger.warning(f"PDF OCR failed for {path}: {e}")
        return ""

    return _clean("\n".join(parts))


def extract_text_from_file(file_path: str | None) -> str:
    """Public entrypoint. Returns extracted text (possibly empty)."""
    if not file_path:
        return ""
    try:
        suffix = Path(str(file_path)).suffix.lower()
    except Exception:
        return ""

    if suffix in IMAGE_EXTS:
        return _ocr_image(str(file_path))
    if suffix in PDF_EXTS:
        return _ocr_pdf(str(file_path))
    logger.info(f"Unsupported OCR file type: {suffix}")
    return ""
