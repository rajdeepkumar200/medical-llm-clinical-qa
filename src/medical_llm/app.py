from __future__ import annotations

import os
import logging
import html
import json
import re
from pathlib import Path
from typing import Generator

import gradio as gr  # type: ignore[import-not-found]

from .config import BASE_MODEL, ADAPTER_REPO_ID, ADAPTER_DIR, REGION
from .infer import generate_answer, load_model, load_tokenizer, stream_answer
from .nlp_processor import process_user_input, build_context_prompt
from .ocr import extract_text_from_file
from .web_context import fetch_wiki_context, should_ground

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = None
TOKENIZER = None
CURRENT_REGION = REGION

REGIONS = [
    "General",
    "United States",
    "United Kingdom",
    "Canada",
    "Australia",
    "New Zealand",
    "India",
    "Singapore",
    "Hong Kong",
    "Other",
]

COUNTRY_BOUNDS = [
    ("Singapore", 1.1, 1.5, 103.6, 104.1),
    ("Hong Kong", 22.1, 22.6, 113.8, 114.5),
    ("New Zealand", -48.5, -33.5, 165.5, 179.5),
    ("Australia", -44.0, -10.0, 112.0, 154.5),
    ("India", 6.0, 38.0, 68.0, 98.0),
    ("United Kingdom", 49.0, 61.0, -9.0, 2.5),
    ("Canada", 42.0, 84.0, -141.5, -52.0),
    ("United States", 18.0, 72.0, -171.0, -66.0),
]

TIMEZONE_TO_REGION = {
    "Asia/Kolkata": "India",
    "Asia/Calcutta": "India",
    "Europe/London": "United Kingdom",
    "America/New_York": "United States",
    "America/Chicago": "United States",
    "America/Denver": "United States",
    "America/Los_Angeles": "United States",
    "America/Phoenix": "United States",
    "America/Anchorage": "United States",
    "Pacific/Honolulu": "United States",
    "America/Toronto": "Canada",
    "America/Vancouver": "Canada",
    "America/Winnipeg": "Canada",
    "America/Halifax": "Canada",
    "Australia/Sydney": "Australia",
    "Australia/Melbourne": "Australia",
    "Australia/Brisbane": "Australia",
    "Australia/Perth": "Australia",
    "Pacific/Auckland": "New Zealand",
    "Asia/Singapore": "Singapore",
    "Asia/Hong_Kong": "Hong Kong",
}


def _region_label(region: str, source: str = "selected") -> str:
    region = region if region in REGIONS else "General"
    if region == "General":
        return "📍 Country: **General** · select country for medicine names"
    source_text = {
        "gps": "GPS",
        "timezone": "browser timezone",
        "selected": "selected",
    }.get(source, "selected")
    return f"📍 Country: **{region}** · {source_text}"


def _region_from_coordinates(lat: float, lon: float) -> str:
    for region, min_lat, max_lat, min_lon, max_lon in COUNTRY_BOUNDS:
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            return region
    return "General"


def detect_region_from_browser_payload(payload: str | None) -> tuple[str, str]:
    if not payload:
        return "General", "selected"
    try:
        data = json.loads(payload)
    except Exception as e:
        logger.warning(f"Browser region payload parse failed: {e}")
        return "General", "selected"

    try:
        lat = data.get("lat")
        lon = data.get("lon")
        if lat is not None and lon is not None:
            region = _region_from_coordinates(float(lat), float(lon))
            if region != "General":
                return region, "gps"
    except Exception as e:
        logger.warning(f"GPS region detection failed: {e}")

    timezone = str(data.get("timezone") or "")
    region = TIMEZONE_TO_REGION.get(timezone, "General")
    if region != "General":
        return region, "timezone"
    return "General", "selected"


EXAMPLE_PROMPTS = [
    "I have a sore throat and mild fever for 2 days. What should I do?",
    "Explain my lab report values: hemoglobin 10.2, ferritin 8, TSH 5.4.",
    "What does high LDL cholesterol mean and how can I lower it?",
    "Persistent headache for a week — what could it be?",
]


# --- Model pipeline ---------------------------------------------------------

def get_pipeline():
    global MODEL, TOKENIZER
    if MODEL is None or TOKENIZER is None:
        try:
            logger.info(f"Loading tokenizer for {BASE_MODEL}...")
            TOKENIZER = load_tokenizer(BASE_MODEL)
            adapter_source = (
                ADAPTER_REPO_ID
                if ADAPTER_REPO_ID
                else (str(ADAPTER_DIR) if ADAPTER_DIR.exists() else None)
            )
            logger.info(f"Loading model from {BASE_MODEL} with adapter: {adapter_source}...")
            MODEL = load_model(BASE_MODEL, adapter_source)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading pipeline: {e}", exc_info=True)
            raise
    return MODEL, TOKENIZER


def generate_response(message: str, region: str) -> Generator[str, None, None]:
    """Stream the model response chunk-by-chunk.

    For short / underspecified queries (e.g. ``"diabetic symptoms"``) the 1B
    base model has nothing to anchor on and tends to hallucinate. We:
      1. Look up the best-matching Wikipedia page and inject its summary as
         grounding context in the prompt.
      2. After streaming finishes, append a deterministic "Did you mean ...?"
         clarifier so the user can confirm the topic.
    """
    if not message.strip():
        yield ""
        return

    try:
        model, tokenizer = get_pipeline()
        nlp_result = process_user_input(message)
        enhanced_message = build_context_prompt(nlp_result, message)

        # ---- Optional grounding via Wikipedia for short queries -----------
        wiki_extract: str | None = None
        wiki_title: str | None = None
        if should_ground(message):
            try:
                wiki_extract, wiki_title = fetch_wiki_context(message)
            except Exception as e:
                logger.warning(f"Wikipedia grounding failed: {e}")

        if wiki_extract and wiki_title:
            logger.info(f"Grounding with Wikipedia page: {wiki_title}")
            enhanced_message = (
                f"{enhanced_message}\n\n"
                f"--- REFERENCE (Wikipedia: \"{wiki_title}\") ---\n"
                f"{wiki_extract}\n"
                f"--- END REFERENCE ---\n\n"
                "Use ONLY the reference above and well-known medical knowledge "
                "to answer the user's question. Stay on-topic with the reference."
            )

        logger.info(f"Streaming answer (Region: {region})")
        for chunk in stream_answer(model, tokenizer, enhanced_message, region=region):
            yield chunk

        # ---- Deterministic clarifier (so it cannot be forgotten / hallucinated)
        if wiki_title:
            yield (
                "\n\n---\n"
                f"*Did you mean **{wiki_title}**? "
                "If not, please rephrase with more detail.*"
            )
        logger.info("Stream finished")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        yield f"⚠️ **Error:** {str(e)[:200]}"


def update_region(region: str):
    global CURRENT_REGION
    CURRENT_REGION = region
    logger.info(f"Region updated to: {region}")


# --- UI helpers -------------------------------------------------------------

def _attachment_display_name(attachment) -> str:
    if isinstance(attachment, dict):
        return str(attachment.get("name") or "uploaded file")
    try:
        return Path(str(attachment)).name
    except Exception:
        return "uploaded file"


def _attachment_ocr_text(attachment) -> str:
    if isinstance(attachment, dict):
        return str(attachment.get("ocr_text") or "")
    if not attachment:
        return ""
    try:
        return extract_text_from_file(str(attachment))
    except Exception as e:
        logger.warning(f"OCR error for {_attachment_display_name(attachment)}: {e}")
        return ""


MEDICAL_REPORT_TERMS = (
    "lab report", "laboratory", "pathology", "diagnostic", "patient", "specimen",
    "collected", "reported", "reference range", "normal range", "result", "unit",
    "prescription", "rx", "doctor", "hospital", "clinic", "clinical", "diagnosis",
    "discharge summary", "medicine", "medication", "dose", "dosage", "tablet",
    "capsule", "injection",
    "flag", "blood", "serum", "plasma", "urine", "cbc", "complete blood count",
    "hemoglobin", "haemoglobin", "wbc", "rbc", "platelet", "hematocrit", "mcv",
    "mch", "mchc", "neutrophil", "lymphocyte", "eosinophil", "glucose", "hba1c",
    "cholesterol", "hdl", "ldl", "triglyceride", "creatinine", "urea", "bun",
    "egfr", "bilirubin", "albumin", "globulin", "protein", "alt", "ast", "sgpt",
    "sgot", "alkaline phosphatase", "tsh", "t3", "t4", "ferritin", "vitamin d",
    "sodium", "potassium", "chloride", "calcium", "crp", "esr",
)

MEDICAL_UNITS_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg/dl|g/dl|mmol/l|µmol/l|umol/l|iu/l|u/l|ng/ml|pg/ml|"
    r"miu/l|ml/min|cells?/u?l|/cumm|mg|mcg|µg|ml|%|x\s*10\^?\d+/l)\b",
    re.IGNORECASE,
)


def _is_probable_medical_report(text: str) -> bool:
    cleaned = (text or "").strip()
    if len(cleaned) < 30:
        return False

    lower = cleaned.lower()
    hits = sum(1 for term in MEDICAL_REPORT_TERMS if term in lower)
    unit_hits = len(MEDICAL_UNITS_PATTERN.findall(lower))
    table_words = sum(1 for term in ("test", "result", "unit", "range", "reference") if term in lower)

    if hits >= 4:
        return True
    if hits >= 2 and unit_hits >= 1:
        return True
    if table_words >= 3 and unit_hits >= 1:
        return True
    return False


def _attachment_is_medical_report(attachment) -> bool:
    if isinstance(attachment, dict):
        return bool(attachment.get("is_medical_report"))
    return _is_probable_medical_report(_attachment_ocr_text(attachment))


def _mentions_uploaded_file(message: str) -> bool:
    text = (message or "").lower()
    return any(
        phrase in text
        for phrase in (
            "uploaded",
            "upload",
            "attached",
            "attachment",
            "image",
            "photo",
            "picture",
            "lab report",
            "report image",
        )
    )


def _render_attach_chip(
    name: str,
    ocr_text: str,
    processing: bool = False,
    is_medical_report: bool | None = None,
) -> str:
    safe_name = html.escape(name)
    if processing:
        return (
            '<div class="attach-card attach-processing">'
            '<span class="attach-icon">📎</span>'
            '<span class="attach-main">'
            f'<strong>{safe_name}</strong>'
            '<em>Uploading and checking the image…</em>'
            '</span>'
            '<span class="attach-spinner">●</span>'
            '</div>'
        )
    if ocr_text.strip():
        preview = html.escape(ocr_text.replace("\n", " ")[:120])
        if is_medical_report is False:
            return (
                '<div class="attach-card attach-warn">'
                '<span class="attach-icon">📎</span>'
                '<span class="attach-main">'
                f'<strong>{safe_name}</strong>'
                f'<em>Uploaded · OCR read {len(ocr_text)} characters · not a medical report</em>'
                f'<small>{preview}</small>'
                '</span>'
                '<span class="attach-check">!</span>'
                '</div>'
            )
        return (
            '<div class="attach-card attach-ok">'
            '<span class="attach-icon">📎</span>'
            '<span class="attach-main">'
            f'<strong>{safe_name}</strong>'
            f'<em>Uploaded · medical report detected · OCR read {len(ocr_text)} characters</em>'
            f'<small>{preview}</small>'
            '</span>'
            '<span class="attach-check">✓</span>'
            '</div>'
        )
    return (
        '<div class="attach-card attach-warn">'
        '<span class="attach-icon">📎</span>'
        '<span class="attach-main">'
        f'<strong>{safe_name}</strong>'
        '<em>Uploaded · OCR could not read text clearly</em>'
        '<small>You can still ask, but typing key lab values will help.</small>'
        '</span>'
        '<span class="attach-check">!</span>'
        '</div>'
    )


def _attach_file_note(message: str, attachment) -> str:
    if not attachment:
        return message

    name = _attachment_display_name(attachment)
    extracted = _attachment_ocr_text(attachment)

    base_msg = message or "Please review my attached lab report and explain the findings."

    if extracted.strip():
        logger.info(f"OCR extracted {len(extracted)} chars from {name}")
        if not _attachment_is_medical_report(attachment):
            return (
                "The user uploaded an image/PDF, and OCR extracted text from it, but the extracted "
                "text does not look like a medical lab report or clinical report. Do not summarize it "
                "as a lab report. Do not invent test results, symptoms, diagnoses, medicines, or values.\n\n"
                f"User request: {base_msg}\n\n"
                f"Uploaded file: `{name}`\n\n"
                "Tell the user that the uploaded file does not appear to be a medical/lab report. "
                "Ask them to upload a clear lab report, prescription, discharge summary, or type the "
                "medical values they want explained."
            )
        return (
            "The user uploaded an image/PDF lab report. You cannot see pixels directly, "
            "but OCR has extracted the text below. Treat this OCR text as the uploaded file content. "
            "Do not say you do not have access to the image.\n\n"
            f"User request: {base_msg}\n\n"
            f"--- BEGIN OCR TEXT FROM UPLOADED FILE `{name}` ---\n"
            f"{extracted}\n"
            f"--- END OCR TEXT FROM UPLOADED FILE `{name}` ---\n\n"
            "Answer using the OCR text above. If values are visible, list them, flag likely abnormal "
            "ones cautiously, explain what they may mean in plain English, mention red flags, and "
            "recommend confirming with a clinician. If OCR text is messy, say which parts are unclear."
        )

    return (
        "The user uploaded an image/PDF, but OCR could not extract readable text from it. "
        "Do not claim the upload failed. Tell the user the file was received, explain that the text "
        "could not be read clearly, and ask them to type the key lab values or upload a clearer image.\n\n"
        f"User request: {base_msg}\n\n"
        f"Uploaded file: `{name}`"
    )


def _render_message_html(role: str, text: str, attachment_name: str | None = None) -> str:
    """Render a single message bubble. role is 'user' or 'assistant'."""
    # Gradio Markdown handles formatting; we wrap with a class for styling.
    css_class = "msg-user" if role == "user" else "msg-assistant"
    avatar = "🧑" if role == "user" else "🩺"
    label = "You" if role == "user" else "Clinical Assistant"
    attach_html = ""
    if attachment_name:
        attach_html = (
            f'<div class="attach-chip">📎 {attachment_name}</div>'
        )
    # Note: we keep raw markdown inside; the message stream is wrapped in a
    # gr.Markdown component so Markdown will render bullets/headings nicely.
    return (
        f'<div class="bubble {css_class}">'
        f'<div class="bubble-head"><span class="avatar">{avatar}</span>'
        f'<span class="who">{label}</span></div>'
        f'{attach_html}'
        f'<div class="bubble-body">\n\n{text}\n\n</div>'
        f"</div>"
    )


def _render_conversation(history: list[dict]) -> str:
    if not history:
        return ""
    parts = [
        _render_message_html(
            m["role"], m["content"], m.get("attachment")
        )
        for m in history
    ]
    return "\n\n".join(parts)


# --- Custom CSS (Claude-inspired) ------------------------------------------

CUSTOM_CSS = """
:root {
    --bg: #0a0a0b;
    --sidebar: #131316;
    --panel: #1a1a1d;
    --panel-2: #232328;
    --ink: #ededed;
    --ink-soft: #9a9a9a;
    --accent: #c96442;
    --accent-soft: rgba(201, 100, 66, 0.18);
    --border: #2a2a2f;
    --user-bg: #2a1f1a;
    --assistant-bg: #1a1a1d;
}

html, body, .gradio-container {
    background: var(--bg) !important;
    color: var(--ink) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    min-height: 100vh;
}
.gradio-container > .main, .gradio-container > div { padding: 0 !important; }
* { box-sizing: border-box; }

footer { display: none !important; }

/* ----- App shell with sidebar ----- */
#app-shell {
    display: flex;
    flex-direction: row;
    min-height: 100vh;
    align-items: stretch;
    gap: 0 !important;
    width: 100%;
}
#sidebar {
    background: var(--sidebar) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 240px !important;
    max-width: 260px !important;
    flex: 0 0 240px !important;
    padding: 20px 14px !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 8px !important;
}
.brand {
    display: flex; align-items: center; gap: 8px;
    font-size: 1.05rem; font-weight: 600; color: var(--ink);
    padding: 6px 4px 18px 4px;
    border-bottom: 1px solid var(--border);
}
.new-chat-pill button, button.new-chat-pill {
    background: var(--panel-2) !important;
    color: var(--ink) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    width: 100% !important;
    text-align: left !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    margin-top: 14px !important;
}
.new-chat-pill button:hover { background: var(--accent-soft) !important; color: var(--accent) !important; border-color: var(--accent) !important; }
.history-label {
    color: var(--ink-soft); font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.6px;
    margin: 20px 4px 8px 4px;
}
#sidebar-history {
    flex: 1 1 auto;
    overflow-y: auto;
    font-size: 0.85rem;
    color: var(--ink-soft);
    padding: 0 4px;
}
#sidebar-history p { margin: 6px 0 !important; line-height: 1.45; }
#sidebar-history strong { color: var(--ink); font-weight: 600; }
.hist-item {
    display: block;
    padding: 8px 10px;
    border-radius: 8px;
    margin: 2px 0;
    background: transparent;
    color: var(--ink-soft);
    font-size: 0.85rem;
    line-height: 1.35;
    border: 1px solid transparent;
}
.hist-item:hover { background: var(--panel-2); color: var(--ink); }

/* ----- Main pane ----- */
#main {
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
    background: var(--bg);
    min-height: 100vh;
    padding: 0 !important;
    overflow: hidden;
}
#topbar {
    flex: 0 0 auto;
    padding: 14px 24px !important;
    border-bottom: 1px solid var(--border);
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 16px;
}
.topbar-title {
    font-size: 1.02rem; font-weight: 600; color: var(--ink);
    display: flex; align-items: center; gap: 8px;
}
#region-badge {
    margin: 0 !important;
    text-align: right;
    font-size: 0.8rem;
    color: var(--ink-soft);
    min-width: 200px;
}
#region-badge p { margin: 0 !important; color: var(--ink-soft) !important; }
#region-badge strong { color: var(--accent); }
#country-select {
    min-width: 180px !important;
    max-width: 210px !important;
}
#country-select label { display: none !important; }
#country-select .wrap,
#country-select input,
#country-select select {
    background: var(--panel-2) !important;
    color: var(--ink) !important;
    border-color: var(--border) !important;
    border-radius: 999px !important;
    font-size: 0.82rem !important;
}

#content-area {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: 24px 24px 12px 24px;
    display: flex;
    flex-direction: column;
}

/* ----- Welcome hero (shown when no messages yet) ----- */
#welcome-block {
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 40vh;
}
.hero { text-align: center; max-width: 600px; margin-bottom: 32px; padding: 0 16px; }
.hero h1 { font-size: 2rem; font-weight: 600; color: var(--ink); margin: 0 0 10px 0; }
.hero p { color: var(--ink-soft); font-size: 1rem; margin: 0; line-height: 1.5; }

#examples-row {
    max-width: 700px; width: 100%; margin: 0 auto;
    display: grid !important;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}
.example-card button, button.example-card {
    background: var(--panel) !important;
    color: var(--ink) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    text-align: left !important;
    padding: 14px 16px !important;
    font-weight: 400 !important;
    font-size: 0.9rem !important;
    line-height: 1.45 !important;
    white-space: normal !important;
    height: auto !important;
    transition: all 0.15s ease;
}
.example-card button:hover, button.example-card:hover {
    background: var(--panel-2) !important;
    border-color: var(--accent) !important;
    color: var(--ink) !important;
}

/* ----- Chat messages ----- */
#chat-stream { padding-bottom: 12px; }
.bubble {
    border-radius: 14px;
    padding: 14px 18px;
    margin: 12px 0;
    border: 1px solid var(--border);
    max-width: 820px;
}
.bubble.msg-user {
    background: var(--user-bg);
    margin-left: auto; margin-right: 0;
}
.bubble.msg-assistant {
    background: var(--assistant-bg);
    margin-right: auto; margin-left: 0;
}
.bubble-head { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.bubble-head .avatar { font-size: 1.05rem; }
.bubble-head .who { font-weight: 600; color: var(--ink-soft); font-size: 0.8rem; }
.bubble-body { font-size: 0.95rem; line-height: 1.65; color: var(--ink); }
.bubble-body p { color: var(--ink); margin: 6px 0; }
.bubble-body h3 { font-size: 1rem; margin: 14px 0 6px 0; color: var(--ink); }
.bubble-body ul, .bubble-body ol { padding-left: 22px; margin: 6px 0; }
.bubble-body li { margin: 4px 0; color: var(--ink); }
.bubble-body strong { color: #fff; font-weight: 600; }
.bubble-body em { color: var(--ink-soft); }
.bubble-body code {
    background: var(--panel-2); color: #f5b893;
    padding: 1px 6px; border-radius: 4px; font-size: 0.9em;
}
.response-pill {
    display: inline-flex;
    align-items: center;
    border: 1px solid var(--accent);
    background: var(--accent-soft);
    color: var(--ink);
    border-radius: 999px;
    padding: 7px 14px;
    font-size: 0.86rem;
    font-weight: 500;
    animation: response-pill-pulse 1.4s ease-in-out infinite;
}
@keyframes response-pill-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.78; transform: scale(0.98); }
}
.attach-chip {
    display: inline-block; background: var(--accent-soft); color: var(--accent);
    border: 1px solid var(--accent);
    border-radius: 999px; padding: 2px 10px; font-size: 0.78rem; margin: 4px 0;
}

/* ----- Input bar (sticky at bottom of main pane) ----- */
#input-bar {
    flex: 0 0 auto;
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 8px 12px 6px 12px !important;
    margin: 0 24px 16px 24px;
    box-shadow: 0 -4px 24px rgba(0,0,0,0.3);
}
#input-bar textarea {
    background: transparent !important;
    color: var(--ink) !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 0.95rem !important;
    min-height: 44px !important;
    resize: none !important;
    padding: 8px !important;
}
#input-bar textarea::placeholder { color: var(--ink-soft) !important; }
#input-bar textarea:focus { outline: none !important; }

.actions-row {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    padding: 2px 4px !important;
    gap: 8px;
}
.icon-btn button, button.icon-btn {
    background: var(--panel-2) !important;
    color: var(--ink) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    width: 38px !important; height: 38px !important;
    min-width: 38px !important;
    padding: 0 !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
}
.icon-btn button:hover {
    background: var(--accent-soft) !important;
    color: var(--accent) !important;
    border-color: var(--accent) !important;
}
.send-btn button, button.send-btn {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 999px !important;
    width: 42px !important; height: 42px !important;
    min-width: 42px !important;
    padding: 0 !important;
    font-size: 1.05rem !important;
}
.send-btn button:hover { filter: brightness(1.1); }

/* ----- Inline attached-file chip (inside input-bar, above textarea) ----- */
.attach-chip-row {
    padding: 4px 4px 8px 4px !important;
    background: transparent !important; border: none !important;
}
.attach-chip-row p { margin: 0 !important; width: 100%; }
.attach-card {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    background: #20242b;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 10px 12px;
    color: var(--ink);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}
.attach-card strong {
    display: block;
    color: #fff;
    font-size: 0.88rem;
    font-weight: 600;
    line-height: 1.15;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.attach-card em {
    display: block;
    color: var(--ink-soft);
    font-size: 0.78rem;
    font-style: normal;
    margin-top: 2px;
}
.attach-card small {
    display: block;
    color: #b8b8b8;
    font-size: 0.72rem;
    margin-top: 4px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.attach-icon {
    width: 30px;
    height: 30px;
    flex: 0 0 30px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: var(--accent-soft);
    border-radius: 8px;
}
.attach-main {
    min-width: 0;
    flex: 1 1 auto;
}
.attach-ok {
    border-color: rgba(54, 211, 153, 0.55);
    background: rgba(54, 211, 153, 0.08);
}
.attach-warn {
    border-color: rgba(245, 158, 11, 0.65);
    background: rgba(245, 158, 11, 0.08);
}
.attach-processing {
    border-color: rgba(201, 100, 66, 0.75);
    background: rgba(201, 100, 66, 0.1);
}
.attach-check {
    width: 24px;
    height: 24px;
    flex: 0 0 24px;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    color: #07130f;
    background: #36d399;
}
.attach-warn .attach-check {
    color: #1f1300;
    background: #f59e0b;
}
.attach-spinner {
    color: var(--accent);
    animation: attach-pulse 1s ease-in-out infinite;
}
@keyframes attach-pulse {
    0%, 100% { opacity: 0.35; transform: scale(0.8); }
    50% { opacity: 1; transform: scale(1.05); }
}

/* ----- Hide Gradio's default progress overlay ----- */
.gradio-container .progress-text,
.gradio-container .progress-bar,
.gradio-container .wrap.default,
.gradio-container div[class*="progress"],
.gradio-container .eta-bar { display: none !important; }

/* ----- Disclaimer ----- */
.disclaimer {
    text-align: center;
    color: var(--ink-soft);
    font-size: 0.72rem;
    padding: 0 24px 14px 24px;
}

/* ----- Mobile responsiveness ----- */
@media (max-width: 768px) {
    #app-shell { flex-direction: column; }
    #sidebar {
        flex: 0 0 auto !important;
        max-width: none !important;
        min-width: 0 !important;
        border-right: none !important;
        border-bottom: 1px solid var(--border) !important;
        padding: 12px 16px !important;
    }
    #sidebar-history { max-height: 120px; }
    .hero h1 { font-size: 1.6rem; }
    #examples-row { grid-template-columns: 1fr; }
    #input-bar { margin: 0 12px 12px 12px; }
    #content-area { padding: 16px; }
}
"""


def _render_sidebar(history):
    """Render a compact list of past user messages for the sidebar."""
    user_msgs = [m for m in history if m.get("role") == "user"]
    if not user_msgs:
        return "_No messages yet. Start by asking a question._"
    items = []
    for i, m in enumerate(user_msgs, 1):
        content = (m.get("content") or "").strip().replace("\n", " ")
        preview = content[:48] + ("…" if len(content) > 48 else "")
        items.append(f'<div class="hist-item">{i}. {preview}</div>')
    return "\n".join(items)


def build_demo():
    """Dark UI with sidebar history. Single main pane: welcome content lives in
    the same area as the chat stream so the response replaces it (no scroll)."""
    with gr.Blocks(title="Clinical AI Assistant", css=CUSTOM_CSS, theme=gr.themes.Base()) as demo:

        # ----- State -----
        history_state = gr.State([])
        pending_file_state = gr.State(None)
        region_state = gr.State(REGION)
        browser_region_payload = gr.Textbox(visible=False)

        # ----- App shell: sidebar + main -----
        with gr.Row(elem_id="app-shell"):

            # ============ SIDEBAR ============
            with gr.Column(elem_id="sidebar", scale=0):
                gr.HTML('<div class="brand">🩺 Clinical AI</div>')
                new_chat_btn = gr.Button("＋ New chat", elem_classes=["new-chat-pill"])
                gr.HTML('<div class="history-label">Conversation</div>')
                sidebar_history = gr.Markdown(
                    "_No messages yet. Start by asking a question._",
                    elem_id="sidebar-history",
                    sanitize_html=False,
                )

            # ============ MAIN PANE ============
            with gr.Column(elem_id="main", scale=1):
                # Top bar
                with gr.Row(elem_id="topbar"):
                    gr.HTML('<div class="topbar-title">🩺 Clinical AI Assistant</div>')
                    region_badge = gr.Markdown(
                        "📍 Detecting your region…",
                        elem_id="region-badge",
                    )
                    country_select = gr.Dropdown(
                        choices=REGIONS,
                        value=REGION if REGION in REGIONS else "General",
                        label="Country",
                        elem_id="country-select",
                        container=True,
                    )

                # Content area: welcome hero OR chat messages (mutually exclusive)
                with gr.Column(elem_id="content-area"):
                    with gr.Column(visible=True, elem_id="welcome-block") as welcome_block:
                        gr.HTML(
                            """
                            <div class="hero">
                                <h1>What can I help you with?</h1>
                                <p>Ask about symptoms, medications, or lab reports — clear answers in plain English. Region-aware guidance, OCR-powered lab report reading.</p>
                            </div>
                            """
                        )
                        with gr.Row(elem_id="examples-row"):
                            example_btns = [
                                gr.Button(p, elem_classes=["example-card"]) for p in EXAMPLE_PROMPTS
                            ]

                    chat_display = gr.Markdown(
                        "",
                        elem_id="chat-stream",
                        sanitize_html=False,
                        visible=False,
                    )

                # Sticky input bar at bottom
                with gr.Column(elem_id="input-bar"):
                    attach_chip = gr.Markdown(
                        "",
                        elem_classes=["attach-chip-row"],
                        sanitize_html=False,
                        visible=False,
                    )
                    input_box = gr.Textbox(
                        placeholder="Ask anything medical — symptoms, medications, lab values…",
                        lines=1,
                        max_lines=1,
                        show_label=False,
                        container=False,
                    )
                    with gr.Row(elem_classes=["actions-row"]):
                        upload_btn = gr.UploadButton(
                            "+",
                            file_types=["image", ".pdf"],
                            type="filepath",
                            elem_classes=["icon-btn"],
                        )
                        send_btn = gr.Button("➤", elem_classes=["send-btn"])

                gr.HTML(
                    '<div class="disclaimer">⚠️ Educational use only. Always consult a qualified healthcare professional.</div>'
                )

        # =================== HANDLERS ===================
        def _upload_processing(file_obj=None):
            if file_obj is None:
                return gr.update(value=_render_attach_chip("upload", "", processing=True), visible=True)
            path = file_obj[0] if isinstance(file_obj, list) and file_obj else file_obj
            name = _attachment_display_name(path)
            return gr.update(value=_render_attach_chip(name, "", processing=True), visible=True)

        def on_upload(file_obj):
            empty = gr.update(value="", visible=False)
            if file_obj is None:
                return None, empty
            path = file_obj[0] if isinstance(file_obj, list) and file_obj else file_obj
            path = path.name if hasattr(path, "name") else str(path)
            display_name = _attachment_display_name(path)
            try:
                preview = extract_text_from_file(path)
            except Exception as e:
                logger.warning(f"OCR preview failed: {e}")
                preview = ""
            is_medical_report = _is_probable_medical_report(preview)
            attachment = {
                "path": path,
                "name": display_name,
                "ocr_text": preview,
                "ocr_chars": len(preview),
                "ocr_ok": bool(preview.strip()),
                "is_medical_report": is_medical_report,
            }
            return attachment, gr.update(
                value=_render_attach_chip(display_name, preview, is_medical_report=is_medical_report),
                visible=True,
            )

        def submit_message(message, region_val, history, pending_file, request: gr.Request):
            if region_val not in REGIONS or not region_val:
                region_val = "General"

            message = (message or "").strip()
            empty_chip = gr.update(value="", visible=False)

            if not message and not pending_file:
                yield (
                    gr.update(),  # welcome_block
                    gr.update(),  # chat_display
                    history,
                    "",  # input_box
                    None,  # pending_file
                    empty_chip,  # attach_chip
                    _render_sidebar(history),  # sidebar_history
                )
                return

            if message and not pending_file and _mentions_uploaded_file(message):
                history = list(history or [])
                history.append({
                    "role": "user",
                    "content": message,
                    "attachment": None,
                })
                history.append({
                    "role": "assistant",
                    "content": (
                        "I don't see an attached file for this message yet.\n\n"
                        "Please click the **+** button, choose your lab report image/PDF, "
                        "and wait until the attachment card says **Uploaded · OCR read ... characters**. "
                        "Then send your question again and I'll use the extracted report text."
                    ),
                    "attachment": None,
                })
                yield (
                    gr.update(visible=False),
                    gr.update(value=_render_conversation(history), visible=True),
                    history,
                    "",
                    None,
                    empty_chip,
                    _render_sidebar(history),
                )
                return

            if pending_file and _attachment_ocr_text(pending_file).strip() and not _attachment_is_medical_report(pending_file):
                attachment_name = _attachment_display_name(pending_file)
                history = list(history or [])
                history.append({
                    "role": "user",
                    "content": message or "(attached file)",
                    "attachment": attachment_name,
                })
                history.append({
                    "role": "assistant",
                    "content": (
                        "I received the uploaded file, but the OCR text does **not** look like a medical "
                        "lab report, prescription, discharge summary, or clinical document.\n\n"
                        "I won’t summarize it as a lab report or invent test values. Please upload a clear "
                        "medical report/lab report image, or type the medical values you want explained."
                    ),
                    "attachment": None,
                })
                yield (
                    gr.update(visible=False),
                    gr.update(value=_render_conversation(history), visible=True),
                    history,
                    "",
                    None,
                    empty_chip,
                    _render_sidebar(history),
                )
                return

            attachment_name = None
            full_msg = message
            if pending_file:
                attachment_name = _attachment_display_name(pending_file)
                full_msg = _attach_file_note(message, pending_file)

            history = list(history or [])
            history.append({
                "role": "user",
                "content": message or "(attached lab report)",
                "attachment": attachment_name,
            })
            # Reserve an empty assistant bubble that we'll fill as tokens stream in.
            history.append({
                "role": "assistant",
                "content": '<span class="response-pill">Generating response…</span>',
                "attachment": None,
            })

            # First yield: hide welcome, show chat with the user message + empty
            # assistant bubble. The user sees their message immediately.
            yield (
                gr.update(visible=False),  # welcome_block
                gr.update(value=_render_conversation(history), visible=True),
                history,
                "",  # clear input
                None,  # clear pending_file
                empty_chip,
                _render_sidebar(history),
            )

            # Stream tokens into the last assistant bubble.
            response_text = ""
            try:
                for chunk in generate_response(full_msg, region_val):
                    response_text += chunk
                    history[-1]["content"] = response_text + "▍"  # cursor caret
                    yield (
                        gr.update(visible=False),
                        gr.update(value=_render_conversation(history), visible=True),
                        history,
                        "",
                        None,
                        empty_chip,
                        _render_sidebar(history),
                    )
            except Exception as e:
                logger.error(f"Generation error: {e}", exc_info=True)
                response_text = f"⚠️ **Error:** {str(e)[:200]}"

            # Final yield: drop the cursor caret and finalize.
            history[-1]["content"] = response_text or "_(no response)_"
            yield (
                gr.update(visible=False),
                gr.update(value=_render_conversation(history), visible=True),
                history,
                "",
                None,
                empty_chip,
                _render_sidebar(history),
            )

        def reset_chat():
            empty_chip = gr.update(value="", visible=False)
            return (
                gr.update(visible=True),   # welcome_block visible
                gr.update(value="", visible=False),  # chat_display hidden
                [],   # history
                "",   # input
                None,  # pending_file
                empty_chip,  # attach_chip
                _render_sidebar([]),  # sidebar reset
            )

        def fill_example(text):
            return text

        # ----- Wire events -----
        submit_outputs = [
            welcome_block,
            chat_display,
            history_state,
            input_box,
            pending_file_state,
            attach_chip,
            sidebar_history,
        ]
        submit_inputs = [input_box, region_state, history_state, pending_file_state]
        prog = "hidden"

        (
            send_btn.click(submit_message, submit_inputs, submit_outputs, api_name=False, show_progress=prog)
        )
        (
            input_box.submit(submit_message, submit_inputs, submit_outputs, api_name=False, show_progress=prog)
        )

        (
            upload_btn.upload(
                fn=_upload_processing,
                inputs=[upload_btn],
                outputs=[attach_chip],
                api_name=False,
                show_progress=prog,
            )
            .then(
                fn=on_upload,
                inputs=[upload_btn],
                outputs=[pending_file_state, attach_chip],
                api_name=False,
                show_progress=prog,
            )
        )

        new_chat_btn.click(fn=reset_chat, outputs=submit_outputs, api_name=False, show_progress=prog)

        for btn in example_btns:
            btn.click(fn=fill_example, inputs=[btn], outputs=[input_box], api_name=False, show_progress=prog)

        def _select_region(region):
            region = region if region in REGIONS else "General"
            global CURRENT_REGION
            CURRENT_REGION = region
            return region, _region_label(region, "selected")

        def _init_region(payload):
            region, source = detect_region_from_browser_payload(payload)
            global CURRENT_REGION
            CURRENT_REGION = region
            label = _region_label(region, source)
            return region, label, gr.update(value=region)

        country_select.change(
            fn=_select_region,
            inputs=[country_select],
            outputs=[region_state, region_badge],
            api_name=False,
            show_progress=prog,
        )

        demo.load(
            fn=_init_region,
            inputs=[browser_region_payload],
            outputs=[region_state, region_badge, country_select],
            api_name=False,
            js="""
async () => {
  const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone || "";
  const payload = { timezone };
  if (!navigator.geolocation) {
    return [JSON.stringify(payload)];
  }
  return await new Promise((resolve) => {
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        payload.lat = pos.coords.latitude;
        payload.lon = pos.coords.longitude;
        resolve([JSON.stringify(payload)]);
      },
      () => resolve([JSON.stringify(payload)]),
      { enableHighAccuracy: false, timeout: 3500, maximumAge: 600000 }
    );
  });
}
""",
        )

    return demo


if __name__ == "__main__":
    in_space = bool(os.getenv("SPACE_ID") or os.getenv("SPACE_HOST"))
    # show_api=False avoids the gradio_client schema-generation bug that
    # crashed earlier on /info, and we don't expose this app as an API anyway.
    build_demo().queue().launch(
        share=False if in_space else True,
        show_api=False,
    )
