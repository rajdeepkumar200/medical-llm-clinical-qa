from __future__ import annotations

import os
import logging
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

# ISO country code → user-friendly region name used by the prompt builder
COUNTRY_TO_REGION = {
    "US": "United States",
    "GB": "United Kingdom", "UK": "United Kingdom",
    "CA": "Canada",
    "AU": "Australia",
    "NZ": "New Zealand",
    "IN": "India",
    "SG": "Singapore",
    "HK": "Hong Kong",
}


def detect_region(request) -> str:
    """Auto-detect a user's region from the browser Accept-Language header.

    Falls back to "General" when nothing useful is available. We deliberately
    avoid external IP-geolocation calls to keep startup fast and offline-safe.
    """
    if request is None:
        return "General"
    try:
        headers = getattr(request, "headers", {}) or {}
        accept_lang = ""
        # gr.Request.headers can be a dict-like
        if hasattr(headers, "get"):
            accept_lang = headers.get("accept-language", "") or headers.get(
                "Accept-Language", ""
            )
        else:
            accept_lang = str(headers)
        accept_lang = accept_lang.upper()
        for code, region in COUNTRY_TO_REGION.items():
            if f"-{code}" in accept_lang or f"_{code}" in accept_lang:
                return region
    except Exception as e:
        logger.warning(f"Region detection failed: {e}")
    return "General"


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

def _attach_file_note(message: str, file_path) -> str:
    """Augment a message with OCR-extracted text from the uploaded file.

    The base model is text-only, so we OCR the lab report and inject the
    recognised text directly into the prompt. If OCR is unavailable or
    yields nothing usable, we fall back to a graceful note asking the user
    to type the values.
    """
    if not file_path:
        return message
    try:
        name = Path(str(file_path)).name
    except Exception:
        name = "lab_report"

    extracted = ""
    try:
        extracted = extract_text_from_file(str(file_path))
    except Exception as e:
        logger.warning(f"OCR error for {name}: {e}")

    base_msg = message or "Please review my attached lab report and explain the findings."

    if extracted.strip():
        logger.info(f"OCR extracted {len(extracted)} chars from {name}")
        return (
            f"{base_msg}\n\n"
            f"--- BEGIN LAB REPORT TEXT (extracted via OCR from `{name}`) ---\n"
            f"{extracted}\n"
            f"--- END LAB REPORT TEXT ---\n\n"
            "Please use the values above to: identify any abnormal results, "
            "explain in plain English what each abnormal value means, suggest "
            "likely causes, recommended follow-up tests and red flags."
        )

    return (
        f"{base_msg}\n\n"
        f"[Attached file `{name}` could not be read by OCR. "
        "Please ask the user to type the key values from the report.]"
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
    padding: 4px 4px 0 4px !important;
    background: transparent !important; border: none !important;
}
.attach-chip-row p { margin: 0 !important; }
.attach-chip-inline {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--accent-soft);
    color: var(--accent);
    border: 1px solid var(--accent);
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.8rem;
    line-height: 1.2;
    max-width: 100%;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.attach-chip-inline strong { color: var(--ink); font-weight: 600; }
.attach-chip-inline.attach-warn {
    background: rgba(179, 38, 30, 0.18);
    color: #ef6b65;
    border-color: #b3261e;
}

/* ----- Pill-shaped processing indicator ----- */
#processing-pill {
    position: fixed;
    top: 16px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 9999;
    background: var(--panel-2);
    color: var(--ink);
    border: 1px solid var(--accent);
    border-radius: 999px;
    padding: 8px 18px;
    font-size: 0.85rem;
    font-weight: 500;
    box-shadow: 0 6px 24px rgba(0,0,0,0.5);
    animation: pill-pulse 1.6s ease-in-out infinite;
    width: auto !important;
    max-width: 90vw;
}
#processing-pill p { margin: 0 !important; color: var(--ink) !important; font-weight: 500; }
@keyframes pill-pulse {
    0%, 100% { opacity: 1; transform: translateX(-50%) scale(1); }
    50% { opacity: 0.8; transform: translateX(-50%) scale(0.97); }
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

        # ----- Floating processing pill -----
        processing_pill = gr.Markdown(
            "⚡ Generating response…",
            elem_id="processing-pill",
            visible=False,
        )

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
                        visible=False,
                    )
                    input_box = gr.Textbox(
                        placeholder="Ask anything medical — symptoms, medications, lab values…",
                        lines=2,
                        show_label=False,
                        container=False,
                    )
                    with gr.Row(elem_classes=["actions-row"]):
                        upload_btn = gr.UploadButton(
                            "+",
                            file_types=["image", ".pdf"],
                            elem_classes=["icon-btn"],
                        )
                        send_btn = gr.Button("➤", elem_classes=["send-btn"])

                gr.HTML(
                    '<div class="disclaimer">⚠️ Educational use only. Always consult a qualified healthcare professional.</div>'
                )

        # =================== HANDLERS ===================
        def on_upload(file_obj):
            empty = gr.update(value="", visible=False)
            if file_obj is None:
                return None, empty
            path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
            try:
                display_name = Path(path).name
            except Exception:
                display_name = "uploaded file"
            try:
                preview = extract_text_from_file(path)
            except Exception as e:
                logger.warning(f"OCR preview failed: {e}")
                preview = ""
            if preview.strip():
                chip = (
                    f'<span class="attach-chip-inline">📎 <strong>{display_name}</strong> '
                    f'· OCR ✅ {len(preview)} chars</span>'
                )
            else:
                chip = (
                    f'<span class="attach-chip-inline attach-warn">📎 <strong>{display_name}</strong> '
                    f'· OCR could not extract text — describe values in your message</span>'
                )
            return path, gr.update(value=chip, visible=True)

        def submit_message(message, region_val, history, pending_file, request: gr.Request):
            detected = detect_region(request)
            if detected and detected != "General":
                region_val = detected
            elif not region_val:
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

            attachment_name = None
            full_msg = message
            if pending_file:
                try:
                    attachment_name = Path(str(pending_file)).name
                except Exception:
                    attachment_name = "lab_report"
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
                "content": "▍",
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

        def _show_pill():
            return gr.update(visible=True)

        def _hide_pill():
            return gr.update(visible=False)

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
            send_btn.click(_show_pill, None, processing_pill, api_name=False, show_progress=prog)
            .then(submit_message, submit_inputs, submit_outputs, api_name=False, show_progress=prog)
            .then(_hide_pill, None, processing_pill, api_name=False, show_progress=prog)
        )
        (
            input_box.submit(_show_pill, None, processing_pill, api_name=False, show_progress=prog)
            .then(submit_message, submit_inputs, submit_outputs, api_name=False, show_progress=prog)
            .then(_hide_pill, None, processing_pill, api_name=False, show_progress=prog)
        )

        upload_btn.upload(
            fn=on_upload,
            inputs=[upload_btn],
            outputs=[pending_file_state, attach_chip],
            api_name=False,
            show_progress=prog,
        )

        new_chat_btn.click(fn=reset_chat, outputs=submit_outputs, api_name=False, show_progress=prog)

        for btn in example_btns:
            btn.click(fn=fill_example, inputs=[btn], outputs=[input_box], api_name=False, show_progress=prog)

        # Auto-detect region on app load
        def _init_region(request: gr.Request):
            region = detect_region(request)
            global CURRENT_REGION
            CURRENT_REGION = region
            label = (
                f"📍 Region: **{region}**"
                if region != "General"
                else "📍 Region: **General**"
            )
            return region, label

        demo.load(
            fn=_init_region,
            inputs=None,
            outputs=[region_state, region_badge],
            api_name=False,
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
