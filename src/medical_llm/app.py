from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Generator

import gradio as gr  # type: ignore[import-not-found]

from .config import BASE_MODEL, ADAPTER_REPO_ID, ADAPTER_DIR, REGION
from .infer import generate_answer, load_model, load_tokenizer
from .nlp_processor import process_user_input, build_context_prompt
from .ocr import extract_text_from_file

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
    """Generate medical response."""
    if not message.strip():
        yield ""
        return

    try:
        model, tokenizer = get_pipeline()
        nlp_result = process_user_input(message)
        enhanced_message = build_context_prompt(nlp_result, message)
        logger.info(f"Generating answer (Region: {region})")
        response = generate_answer(model, tokenizer, enhanced_message, region=region)
        logger.info("Answer generated successfully")
        yield response
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
    --bg: #faf9f5;
    --panel: #ffffff;
    --ink: #1f1e1c;
    --ink-soft: #5c5a54;
    --accent: #c96442;
    --accent-soft: #f4ead8;
    --border: #e7e2d6;
    --user-bg: #f1ece0;
    --assistant-bg: #ffffff;
}

.gradio-container {
    background: var(--bg) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif !important;
    color: var(--ink) !important;
    max-width: 880px !important;
    margin: 0 auto !important;
}

footer { display: none !important; }

/* ----- Welcome view ----- */
#welcome-view {
    min-height: 78vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding: 24px 16px;
}
#welcome-title {
    text-align: center;
    margin-bottom: 8px;
}
#welcome-title h1 {
    font-size: 2.4rem;
    font-weight: 600;
    color: var(--ink);
    margin: 0 0 8px 0;
}
#welcome-title p {
    color: var(--ink-soft);
    font-size: 1.05rem;
    margin: 0;
}
#welcome-card {
    width: 100%;
    max-width: 720px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 14px 14px 10px 14px;
    box-shadow: 0 8px 32px rgba(31, 30, 28, 0.06);
    margin-top: 28px;
}
#welcome-card textarea {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    font-size: 1.05rem !important;
    resize: none !important;
    padding: 12px !important;
    min-height: 60px !important;
}
#welcome-card textarea:focus { outline: none !important; }
#welcome-card .scroll-hide { border: none !important; background: transparent !important; }

#welcome-actions {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 6px 2px 6px;
    gap: 8px;
}
.icon-btn button, button.icon-btn {
    background: var(--accent-soft) !important;
    color: var(--accent) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    width: 40px !important; height: 40px !important;
    min-width: 40px !important;
    padding: 0 !important;
    font-size: 1.3rem !important;
    font-weight: 600 !important;
}
.icon-btn button:hover, button.icon-btn:hover {
    background: #ecdcc4 !important;
}
.send-btn button, button.send-btn {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 999px !important;
    width: 42px !important; height: 42px !important;
    min-width: 42px !important;
    padding: 0 !important;
    font-size: 1.1rem !important;
}
.send-btn button:hover, button.send-btn:hover { filter: brightness(0.95); }

#region-badge {
    max-width: 720px; width: 100%; margin: 14px auto 0 auto;
    text-align: center; color: var(--ink-soft); font-size: 0.85rem;
}
#region-badge strong { color: var(--accent); }

#examples-row {
    max-width: 720px; width: 100%; margin: 24px auto 0 auto;
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
}
.example-card button {
    background: var(--panel) !important;
    color: var(--ink) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    text-align: left !important;
    padding: 12px 14px !important;
    font-weight: 400 !important;
    font-size: 0.92rem !important;
    line-height: 1.35 !important;
    white-space: normal !important;
    height: auto !important;
}
.example-card button:hover { background: var(--accent-soft) !important; }

/* ----- Inline attached-file chip (inside the input card, above textarea) ----- */
.attach-chip-row {
    padding: 6px 8px 0 8px !important;
    background: transparent !important;
    border: none !important;
}
.attach-chip-row p { margin: 0 !important; }
.attach-chip-inline {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--accent-soft);
    color: var(--accent);
    border: 1px solid #e6d5b8;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.82rem;
    line-height: 1.2;
    max-width: 100%;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.attach-chip-inline strong { color: var(--ink); font-weight: 600; }
.attach-chip-inline.attach-warn {
    background: #fdecea; color: #b3261e; border-color: #f5c6c0;
}

/* ----- Pill-shaped processing indicator at the top ----- */
#processing-pill {
    position: fixed;
    top: 16px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 9999;
    background: var(--ink);
    color: #fff;
    border-radius: 999px;
    padding: 8px 18px;
    font-size: 0.85rem;
    font-weight: 500;
    box-shadow: 0 6px 24px rgba(31, 30, 28, 0.18);
    animation: pill-pulse 1.6s ease-in-out infinite;
    width: auto !important;
    max-width: 90vw;
}
#processing-pill p { margin: 0 !important; color: #fff !important; }
@keyframes pill-pulse {
    0%, 100% { opacity: 1; transform: translateX(-50%) scale(1); }
    50% { opacity: 0.85; transform: translateX(-50%) scale(0.98); }
}

/* ----- Hide Gradio's default progress overlay (it was overlapping the input row) ----- */
.gradio-container .progress-text,
.gradio-container .progress-bar,
.gradio-container .wrap.default,
.gradio-container .wrap.svelte-1ipelgc,
.gradio-container div[class*="progress"] {
    display: none !important;
}
/* Also hide the eta/timer pill some Gradio versions show */
.gradio-container .eta-bar { display: none !important; }

/* ----- Chat view ----- */
#chat-view { padding: 12px 4px 24px 4px; }
#chat-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 8px 16px 8px; border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
}
#chat-header h2 { font-size: 1.1rem; margin: 0; font-weight: 600; }

.bubble {
    border-radius: 16px; padding: 14px 18px; margin: 10px 0;
    border: 1px solid var(--border);
}
.bubble.msg-user { background: var(--user-bg); }
.bubble.msg-assistant { background: var(--assistant-bg); }
.bubble-head { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.bubble-head .avatar { font-size: 1.1rem; }
.bubble-head .who { font-weight: 600; color: var(--ink-soft); font-size: 0.85rem; }
.bubble-body { font-size: 0.98rem; line-height: 1.6; color: var(--ink); }
.bubble-body h3 { font-size: 1.02rem; margin-top: 14px; margin-bottom: 6px; color: var(--ink); }
.bubble-body ul { padding-left: 22px; margin: 6px 0; }
.bubble-body li { margin: 4px 0; }
.bubble-body strong { color: var(--ink); }
.attach-chip {
    display: inline-block; background: var(--accent-soft); color: var(--accent);
    border-radius: 999px; padding: 2px 10px; font-size: 0.8rem; margin: 4px 0;
}

#chat-input-card {
    position: sticky; bottom: 0;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 10px 12px 6px 12px;
    margin-top: 16px;
    box-shadow: 0 -4px 16px rgba(31,30,28,0.04);
}
#chat-input-card textarea {
    border: none !important; background: transparent !important;
    box-shadow: none !important; font-size: 1rem !important;
    min-height: 48px !important; resize: none !important;
}
#chat-input-card textarea:focus { outline: none !important; }

.new-chat-btn button {
    background: transparent !important;
    color: var(--ink-soft) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    padding: 6px 14px !important;
    font-size: 0.85rem !important;
}
.new-chat-btn button:hover { background: var(--accent-soft) !important; color: var(--accent) !important; }

.disclaimer {
    text-align: center; color: var(--ink-soft); font-size: 0.78rem;
    margin-top: 12px;
}
"""


def build_demo():
    """Claude-inspired UI: centered welcome screen that transitions to a chat view."""
    with gr.Blocks(title="Clinical AI Assistant", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:

        # State: conversation history as list of {role, content, attachment}
        history_state = gr.State([])
        pending_file_state = gr.State(None)  # file path waiting to be attached

        # Pill-shaped processing indicator visible during generation
        processing_pill = gr.Markdown(
            "⚡ Generating response…",
            elem_id="processing-pill",
            visible=False,
        )

        # =================== WELCOME VIEW ===================
        with gr.Column(visible=True, elem_id="welcome-view") as welcome_view:
            gr.HTML(
                """
                <div id="welcome-title">
                    <h1>🩺 Clinical AI Assistant</h1>
                    <p>Ask anything about symptoms, medications, or lab reports — clear answers, in plain English.</p>
                </div>
                """
            )

            with gr.Column(elem_id="welcome-card"):
                # Attached-file chip sits inside the input card, above the textarea
                welcome_attach_chip = gr.Markdown(
                    "",
                    elem_classes=["attach-chip-row"],
                    visible=False,
                )
                welcome_input = gr.Textbox(
                    placeholder="Describe your symptoms or paste lab values…",
                    lines=2,
                    show_label=False,
                    container=False,
                    elem_id="welcome-textbox",
                )
                with gr.Row(elem_classes=["actions-row"]):
                    welcome_upload = gr.UploadButton(
                        "+",
                        file_types=["image", ".pdf"],
                        elem_classes=["icon-btn"],
                    )
                    welcome_send = gr.Button(
                        "➤", elem_classes=["send-btn"]
                    )

            # Detected region badge (auto-filled on app load)
            region_state = gr.State(REGION)
            region_badge = gr.Markdown(
                "📍 Detecting your region…",
                elem_id="region-badge",
            )

            with gr.Row(elem_id="examples-row"):
                example_btns = [
                    gr.Button(p, elem_classes=["example-card"]) for p in EXAMPLE_PROMPTS
                ]

            gr.HTML(
                '<div class="disclaimer">⚠️ Educational use only. Always consult a qualified healthcare professional.</div>'
            )

        # =================== CHAT VIEW ===================
        with gr.Column(visible=False, elem_id="chat-view") as chat_view:
            with gr.Row(elem_id="chat-header"):
                gr.HTML("<h2>🩺 Clinical AI Assistant</h2>")
                new_chat_btn = gr.Button(
                    "＋ New chat", elem_classes=["new-chat-btn"]
                )

            chat_display = gr.Markdown(
                "", elem_id="chat-stream", sanitize_html=False
            )

            with gr.Column(elem_id="chat-input-card"):
                chat_attach_chip = gr.Markdown(
                    "",
                    elem_classes=["attach-chip-row"],
                    visible=False,
                )
                chat_input = gr.Textbox(
                    placeholder="Ask a follow-up…",
                    lines=2,
                    show_label=False,
                    container=False,
                )
                with gr.Row(elem_classes=["actions-row"]):
                    chat_upload = gr.UploadButton(
                        "+",
                        file_types=["image", ".pdf"],
                        elem_classes=["icon-btn"],
                    )
                    chat_send = gr.Button(
                        "➤", elem_classes=["send-btn"]
                    )

            gr.HTML(
                '<div class="disclaimer">⚠️ Educational information — not a diagnosis. Always confirm with a clinician.</div>'
            )

        # =================== HANDLERS ===================
        def on_upload(file_obj):
            """Returns (path, chip_update_for_welcome, chip_update_for_chat)."""
            empty = gr.update(value="", visible=False)
            if file_obj is None:
                return None, empty, empty

            # gr.UploadButton may return a path string or an object with .name
            path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
            try:
                display_name = Path(path).name
            except Exception:
                display_name = "uploaded file"

            # Try OCR right away so the user gets immediate feedback.
            try:
                preview = extract_text_from_file(path)
            except Exception as e:
                logger.warning(f"OCR preview failed: {e}")
                preview = ""

            if preview.strip():
                chip_text = (
                    f'<span class="attach-chip-inline">📎 <strong>{display_name}</strong> '
                    f'· OCR ✅ {len(preview)} chars</span>'
                )
            else:
                chip_text = (
                    f'<span class="attach-chip-inline attach-warn">📎 <strong>{display_name}</strong> '
                    f'· OCR could not extract text — describe values in your message</span>'
                )
            chip_update = gr.update(value=chip_text, visible=True)
            # Same chip is shown in whichever view is active; harmless to update both.
            return path, chip_update, chip_update

        def submit_message(message, region_val, history, pending_file, request: gr.Request):
            """Submit from the welcome view (also covers follow-ups)."""
            # Re-detect region from the live request in case state was stale.
            detected = detect_region(request)
            if detected and detected != "General":
                region_val = detected
            elif not region_val:
                region_val = "General"

            message = (message or "").strip()
            empty_chip = gr.update(value="", visible=False)
            if not message and not pending_file:
                return (
                    gr.update(),  # welcome_view
                    gr.update(),  # chat_view
                    history,
                    "",  # chat_display
                    "",  # welcome_input
                    "",  # chat_input
                    None,  # pending_file
                    empty_chip,  # welcome_attach_chip
                    empty_chip,  # chat_attach_chip
                )

            attachment_name = None
            full_msg = message
            if pending_file:
                try:
                    attachment_name = Path(str(pending_file)).name
                except Exception:
                    attachment_name = "lab_report"
                full_msg = _attach_file_note(message, pending_file)

            history = list(history or [])
            history.append(
                {"role": "user", "content": message or "(attached lab report)",
                 "attachment": attachment_name}
            )

            # Generate full answer
            response_text = ""
            try:
                for chunk in generate_response(full_msg, region_val):
                    response_text += chunk
            except Exception as e:
                logger.error(f"Generation error: {e}", exc_info=True)
                response_text = f"⚠️ **Error:** {str(e)[:200]}"

            history.append(
                {"role": "assistant", "content": response_text, "attachment": None}
            )

            rendered = _render_conversation(history)

            return (
                gr.update(visible=False),  # welcome_view
                gr.update(visible=True),  # chat_view
                history,
                rendered,
                "",  # clear welcome_input
                "",  # clear chat_input
                None,  # clear pending_file
                empty_chip,  # clear welcome_attach_chip
                empty_chip,  # clear chat_attach_chip
            )

        def reset_chat():
            empty_chip = gr.update(value="", visible=False)
            return (
                gr.update(visible=True),  # welcome_view
                gr.update(visible=False),  # chat_view
                [],  # history
                "",  # chat_display
                "",  # welcome_input
                "",  # chat_input
                None,  # pending_file
                empty_chip,  # welcome_attach_chip
                empty_chip,  # chat_attach_chip
            )

        def fill_example(text):
            return text

        # --- Wire events ---
        submit_outputs = [
            welcome_view, chat_view, history_state,
            chat_display, welcome_input, chat_input,
            pending_file_state, welcome_attach_chip, chat_attach_chip,
        ]

        # Helpers to show/hide the pill-shaped processing indicator before/after generation
        def _show_pill():
            return gr.update(visible=True)

        def _hide_pill():
            return gr.update(visible=False)

        submit_inputs_w = [welcome_input, region_state, history_state, pending_file_state]
        submit_inputs_c = [chat_input, region_state, history_state, pending_file_state]

        prog = "hidden"  # disable Gradio's default loader; we use our own pill
        (
            welcome_send.click(_show_pill, None, processing_pill, api_name=False, show_progress=prog)
            .then(submit_message, submit_inputs_w, submit_outputs, api_name=False, show_progress=prog)
            .then(_hide_pill, None, processing_pill, api_name=False, show_progress=prog)
        )
        (
            welcome_input.submit(_show_pill, None, processing_pill, api_name=False, show_progress=prog)
            .then(submit_message, submit_inputs_w, submit_outputs, api_name=False, show_progress=prog)
            .then(_hide_pill, None, processing_pill, api_name=False, show_progress=prog)
        )
        (
            chat_send.click(_show_pill, None, processing_pill, api_name=False, show_progress=prog)
            .then(submit_message, submit_inputs_c, submit_outputs, api_name=False, show_progress=prog)
            .then(_hide_pill, None, processing_pill, api_name=False, show_progress=prog)
        )
        (
            chat_input.submit(_show_pill, None, processing_pill, api_name=False, show_progress=prog)
            .then(submit_message, submit_inputs_c, submit_outputs, api_name=False, show_progress=prog)
            .then(_hide_pill, None, processing_pill, api_name=False, show_progress=prog)
        )

        welcome_upload.upload(
            fn=on_upload,
            inputs=[welcome_upload],
            outputs=[pending_file_state, welcome_attach_chip, chat_attach_chip],
            api_name=False,
            show_progress=prog,
        )
        chat_upload.upload(
            fn=on_upload,
            inputs=[chat_upload],
            outputs=[pending_file_state, welcome_attach_chip, chat_attach_chip],
            api_name=False,
            show_progress=prog,
        )

        new_chat_btn.click(fn=reset_chat, outputs=submit_outputs, api_name=False, show_progress=prog)

        for btn in example_btns:
            btn.click(fn=fill_example, inputs=[btn], outputs=[welcome_input], api_name=False, show_progress=prog)

        # Auto-detect region on app load and update both the badge + state
        def _init_region(request: gr.Request):
            region = detect_region(request)
            global CURRENT_REGION
            CURRENT_REGION = region
            label = (
                f"📍 Region auto-detected: **{region}**"
                if region != "General"
                else "📍 Region: **General** (no local preference detected)"
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
