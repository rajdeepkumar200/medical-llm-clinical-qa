from __future__ import annotations

import os
import logging
from typing import Optional

import gradio as gr  # type: ignore[import-not-found]

from .config import ADAPTER_DIR, ADAPTER_REPO_ID, BASE_MODEL, REGION
from .infer import generate_answer, load_model, load_tokenizer
from .prompts import build_region_aware_system_prompt
from .nlp_processor import process_user_input, build_context_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = None
TOKENIZER = None

REGIONS = ["General", "United States", "United Kingdom", "Canada", "Australia", "New Zealand", "India", "Singapore", "Hong Kong", "Other"]


def get_pipeline():
    global MODEL, TOKENIZER
    if MODEL is None or TOKENIZER is None:
        try:
            logger.info(f"Loading tokenizer for {BASE_MODEL}...")
            TOKENIZER = load_tokenizer(BASE_MODEL)
            adapter_source = ADAPTER_REPO_ID if ADAPTER_REPO_ID else (str(ADAPTER_DIR) if ADAPTER_DIR.exists() else None)
            logger.info(f"Loading model from {BASE_MODEL} with adapter: {adapter_source}...")
            MODEL = load_model(BASE_MODEL, adapter_source)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading pipeline: {e}", exc_info=True)
            raise
    return MODEL, TOKENIZER


def process_message(message: str, file_obj: Optional[object], region: str, history: list) -> tuple[list, str, None]:
    """Process user message and return updated history."""
    
    # Handle file upload
    if file_obj is not None:
        try:
            file_path = file_obj.name
            if file_path.endswith('.txt'):
                with open(file_path, 'r') as f:
                    file_content = f.read()
            else:
                file_content = f"[{file_path.split('.')[-1].upper()} file uploaded]"
            
            message = f"{message}\n\n[DOCUMENT]\n{file_content}" if message.strip() else f"Please analyze this document:\n{file_content}"
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            message = message or "Error reading file"
    
    if not message.strip():
        return history, "", None
    
    try:
        # Get pipeline
        model, tokenizer = get_pipeline()
        
        # Process with NLP
        nlp_result = process_user_input(message)
        enhanced_message = build_context_prompt(nlp_result, message)
        
        # Generate response
        logger.info(f"Generating answer (Region: {region})")
        response = generate_answer(model, tokenizer, enhanced_message, region=region)
        logger.info("Answer generated successfully")
        
        # Update history
        history.append([message, response])
        return history, "", None
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        error_msg = f"⚠️ Error: {str(e)[:150]}"
        history.append([message, error_msg])
        return history, "", None


def build_demo() -> gr.Blocks:
    """Build ChatGPT-style medical assistant UI."""
    with gr.Blocks(title="Clinical AI Assistant", theme=gr.themes.Soft(), css="""
    #chat_box { height: 500px; overflow-y: auto; }
    .message-box { padding: 16px; }
    footer { display: none; }
    """) as demo:
        
        # Header
        gr.Markdown("# 🏥 Clinical AI Assistant\n*Powered by Medical LLM - Ask questions, upload reports, get region-aware guidance*")
        
        # Region selector (compact)
        with gr.Row():
            region = gr.Dropdown(REGIONS, value=REGION, label="🌍 Region", scale=2)
            gr.Markdown("**⚠️ For educational use. Consult healthcare professionals for diagnosis.**", scale=3)
        
        # Conversation display
        chatbot = gr.Chatbot(label="Conversation", height=450, show_label=False)
        
        # Input area - designed like ChatGPT
        with gr.Row():
            # Upload button with + sign
            upload_btn = gr.UploadButton(
                "➕",
                file_count="single",
                file_types=["text", ".pdf", ".png", ".jpg", ".jpeg"],
                scale=1
            )
            
            # Input textbox
            msg_input = gr.Textbox(
                placeholder="Ask a medical question or describe your symptoms...",
                label=None,
                lines=1,
                scale=10,
                show_label=False
            )
            
            # Submit button (pill-shaped search icon)
            submit_btn = gr.Button("🔍", scale=1)
        
        # Hidden file state
        file_state = gr.State(None)
        
        def handle_submit(user_msg, file_obj, region_val, chat_history):
            return process_message(user_msg, file_obj, region_val, chat_history), "", None
        
        def handle_upload(files, chat_history):
            if files:
                file_state.value = files
                return files
            return None
        
        # Events
        submit_btn.click(
            fn=handle_submit,
            inputs=[msg_input, file_state, region, chatbot],
            outputs=[chatbot, msg_input, file_state]
        )
        
        msg_input.submit(
            fn=handle_submit,
            inputs=[msg_input, file_state, region, chatbot],
            outputs=[chatbot, msg_input, file_state]
        )
        
        upload_btn.upload(
            fn=handle_upload,
            inputs=[upload_btn, chatbot],
            outputs=[file_state]
        )
    
    return demo


if __name__ == "__main__":
    build_demo().launch(share=bool(os.getenv("SPACE_ID") or os.getenv("SPACE_HOST")))
