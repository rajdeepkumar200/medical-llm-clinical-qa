from __future__ import annotations

import os
import logging
from typing import Generator

import gradio as gr  # type: ignore[import-not-found]

from .config import BASE_MODEL, ADAPTER_REPO_ID, ADAPTER_DIR, REGION
from .infer import generate_answer, load_model, load_tokenizer
from .nlp_processor import process_user_input, build_context_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = None
TOKENIZER = None
CURRENT_REGION = REGION

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


def generate_response(message: str, region: str) -> Generator[str, None, None]:
    """Generate medical response as a stream."""
    if not message.strip():
        yield ""
        return
    
    try:
        model, tokenizer = get_pipeline()
        
        # Process with NLP
        nlp_result = process_user_input(message)
        enhanced_message = build_context_prompt(nlp_result, message)
        
        # Generate response
        logger.info(f"Generating answer (Region: {region})")
        response = generate_answer(model, tokenizer, enhanced_message, region=region)
        logger.info("Answer generated successfully")
        
        yield response
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        yield f"⚠️ Error: {str(e)[:200]}"


def update_region(region: str):
    global CURRENT_REGION
    CURRENT_REGION = region
    logger.info(f"Region updated to: {region}")


def build_demo():
    """Build Claude-style medical assistant UI."""
    with gr.Blocks(title="Clinical AI Assistant") as demo:
        # Header
        gr.Markdown("# 🏥 Clinical AI Assistant")
        gr.Markdown("*Ask medical questions, upload lab reports - get region-aware guidance*")
        
        # Region selector
        with gr.Row():
            region_selector = gr.Dropdown(REGIONS, value=REGION, label="Region")
        
        # Disclaimer
        gr.Markdown("⚠️ **For educational use only.** Always consult healthcare professionals.")
        
        # Chat display (full height like Claude)
        chatbot = gr.Chatbot(height=600, label="")
        
        # Input area - Claude style
        with gr.Row():
            upload_btn = gr.UploadButton(
                "➕",
                file_count="single",
                file_types=["text", ".pdf", ".png", ".jpg", ".jpeg"],
                scale=1
            )
            msg_input = gr.Textbox(
                placeholder="Ask a medical question...",
                lines=1,
                scale=10
            )
            submit_btn = gr.Button("Send", scale=1)
        
        def process_and_respond(message, uploaded_file, region_val, chat_hist):
            """Process message with optional file and generate response."""
            full_message = message
            
            # Handle file upload
            if uploaded_file is not None:
                try:
                    file_path = uploaded_file if isinstance(uploaded_file, str) else getattr(uploaded_file, 'name', str(uploaded_file))
                    if file_path.endswith('.txt'):
                        with open(file_path, 'r') as f:
                            file_content = f.read()
                        full_message = f"{message}\n\n[LAB REPORT]\n{file_content}"
                    else:
                        file_type = file_path.split('.')[-1].upper()
                        full_message = f"{message}\n\n[{file_type} file uploaded for analysis]"
                except Exception as e:
                    logger.error(f"Error reading file: {e}")
                    full_message = message or "Error reading file"
            
            if not full_message.strip():
                return chat_hist, ""
            
            try:
                # Generate response
                response = ""
                for chunk in generate_response(full_message, region_val):
                    response += chunk
                
                # Add to chat history
                chat_hist.append([message, response])
                return chat_hist, ""
                
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                error_msg = f"⚠️ Error: {str(e)[:150]}"
                chat_hist.append([message, error_msg])
                return chat_hist, ""
        
        # Event handlers
        submit_btn.click(
            fn=process_and_respond,
            inputs=[msg_input, upload_btn, region_selector, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        msg_input.submit(
            fn=process_and_respond,
            inputs=[msg_input, upload_btn, region_selector, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        region_selector.change(fn=update_region, inputs=[region_selector])
    
    return demo


if __name__ == "__main__":
    build_demo().launch(share=bool(os.getenv("SPACE_ID") or os.getenv("SPACE_HOST")))
