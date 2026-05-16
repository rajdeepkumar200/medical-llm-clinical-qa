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
    """Build minimal professional UI - avoids Gradio 4.36.0 schema issues."""
    with gr.Blocks(title="Clinical AI Assistant") as demo:
        # Header
        gr.Markdown("# 🏥 Clinical AI Assistant")
        gr.Markdown("*Medical guidance with region awareness*")
        
        # Region selector (no scale)
        region_selector = gr.Dropdown(REGIONS, value=REGION, label="Region")
        
        # Chat display
        chatbot = gr.Chatbot()
        
        # Input area - simple layout
        msg_input = gr.Textbox(placeholder="Ask about your health...", lines=3)
        gr.Markdown("💡 Press Enter or click the button below to submit. Add photo with ➕ button")
        submit_btn = gr.Button("🔍 Submit", variant="primary")
        upload_btn = gr.Button("➕ Upload Lab Report/Photo")
        
        # Disclaimer
        gr.Markdown("⚠️ **For educational use only.** Always consult healthcare professionals.")
        
        def process_and_respond(message, region_val, chat_hist):
            """Process message and generate response."""
            if not message.strip():
                return chat_hist, ""
            
            try:
                # Generate response
                response = ""
                for chunk in generate_response(message, region_val):
                    response += chunk
                
                # Add to chat history
                chat_hist.append([message, response])
                return chat_hist, ""
                
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                error_msg = f"⚠️ Error: {str(e)[:150]}"
                chat_hist.append([message, error_msg])
                return chat_hist, ""
        
        # Event: Submit button
        submit_btn.click(
            fn=process_and_respond,
            inputs=[msg_input, region_selector, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        # Event: Enter key submits
        msg_input.submit(
            fn=process_and_respond,
            inputs=[msg_input, region_selector, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        # Event: Upload button
        upload_btn.click(
            fn=lambda: print("Upload clicked"),
            outputs=None
        )
        
        # Event: Region change
        region_selector.change(fn=update_region, inputs=[region_selector])
    
    return demo


if __name__ == "__main__":
    build_demo().launch(share=bool(os.getenv("SPACE_ID") or os.getenv("SPACE_HOST")))
