from __future__ import annotations

import os
import logging

import gradio as gr  # type: ignore[import-not-found]

from .config import ADAPTER_DIR, ADAPTER_REPO_ID, BASE_MODEL, REGION
from .infer import generate_answer, load_model, load_tokenizer
from .prompts import build_region_aware_system_prompt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = None
TOKENIZER = None
CURRENT_REGION = REGION

SUPPORTED_REGIONS = [
    "General",
    "United States",
    "United Kingdom", 
    "Canada",
    "Australia",
    "New Zealand",
    "India",
    "Singapore",
    "Hong Kong",
    "Other"
]


def get_pipeline():
    global MODEL, TOKENIZER
    if MODEL is None or TOKENIZER is None:
        try:
            logger.info(f"Loading tokenizer for {BASE_MODEL}...")
            TOKENIZER = load_tokenizer(BASE_MODEL)
            
            # Try to load adapter from repo ID first, then fall back to local
            adapter_source = None
            if ADAPTER_REPO_ID:
                logger.info(f"Attempting to load adapter from HF repo: {ADAPTER_REPO_ID}")
                try:
                    adapter_source = ADAPTER_REPO_ID
                except Exception as e:
                    logger.warning(f"Failed to load from {ADAPTER_REPO_ID}: {e}")
                    adapter_source = None
            
            # Fall back to local adapter directory
            if not adapter_source and ADAPTER_DIR.exists():
                logger.info(f"Falling back to local adapter directory: {ADAPTER_DIR}")
                adapter_source = str(ADAPTER_DIR)
            
            logger.info(f"Loading model from {BASE_MODEL} with adapter: {adapter_source}...")
            MODEL = load_model(BASE_MODEL, adapter_source)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading pipeline: {e}", exc_info=True)
            raise
    return MODEL, TOKENIZER


def respond(message: str, history, region: str = REGION):
    try:
        global CURRENT_REGION
        CURRENT_REGION = region
        model, tokenizer = get_pipeline()
        logger.info(f"Generating answer for: {message[:50]}... (Region: {region})")
        answer = generate_answer(model, tokenizer, message, region=region)
        logger.info("Answer generated successfully")
        return answer
    except Exception as e:
        logger.error(f"Error in respond: {e}", exc_info=True)
        return f"Error generating response: {str(e)}\n\nPlease try again or check the Space logs for details."


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Clinical Q&A Assistant") as demo:
        gr.Markdown("""
        # Clinical Q&A Assistant
        Ask a medical question and get a concise, evidence-based answer tailored to your region's healthcare standards.
        
        **Important:** Always verify important medical advice with a qualified healthcare professional.
        """)
        
        with gr.Row():
            region_selector = gr.Dropdown(
                choices=SUPPORTED_REGIONS,
                value=REGION,
                label="Select Your Region/Country",
                info="Healthcare standards and practices vary by region"
            )
        
        with gr.Row():
            msg = gr.Textbox(
                label="Your Question",
                placeholder="Ask your medical question here...",
                lines=2
            )
        
        output = gr.Textbox(
            label="Response",
            interactive=False,
            lines=6
        )
        
        def answer_question(question: str, region: str) -> str:
            if not question.strip():
                return ""
            try:
                model, tokenizer = get_pipeline()
                logger.info(f"Generating answer for: {question[:50]}... (Region: {region})")
                result = generate_answer(model, tokenizer, question, region=region)
                logger.info("Answer generated successfully")
                return result
            except Exception as e:
                logger.error(f"Error in answer_question: {e}", exc_info=True)
                return f"Error: {str(e)}"
        
        submit = gr.Button("Get Answer")
        submit.click(
            fn=answer_question,
            inputs=[msg, region_selector],
            outputs=output
        )
        msg.submit(
            fn=answer_question,
            inputs=[msg, region_selector],
            outputs=output
        )
    return demo


if __name__ == "__main__":
    build_demo().launch(share=bool(os.getenv("SPACE_ID") or os.getenv("SPACE_HOST")))
