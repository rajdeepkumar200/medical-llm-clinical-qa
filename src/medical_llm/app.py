from __future__ import annotations

import os
import logging

import gradio as gr  # type: ignore[import-not-found]

from .config import ADAPTER_DIR, ADAPTER_REPO_ID, BASE_MODEL
from .infer import generate_answer, load_model, load_tokenizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = None
TOKENIZER = None


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


def respond(message: str, history):
    try:
        model, tokenizer = get_pipeline()
        logger.info(f"Generating answer for: {message[:50]}...")
        answer = generate_answer(model, tokenizer, message)
        logger.info("Answer generated successfully")
        return answer
    except Exception as e:
        logger.error(f"Error in respond: {e}", exc_info=True)
        return f"Error: {str(e)}"


def build_demo() -> gr.Blocks:
    demo = gr.ChatInterface(
        fn=respond,
        title="Clinical Q&A Assistant",
        description="Ask a medical question and get a concise answer. Always verify important advice with a clinician.",
        examples=[
            "What are the common side effects of amoxicillin?",
            "How do I recognize signs of dehydration in a child?",
            "What is the first-line treatment for seasonal allergic rhinitis?",
        ],
        cache_examples=False,  # Disable example caching to avoid model load on startup
    )
    # Disable API docs to avoid Gradio schema generation crash
    demo.show_api = False
    return demo


if __name__ == "__main__":
    build_demo().launch(share=bool(os.getenv("SPACE_ID") or os.getenv("SPACE_HOST")))
