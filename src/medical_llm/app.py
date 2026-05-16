from __future__ import annotations

import os

import gradio as gr  # type: ignore[import-not-found]

from .config import ADAPTER_DIR, ADAPTER_REPO_ID, BASE_MODEL
from .infer import generate_answer, load_model, load_tokenizer


MODEL = None
TOKENIZER = None


def get_pipeline():
    global MODEL, TOKENIZER
    if MODEL is None or TOKENIZER is None:
        TOKENIZER = load_tokenizer(BASE_MODEL)
        adapter_source = ADAPTER_REPO_ID if ADAPTER_REPO_ID else (str(ADAPTER_DIR) if ADAPTER_DIR.exists() else None)
        MODEL = load_model(BASE_MODEL, adapter_source)
    return MODEL, TOKENIZER


def respond(message: str, history):
    model, tokenizer = get_pipeline()
    return generate_answer(model, tokenizer, message)


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
    )
    # Disable API docs to avoid Gradio schema generation crash
    demo.show_api = False
    return demo


if __name__ == "__main__":
    build_demo().launch(share=bool(os.getenv("SPACE_ID") or os.getenv("SPACE_HOST")))
