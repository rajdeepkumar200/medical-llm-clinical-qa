"""
Hugging Face Spaces entry point for the medical LLM chat app.
"""
import os

from src.medical_llm.app import build_demo

if __name__ == "__main__":
    in_space = bool(os.getenv("SPACE_ID") or os.getenv("SPACE_HOST"))
    # show_api=False avoids the gradio_client schema bug ("argument of type
    # 'bool' is not iterable") that previously crashed startup on /info.
    build_demo().queue().launch(
        share=False if in_space else True,
        show_api=False,
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
