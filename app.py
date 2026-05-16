"""
Hugging Face Spaces entry point for the medical LLM chat app.
"""
from src.medical_llm.app import build_demo

if __name__ == "__main__":
    build_demo().launch(share=True)
