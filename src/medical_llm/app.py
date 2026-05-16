from __future__ import annotations

import os
import json
import logging
from pathlib import Path

import gradio as gr  # type: ignore[import-not-found]

from .config import ADAPTER_DIR, ADAPTER_REPO_ID, BASE_MODEL, REGION
from .infer import generate_answer, load_model, load_tokenizer
from .prompts import build_region_aware_system_prompt
from .nlp_processor import process_user_input, build_context_prompt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = None
TOKENIZER = None

SUPPORTED_REGIONS = [
    "General", "United States", "United Kingdom", "Canada",
    "Australia", "New Zealand", "India", "Singapore", "Hong Kong", "Other"
]


def get_pipeline():
    """Load model and tokenizer on first call."""
    global MODEL, TOKENIZER
    if MODEL is None or TOKENIZER is None:
        try:
            logger.info(f"Loading tokenizer for {BASE_MODEL}...")
            TOKENIZER = load_tokenizer(BASE_MODEL)
            
            adapter_source = None
            if ADAPTER_REPO_ID:
                logger.info(f"Attempting to load adapter from HF repo: {ADAPTER_REPO_ID}")
                try:
                    adapter_source = ADAPTER_REPO_ID
                except Exception as e:
                    logger.warning(f"Failed to load from {ADAPTER_REPO_ID}: {e}")
                    adapter_source = None
            
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


def answer_question(question: str, region: str, history: list) -> tuple[list, str]:
    """Generate answer with enhanced NLP context."""
    if not question.strip():
        return history, ""
    
    try:
        # Process input with NLP
        nlp_result = process_user_input(question)
        enhanced_question = build_context_prompt(nlp_result, question)
        
        # Generate response
        model, tokenizer = get_pipeline()
        logger.info(f"Generating answer for: {question[:50]}... (Region: {region})")
        response = generate_answer(model, tokenizer, enhanced_question, region=region)
        logger.info("Answer generated successfully")
        
        # Add to history
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response})
        
        return history, ""
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        error_msg = f"Error: {str(e)[:200]}"
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": error_msg})
        return history, ""


def extract_text_from_file(file_path: str) -> str:
    """Extract text from uploaded file (txt, pdf placeholder)."""
    try:
        if file_path.endswith('.txt'):
            with open(file_path, 'r') as f:
                return f.read()
        elif file_path.endswith('.pdf'):
            return "[PDF detected - OCR would be used in production. For now, please describe the report.]"
        elif file_path.endswith(('.png', '.jpg', '.jpeg')):
            return "[Image detected - Medical image analysis would be used in production. For now, please describe what you see.]"
        else:
            return "[File format not supported yet. Please describe the content.]"
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return f"Error reading file: {str(e)}"


def process_file_upload(file_obj, region: str, history: list) -> tuple[list, str]:
    """Process uploaded file (lab report, prescription, image)."""
    if file_obj is None:
        return history, ""
    
    try:
        file_path = file_obj.name
        file_content = extract_text_from_file(file_path)
        
        prompt = f"""I have uploaded a medical document/image. Here's what I can extract from it:

{file_content}

Based on this document, please:
1. Identify any mentioned conditions, symptoms, or findings
2. Suggest what tests or treatments might be recommended
3. Provide general guidance on next steps (noting this requires professional review)

Always remind the user to consult with a healthcare professional."""
        
        return answer_question(prompt, region, history)
    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        error_msg = f"Error processing file: {str(e)[:200]}"
        history.append({"role": "assistant", "content": error_msg})
        return history, ""


def build_demo() -> gr.Blocks:
    """Build ChatGPT-like medical Q&A interface."""
    with gr.Blocks(title="Clinical AI Assistant", theme=gr.themes.Soft()) as demo:
        # Header
        gr.Markdown("""
        # 🏥 Clinical AI Assistant
        **Powered by Medical Fine-Tuned LLM**
        
        Ask medical questions, upload lab reports, or describe symptoms. Get evidence-based guidance tailored to your region.
        
        ⚠️ **Disclaimer:** This is for educational purposes. Always consult healthcare professionals for diagnosis and treatment.
        """)
        
        # Settings Row
        with gr.Row():
            region_selector = gr.Dropdown(
                choices=SUPPORTED_REGIONS,
                value=REGION,
                label="🌍 Select Your Region",
                info="Responses will be tailored to your regional healthcare standards"
            )
            clear_btn = gr.Button("🗑️ Clear Conversation", scale=1)
        
        # Conversation display
        chatbot_display = gr.Chatbot(
            label="💬 Conversation",
            height=400,
            show_label=True
        )
        
        # Hidden state to store conversation history
        history_state = gr.State([])
        
        # Input Section
        gr.Markdown("### 📝 Ask Your Medical Question")
        with gr.Row():
            question_input = gr.Textbox(
                placeholder="Describe your symptoms or ask a medical question...",
                label="Your Question",
                lines=3,
                show_label=False
            )
        
        with gr.Row():
            submit_btn = gr.Button("Send 📤", variant="primary", scale=2)
            clear_input_btn = gr.Button("Clear ❌", scale=1)
        
        # File upload section
        gr.Markdown("### 📄 Optional: Upload Medical Documents")
        with gr.Row():
            file_upload = gr.File(
                label="Upload Lab Report, Prescription, or Medical Image",
                file_count="single",
                file_types=["text", ".pdf", ".png", ".jpg", ".jpeg"]
            )
            upload_btn = gr.Button("Analyze Document 🔍", scale=1)
        
        # Processing indicator
        gr.Markdown("*Loading indicator shows when response is being generated...*")
        
        def update_display(history_data):
            """Convert history to chatbot display format."""
            messages = []
            i = 0
            while i < len(history_data):
                if i < len(history_data) and history_data[i]["role"] == "user":
                    user_msg = history_data[i]["content"]
                    bot_msg = ""
                    if i + 1 < len(history_data) and history_data[i + 1]["role"] == "assistant":
                        bot_msg = history_data[i + 1]["content"]
                        i += 2
                    else:
                        i += 1
                    messages.append([user_msg, bot_msg])
                else:
                    i += 1
            return messages
        
        def clear_conversation():
            """Clear conversation history."""
            return [], [], ""
        
        def submit_question(question: str, region: str, hist):
            """Handle question submission."""
            new_hist, _ = answer_question(question, region, hist)
            return update_display(new_hist), "", new_hist
        
        def handle_file_upload(file_obj, region: str, hist):
            """Handle file upload."""
            new_hist, _ = process_file_upload(file_obj, region, hist)
            return update_display(new_hist), new_hist
        
        # Event handlers
        submit_btn.click(
            fn=submit_question,
            inputs=[question_input, region_selector, history_state],
            outputs=[chatbot_display, question_input, history_state]
        )
        
        question_input.submit(
            fn=submit_question,
            inputs=[question_input, region_selector, history_state],
            outputs=[chatbot_display, question_input, history_state]
        )
        
        upload_btn.click(
            fn=handle_file_upload,
            inputs=[file_upload, region_selector, history_state],
            outputs=[chatbot_display, history_state]
        )
        
        clear_btn.click(
            fn=clear_conversation,
            outputs=[chatbot_display, history_state, question_input]
        )
        
        clear_input_btn.click(
            fn=lambda: "",
            outputs=[question_input]
        )
    
    return demo


if __name__ == "__main__":
    build_demo().launch(share=bool(os.getenv("SPACE_ID") or os.getenv("SPACE_HOST")))
