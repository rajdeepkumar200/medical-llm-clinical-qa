from __future__ import annotations

import os

import gradio as gr
import torch
from huggingface_hub import HfApi, login
from huggingface_hub.errors import HfHubHTTPError
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = os.getenv("BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
ADAPTER_REPO_ID = os.getenv("ADAPTER_REPO_ID", "RajdeepSingh-ai/medical-llama-medical-qa")
HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_TOKEN")
    or os.getenv("HUGGINGFACE_TOKEN")
)
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.05"))
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a careful medical assistant. Give concise, evidence-aware answers and note when professional care is needed.",
)

MODEL = None
TOKENIZER = None
AUTHENTICATED_USER = None
AUTH_ERROR = None


if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        AUTHENTICATED_USER = HfApi().whoami(token=HF_TOKEN)["name"]
    except HfHubHTTPError as exc:
        AUTH_ERROR = f"Invalid Hugging Face token in Space secret: {exc}"
    except Exception as exc:
        AUTH_ERROR = f"Unable to validate Hugging Face token: {exc}"


def build_prompt(question: str) -> str:
    return (
        "<s>[INST] <<SYS>>\n"
        f"{SYSTEM_PROMPT}\n"
        "<</SYS>>\n\n"
        f"{question.strip()} [/INST]"
    )


def load_tokenizer(model_name: str):
    if not HF_TOKEN:
        raise RuntimeError(
            "Missing Hugging Face token. Add a Space secret named HF_TOKEN or HUGGINGFACE_HUB_TOKEN."
        )
    if AUTH_ERROR:
        raise RuntimeError(AUTH_ERROR)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(base_model: str, adapter_repo_id: str):
    if not HF_TOKEN:
        raise RuntimeError(
            "Missing Hugging Face token. Add a Space secret named HF_TOKEN or HUGGINGFACE_HUB_TOKEN."
        )
    if AUTH_ERROR:
        raise RuntimeError(AUTH_ERROR)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map="auto",
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.use_cache = False
    return PeftModel.from_pretrained(model, adapter_repo_id, token=HF_TOKEN)


def get_pipeline():
    global MODEL, TOKENIZER
    if MODEL is None or TOKENIZER is None:
        TOKENIZER = load_tokenizer(BASE_MODEL)
        MODEL = load_model(BASE_MODEL, ADAPTER_REPO_ID)
    return MODEL, TOKENIZER


@torch.inference_mode()
def respond(message: str, history):
    model, tokenizer = get_pipeline()
    prompt = build_prompt(message)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = generated.split("[/INST]", 1)[-1].strip() if "[/INST]" in generated else generated.strip()
    return answer


def submit_message(message: str, history: list[list[str]]):
    history = history or []
    answer = respond(message, history)
    history.append([message, answer])
    return "", history


with gr.Blocks() as demo:
    gr.Markdown(
        "# Medical Llama Clinical Q&A\n"
        "A compact demo that answers everyday medical questions carefully and concisely.\n"
        "The model is for educational use only and should not replace professional advice."
    )
    if AUTHENTICATED_USER:
        gr.Markdown(f"Authenticated Hugging Face user: `{AUTHENTICATED_USER}`")
    if AUTH_ERROR:
        gr.Markdown(
            "### Hugging Face authentication error\n"
            f"{AUTH_ERROR}\n\n"
            "Update the Space secret `HF_TOKEN` with the approved token for the account that has access to the gated model."
        )
    if not AUTH_ERROR:
        chatbot = gr.Chatbot(label="Clinical Q&A Assistant", height=420)
        message = gr.Textbox(
            label="Your question",
            placeholder="Ask a medical question...",
            lines=2,
        )
        send = gr.Button("Send")
        message.submit(submit_message, inputs=[message, chatbot], outputs=[message, chatbot])
        send.click(submit_message, inputs=[message, chatbot], outputs=[message, chatbot])
    else:
        gr.Markdown(
            "The chat demo is disabled until the Hugging Face token is fixed."
        )


if __name__ == "__main__":
    if os.getenv("SPACE_ID") or os.getenv("SPACE_HOST") or os.getenv("SPACE_REPO_ID"):
        demo.launch(share=True)
    else:
        demo.launch(share=True, server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
