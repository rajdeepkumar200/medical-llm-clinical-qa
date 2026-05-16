from __future__ import annotations

from typing import Iterable

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import BASE_MODEL, MAX_NEW_TOKENS, REPETITION_PENALTY, TEMPERATURE, TOP_P
from .prompts import build_chat_prompt


def load_tokenizer(model_name: str = BASE_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(base_model: str = BASE_MODEL, adapter_path: str | None = None):
    # Try 4-bit quantization if available; fall back to standard loading for Space/CPU environments
    quantization_config = None
    try:
        if torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
    except Exception:
        pass  # Fall back to standard loading

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.use_cache = False

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    return model


@torch.inference_mode()
def generate_answer(model, tokenizer, question: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    prompt = build_chat_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated.split("[/INST]", 1)[-1].strip() if "[/INST]" in generated else generated.strip()
