from __future__ import annotations

import logging
from threading import Thread
from typing import Iterable, Iterator

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from .config import BASE_MODEL, MAX_NEW_TOKENS, REPETITION_PENALTY, TEMPERATURE, TOP_P
from .prompts import build_chat_prompt

logger = logging.getLogger(__name__)


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
            logger.info("CUDA available, attempting 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
    except Exception as e:
        logger.warning(f"4-bit quantization not available: {e}, falling back to standard loading")
        pass  # Fall back to standard loading

    logger.info(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    # KV cache is REQUIRED for fast inference. The training pipeline disables it
    # for gradient checkpointing; we must turn it back on here.
    model.config.use_cache = True
    model.eval()
    logger.info("Base model loaded successfully (use_cache=True, eval mode)")

    if adapter_path:
        try:
            logger.info(f"Loading adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            logger.info("Adapter loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load adapter {adapter_path}: {e}. Using base model only.")
            # Continue with base model if adapter fails

    return model


@torch.inference_mode()
def stream_answer(
    model,
    tokenizer,
    question: str,
    region: str = "General",
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Iterator[str]:
    """Yield generated text incrementally so the UI can display tokens as they
    arrive. Uses `TextIteratorStreamer` running the model on a background thread.
    """
    prompt = build_chat_prompt(question, region=region)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=120.0,
    )

    # Greedy + low temperature: medical answers benefit from determinism, and
    # greedy is meaningfully faster than sampling on CPU.
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
        use_cache=True,
    )

    thread = Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
    thread.start()

    try:
        for chunk in streamer:
            if chunk:
                yield chunk
    finally:
        thread.join(timeout=1.0)


@torch.inference_mode()
def generate_answer(model, tokenizer, question: str, region: str = "General", max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    try:
        logger.info(f"Building prompt for question: {question[:50]}... (Region: {region})")
        prompt = build_chat_prompt(question, region=region)
        logger.info(f"Prompt: {prompt[:100]}...")
        
        logger.info(f"Tokenizing input, moving to device: {model.device}")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        logger.info(f"Input shape: {inputs['input_ids'].shape}")
        
        logger.info("Starting generation...")
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        logger.info(f"Generation complete, output shape: {output_ids.shape}")

        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logger.info(f"Decoded output (first 100 chars): {generated[:100]}...")
        
        result = generated.split("[/INST]", 1)[-1].strip() if "[/INST]" in generated else generated.strip()
        logger.info(f"Final result (first 100 chars): {result[:100]}...")
        return result
    except Exception as e:
        logger.error(f"Error in generate_answer: {e}", exc_info=True)
        raise
