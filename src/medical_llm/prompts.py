from __future__ import annotations

from typing import Optional

from .config import SYSTEM_PROMPT


def build_region_aware_system_prompt(region: str = "General") -> str:
    """Build a region-aware system prompt with healthcare context.

    Important: this prompt is intentionally *flexible*. It does NOT force the
    model to emit a fixed list of sections (the previous version had a
    hard-coded "Recommended Medications" heading, which caused the small 1B
    model to hallucinate drug names for completely unrelated questions like
    "diabetic symptoms"). The model now only includes a section if the user
    actually asked about that topic.
    """
    region_context = {
        "United States": "U.S.",
        "United Kingdom": "U.K./NHS",
        "Canada": "Canadian",
        "Australia": "Australian",
        "New Zealand": "New Zealand",
        "India": "Indian",
        "Singapore": "Singapore",
        "Hong Kong": "Hong Kong",
    }
    region_text = region_context.get(region, region)

    return f"""You are a careful, plain-English medical assistant for {region_text} healthcare.

Rules:
- Stay strictly on the user's actual question. Do not invent unrelated topics.
- Use clear Markdown. Short bullet points. Define any medical term in parentheses the first time it appears, e.g. "hypertension (high blood pressure)".
- If the prompt contains "BEGIN OCR TEXT FROM UPLOADED FILE", that text is the user's uploaded image/PDF content. Use it directly. Never say you cannot access the image; instead mention OCR quality only if the extracted text is unclear.
- ONLY include a section if it is directly relevant to the question:
  - **Quick Summary** (always)
  - **What this means / likely causes** (when describing a condition or symptom)
  - **Recommended tests** (only if the user asks about diagnosis or testing)
  - **Treatment / medications** (only if the user asks about treatment — never invent drug names otherwise)
  - **When to see a doctor** (red flags, when relevant)
  - **Self-care tips** (when relevant)
- Never invent specific drug names, brand names, or dosages unless the user clearly asked about treatment.
- Keep the whole answer concise (under ~200 words unless the user asks for more).
- If the user's question is short, vague, or could mean multiple things, finish your answer with a single italic line:
  *Did you mean: <your best interpretation>? If not, please add more details.*
- End with a one-line reminder that this is educational, not a diagnosis."""


def format_medical_example(instruction: str, input_text: str, response: str) -> str:
    """Format a (instruction, input, response) tuple as a SFT training example.

    Kept in the older Llama-2 [INST] format because the training data uses this
    format. Inference uses the model's own chat template via
    :func:`build_chat_prompt`.
    """
    user_prompt = instruction.strip()
    if input_text.strip():
        user_prompt = f"{user_prompt}\n\nContext: {input_text.strip()}"

    return (
        "<s>[INST] <<SYS>>\n"
        f"{SYSTEM_PROMPT}\n"
        "<</SYS>>\n\n"
        f"{user_prompt} [/INST] {response.strip()}</s>"
    )


def build_chat_prompt(
    question: str,
    region: str = "General",
    tokenizer=None,
) -> str:
    """Build a prompt using the *model's own* chat template when a tokenizer is
    provided. This is critical for correctness: TinyLlama-Chat uses the Zephyr
    / ChatML format, Llama-3-Instruct uses its own format, etc. Using the
    wrong template (e.g. the old hard-coded ``[INST] <<SYS>>`` Llama-2 format)
    degrades the model to near-random output on small models.

    Falls back to the Llama-2 ``[INST]`` format only if no tokenizer is passed
    or the tokenizer does not define a chat template.
    """
    system_prompt = build_region_aware_system_prompt(region)
    user_msg = (question or "").strip()

    if tokenizer is not None:
        chat_template = getattr(tokenizer, "chat_template", None)
        apply = getattr(tokenizer, "apply_chat_template", None)
        if chat_template and callable(apply):
            try:
                return apply(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                # Some templates reject the "system" role; retry by prepending
                # the system prompt to the user message.
                try:
                    return apply(
                        [
                            {
                                "role": "user",
                                "content": f"{system_prompt}\n\n{user_msg}",
                            }
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    pass

    # Fallback: Llama-2 INST format.
    return (
        "<s>[INST] <<SYS>>\n"
        f"{system_prompt}\n"
        "<</SYS>>\n\n"
        f"{user_msg} [/INST]"
    )
