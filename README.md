---
title: Medical Clinical Q&A
emoji: ⚕️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.36.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# Medical Llama Clinical Q&A

I built this project as a compact, practical demo of LLM fine-tuning and deployment. It uses `TinyLlama-1.1B-Chat-v1.0` with 4-bit QLoRA, PEFT, and TRL, then serves the result through a small Gradio chat app.

## What’s in the repo

- `src/medical_llm/train.py` trains the adapter.
- `src/medical_llm/app.py` runs the chat interface.
- `src/medical_llm/infer.py` handles loading and generation.
- `src/medical_llm/config.py` keeps the knobs in one place.
- `src/medical_llm/prompts.py` formats the medical instruction prompts.

## Why the setup looks this way

- The base model is small enough to run on modest hardware.
- QLoRA keeps the adapter light and fast to publish.
- The training script falls back to a small local medical set when the Hub dataset is unavailable, so the workflow still runs end to end.
- The public adapter repo and Space make the project easy to share in a portfolio review.

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

To train:

```bash
python -m src.medical_llm.train
```

To run the chat app:

```bash
python -m src.medical_llm.app
```

## Published artifacts

- Model repo: `RajdeepSingh-ai/medical-llama-medical-qa`
- Space repo: to be published from the space folder in this workspace

## Quick evaluation

This is a small portfolio demo, so the evaluation is intentionally lightweight and honest.

- Training run: 1 epoch on 6 fallback medical examples
- Final train loss: `3.417`
- Mean token accuracy: `0.3951`
- Runtime: `156.2s`

That is enough to show the pipeline works, the adapter trains, and the project is reproducible. It is not meant to be claimed as a clinical benchmark.

## Portfolio note

The point of this project is not just the model. It shows the full path from data formatting to adapter training to a public Hugging Face release and a live demo.

## Safety note

This is a research and educational project only. Medical answers should always be reviewed by a qualified professional before real-world use.
