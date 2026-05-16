---
title: Medical Clinical Q&A
emoji: ⚕️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.11"
app_file: app.py
pinned: false
---

# Medical Llama Clinical Q&A

A fine-tuned LLM for region-aware medical Q&A. Uses `TinyLlama-1.1B-Chat-v1.0` with 4-bit QLoRA, PEFT, and TRL with Gradio interface. **Features region-specific healthcare context** (US, UK, Canada, India, etc.).

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

## Region Customization

The Space app includes a dropdown to select healthcare region:
- **United States** (FDA standards, common US practices)
- **United Kingdom** (NHS standards)
- **Canada** (Health Canada protocols)
- **Australia** (TGA standards)
- Plus: New Zealand, India, Singapore, Hong Kong, or General

Responses will reference region-specific healthcare practices where applicable.

## Using a Larger Model for Better Accuracy

The default is TinyLlama (1.1B) for speed. For better accuracy, switch to 7B:

```bash
# Set environment variable or edit config.py:
export BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
```

**Trade-offs:**
- 1.1B (current): Fast, runs on CPU, lower accuracy
- 7B: Better accuracy, needs 16GB+ RAM or quantization

## Providing Custom Training Data

To fine-tune on your own medical Q&A data:

1. **Create a CSV with this format:**
   ```
   instruction,input,output
   What is hypertension,"",'High blood pressure occurs when...'
   ```

2. **Upload to Hugging Face Hub** as a dataset

3. **Update config.py:**
   ```python
   DATASET_NAME = "your_username/your-medical-qa-dataset"
   ADAPTER_REPO_ID = "your_username/medical-adapter"
   ```

4. **Train:**
   ```bash
   python -m src.medical_llm.train
   ```

5. **Deploy** - the adapter automatically pulls on inference

## Adjusting Response Quality

In `src/medical_llm/config.py`:
- `MAX_NEW_TOKENS`: 256-512 (higher = longer responses)
- `TEMPERATURE`: 0.7 (balanced), 0.3 (factual), 0.9 (creative)
- `TOP_P`: 0.95 (diverse), 0.9 (balanced), 0.7 (focused)

## Getting started (Local)

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
