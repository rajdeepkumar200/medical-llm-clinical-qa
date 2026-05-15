---
title: Medical Llama Clinical Q&A
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
python_version: 3.11
app_file: app.py
pinned: false
---

# Medical Llama Clinical Q&A

This Space is a compact demo built to show a realistic LLM project end to end: gated model access, adapter fine-tuning, artifact publishing, and a live Gradio interface.

## What this demo does

- Loads the `meta-llama/Llama-3.2-1B-Instruct` base model.
- Applies the published LoRA adapter from `RajdeepSingh-ai/medical-llama-medical-qa`.
- Answers simple medical questions in a careful, concise style.

## Why it is useful in a portfolio

This is intentionally small and practical. It shows that the project is not just a notebook experiment. It has model access handling, a training pipeline, a published adapter, and a deployable interface that a recruiter can open quickly.

## Evaluation snapshot

This is a smoke-test evaluation, not a clinical benchmark.

- Training set size: 6 fallback examples
- Epochs: 1
- Final train loss: `3.417`
- Mean token accuracy: `0.3951`
- Runtime: `156.2s`

## Notes

This demo is for research and educational use only. Medical answers should always be reviewed by a qualified professional.

Last refreshed to trigger a new Space build.

## Required secret

Set `HF_TOKEN` in the Space secrets so the gated Llama base model can be loaded. This app also accepts `HUGGINGFACE_HUB_TOKEN` or `HUGGINGFACE_TOKEN` if you prefer one of those names.

## Runtime note

This Space should run on Python 3.11. The error shown in the logs comes from Python 3.13 removing `audioop`, which Gradio/PyDub still expects in this environment.
