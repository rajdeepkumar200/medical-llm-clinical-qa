
# 🩺 Clinical AI Assistant

> A fine-tuned, region-aware Medical Q&A system built with **QLoRA**, **PEFT**, and **Gradio**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.44.1-orange)](https://gradio.app)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Spaces-yellow)](https://huggingface.co/spaces)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A lightweight, production-ready Medical LLM that answers clinical questions with **region-specific healthcare context**. Built on `TinyLlama-1.1B-Chat-v1.0` (or upgradeable to 7B models) with 4-bit QLoRA quantization for efficient fine-tuning and inference.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🌍 **Region-Aware Responses** | Auto-detects or lets users select their country (US, UK, Canada, India, Australia, NZ, Singapore, Hong Kong) for localized healthcare context and medicine names. |
| 🧠 **QLoRA Fine-Tuning** | Efficient adapter training with 4-bit quantization via PEFT + TRL. |
| 💬 **Streaming Chat UI** | Dark-themed Gradio interface with real-time token streaming, sidebar history, and mobile responsiveness. |
| 📄 **Lab Report OCR** | Upload images or PDFs; OCR extracts text for analysis with medical-report validation. |
| 🔍 **Wikipedia Grounding** | Short/ambiguous queries are grounded with Wikipedia summaries to reduce hallucinations. |
| 🏥 **NLP Symptom Detection** | Built-in medical term extraction and symptom-disease associations. |
| ⚡ **CPU & GPU Support** | Runs on CPU with fallback; CUDA automatically enables 4-bit quantization. |

---

## 📁 Project Structure

```
medical-llm/
├── app.py                          # Hugging Face Spaces entry point
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Package metadata
├── adapters/                       # Saved LoRA adapters
├── outputs/                        # Training outputs
├── hf_space_medical_llama/         # HF Space deployment files
└── src/medical_llm/
    ├── app.py                      # Gradio chat UI (dark theme, sidebar, streaming)
    ├── train.py                    # QLoRA fine-tuning script
    ├── infer.py                    # Model loading & text generation
    ├── config.py                   # Central configuration & hyperparameters
    ├── prompts.py                  # Chat prompt builders (region-aware)
    ├── nlp_processor.py            # Symptom detection & medical NLP
    ├── ocr.py                      # Lab report OCR (Tesseract + PyMuPDF)
    └── web_context.py              # Wikipedia grounding for ambiguous queries
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/rajdeepkumar200/medical-llm-clinical-qa.git
cd medical-llm-clinical-qa

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Chat App

```bash
python -m src.medical_llm.app
```

Open your browser to `http://localhost:7860`.

### 3. Fine-Tune Your Own Adapter

```bash
python -m src.medical_llm.train
```

---

## 🌍 Region Customization

The app automatically suggests a region based on browser timezone and GPS (with user permission), or you can manually select:

- **United States** — FDA terminology & generic names
- **United Kingdom** — NHS terminology
- **Canada** — Health Canada / provincial coverage notes
- **Australia** — TGA / PBS references
- **India** — Indian generic + optional brand-name examples
- **New Zealand, Singapore, Hong Kong** — Localized terminology

Responses adapt medicine names, healthcare standards, and availability notes accordingly.

---

## 🧠 Model Configuration

All settings live in `src/medical_llm/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `BASE_MODEL` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Base LLM to fine-tune |
| `MAX_NEW_TOKENS` | `220` | Response length (higher = longer answers) |
| `TEMPERATURE` | `0.35` | Lower = more factual |
| `TOP_P` | `0.9` | Nucleus sampling |
| `REPETITION_PENALTY` | `1.15` | Reduces repetitive output |
| `REGION` | `"General"` | Default healthcare region |

### Switch to a Larger Model

For better accuracy at the cost of more VRAM:

```bash
# Linux/macOS
export BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"

# Windows PowerShell
$env:BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
```

| Model | Speed | Accuracy | VRAM Required |
|-------|-------|----------|---------------|
| TinyLlama 1.1B | ⚡ Fast | Good | ~4GB (CPU works) |
| Mistral 7B | Moderate | Better | ~16GB+ (GPU recommended) |

---

## 📊 Training with Custom Data

### 1. Prepare Your Dataset

Create a CSV with columns: `instruction`, `input`, `output`

```csv
instruction,input,output
What are the symptoms of dehydration?,"","Signs include dry mouth, thirst, dark urine, dizziness, and fatigue. Severe dehydration requires urgent care."
```

### 2. Upload to Hugging Face Hub

Upload as a dataset: `your-username/your-medical-qa-dataset`

### 3. Update Config

In `src/medical_llm/config.py`:

```python
DATASET_NAME = "your-username/your-medical-qa-dataset"
ADAPTER_REPO_ID = "your-username/medical-adapter"
```

### 4. Train

```bash
python -m src.medical_llm.train --num-train-epochs 3 --learning-rate 2e-4
```

### 5. Deploy

The adapter automatically loads from Hugging Face at inference time.

---

## 🔧 Advanced Usage

### Environment Variables

All `config.py` values can be overridden via environment variables:

```bash
export BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
export MAX_NEW_TOKENS=512
export TEMPERATURE=0.3
export REGION="India"
export DATASET_NAME="medalpaca/medalpaca-40k"
```

### Adjust Generation Quality

| Parameter | Range | Effect |
|-----------|-------|--------|
| `MAX_NEW_TOKENS` | 128–512 | Longer responses vs. faster generation |
| `TEMPERATURE` | 0.3–0.9 | 0.3 = very factual, 0.9 = more varied |
| `TOP_P` | 0.7–0.95 | Lower = more focused, higher = more diverse |

---

## 📈 Evaluation

Training metrics from the reference run:

| Metric | Value |
|--------|-------|
| Dataset | 6 fallback medical examples (auto-fallback when Hub unavailable) |
| Epochs | 1 |
| Final Train Loss | `3.417` |
| Mean Token Accuracy | `0.3951` |
| Runtime | `156.2s` |

> **Note:** This is a reproducible pipeline demo, not a clinical benchmark. The goal is to demonstrate end-to-end adapter training, not SOTA medical accuracy.

---

## 🏗️ Architecture Highlights

### Hallucination Reduction

- **Greedy decoding** (`do_sample=False`) for deterministic, factual medical answers
- **Wikipedia grounding** for short/ambiguous queries (e.g., "diabetic symptoms")
- **Deterministic clarifiers** appended after streaming: *"Did you mean [topic]?"*
- **Medical report validation** — OCR text is scored against medical terms; non-medical uploads are rejected

### Prompt Engineering

The flexible prompt system in `prompts.py` does **not** force fixed sections. The model only includes a section if the user actually asked about that topic — preventing hallucinated drug names on unrelated queries.

### OCR Pipeline

- Supports `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`, `.gif`, `.pdf`
- Tesseract OCR with PyMuPDF fallback
- Medical-report validation via keyword + unit-pattern scoring
- Graceful degradation if OCR libraries are missing

---

## 🌐 Deployment

### Hugging Face Spaces

This project is ready to deploy on Hugging Face Spaces with Gradio:

```bash
# From the hf_space_medical_llama/ folder
# Follow HF Spaces deployment instructions
```

### Docker (optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "src.medical_llm.app"]
```

---

## 🛡️ Safety Disclaimer

> **This is a research and educational project only.**
>
> Medical answers generated by this system should **always** be reviewed by a qualified healthcare professional before any real-world diagnosis or treatment decisions. Do not rely on this tool for emergency medical situations.

---

## 📦 Dependencies

Key libraries:

- `torch` — PyTorch for model inference
- `transformers>=4.41.0` — Hugging Face model loading
- `peft>=0.11.1` — LoRA/QLoRA adapters
- `trl>=0.9.6` — SFT training
- `bitsandbytes>=0.43.1` — 4-bit quantization
- `gradio==4.44.1` — Chat UI
- `datasets>=2.19.0` — Dataset loading
- `pytesseract>=0.3.10` — OCR
- `pymupdf>=1.24.0` — PDF parsing

See `requirements.txt` for the full list.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- [TinyLlama](https://github.com/jzhang38/TinyLlama) by Zhang et al. for the efficient base model
- [Hugging Face](https://huggingface.co) for PEFT, TRL, and the Hub ecosystem
- [MedAlpaca](https://huggingface.co/medalpaca) for the medical training dataset
- [Wikipedia](https://www.wikipedia.org) for the open grounding API

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or PRs for:

- Additional region support
- Better hallucination mitigation
- UI/UX improvements
- Expanded medical NLP coverage

---

## 📬 Contact

- **Author:** [Rajdeep Kumar](https://github.com/rajdeepkumar200)
- **Hugging Face:** [@RajdeepSingh-ai](https://huggingface.co/RajdeepSingh-ai)
