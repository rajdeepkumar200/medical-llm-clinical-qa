from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset
from datasets.exceptions import DatasetNotFoundError
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import torch

from .config import ADAPTER_DIR, BASE_MODEL, DATASET_NAME, DATASET_SPLIT, LORA_CONFIG, MAX_SEQ_LENGTH, OUTPUT_DIR
from .prompts import format_medical_example


FALLBACK_MEDICAL_EXAMPLES = [
    {
        "instruction": "What are the common side effects of amoxicillin?",
        "input": "",
        "output": "Common side effects include nausea, diarrhea, rash, and mild stomach upset. Seek care quickly for swelling, trouble breathing, or severe rash.",
    },
    {
        "instruction": "How do I recognize signs of dehydration in a child?",
        "input": "",
        "output": "Warning signs include dry mouth, fewer wet diapers, sunken eyes, lethargy, and no tears when crying. Severe dehydration needs urgent medical care.",
    },
    {
        "instruction": "What is the first-line treatment for seasonal allergic rhinitis?",
        "input": "",
        "output": "Intranasal corticosteroids are commonly first-line. Antihistamines may also help, along with avoiding triggers when possible.",
    },
    {
        "instruction": "When should chest pain be treated as an emergency?",
        "input": "",
        "output": "Chest pain with shortness of breath, sweating, fainting, nausea, or pain spreading to the arm, jaw, or back should be treated as an emergency.",
    },
    {
        "instruction": "What should I do for a mild fever at home?",
        "input": "",
        "output": "Rest, drink fluids, and use acetaminophen or ibuprofen if appropriate. Get medical advice if fever is high, persistent, or linked with serious symptoms.",
    },
    {
        "instruction": "How can I reduce the risk of type 2 diabetes?",
        "input": "",
        "output": "Maintain a healthy weight, stay physically active, eat a balanced diet, and follow up regularly with a clinician if you have risk factors.",
    },
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a lightweight instruct model with QLoRA for medical Q&A.")
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--dataset-split", default=DATASET_SPLIT)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--adapter-dir", default=str(ADAPTER_DIR))
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    return parser


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_training_dataset(dataset_name: str, dataset_split: str):
    try:
        return load_dataset(dataset_name, split=dataset_split)
    except DatasetNotFoundError:
        return Dataset.from_list(FALLBACK_MEDICAL_EXAMPLES)


def load_training_model(model_name: str):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.use_cache = False
    return prepare_model_for_kbit_training(model)


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=LORA_CONFIG.r,
        lora_alpha=LORA_CONFIG.lora_alpha,
        lora_dropout=LORA_CONFIG.lora_dropout,
        bias=LORA_CONFIG.bias,
        task_type=LORA_CONFIG.task_type,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    tokenizer = load_tokenizer(args.base_model)
    model = load_training_model(args.base_model)

    dataset = load_training_dataset(args.dataset_name, args.dataset_split)
    columns = dataset.column_names

    def to_text(example):
        instruction = example.get("instruction") or example.get("question") or "Answer the medical question clearly."
        input_text = example.get("input") or example.get("context") or ""
        response = example.get("output") or example.get("answer") or example.get("response") or ""
        return {"text": format_medical_example(instruction, input_text, response)}

    dataset = dataset.map(to_text, remove_columns=columns)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=build_lora_config(),
        args=training_args,
        processing_class=tokenizer,
        formatting_func=lambda example: example["text"],
    )
    trainer.train()

    output_dir = Path(args.adapter_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
