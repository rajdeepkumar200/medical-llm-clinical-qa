from dataclasses import dataclass
from pathlib import Path
import os

BASE_MODEL = os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DATASET_NAME = os.getenv("DATASET_NAME", "medalpaca/medalpaca-40k")
DATASET_SPLIT = os.getenv("DATASET_SPLIT", "train")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs/medical-tinyllama-medical-qa"))
ADAPTER_DIR = Path(os.getenv("ADAPTER_DIR", "adapters/medical-tinyllama-medical-qa"))
ADAPTER_REPO_ID = os.getenv("ADAPTER_REPO_ID", "RajdeepSingh-ai/medical-tinyllama-medical-qa")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "1024"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.05"))
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a careful medical assistant. Give concise, evidence-aware answers and note when professional care is needed.",
)


@dataclass(frozen=True)
class LoraConfigValues:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


LORA_CONFIG = LoraConfigValues()
