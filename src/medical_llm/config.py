from dataclasses import dataclass
from pathlib import Path
import os

# Model configuration - can be overridden via environment
# For better accuracy: use "mistralai/Mistral-7B-Instruct-v0.2" (requires more resources)
# For faster response: use "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BASE_MODEL = os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

DATASET_NAME = os.getenv("DATASET_NAME", "medalpaca/medalpaca-40k")
DATASET_SPLIT = os.getenv("DATASET_SPLIT", "train")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs/medical-tinyllama-medical-qa"))
ADAPTER_DIR = Path(os.getenv("ADAPTER_DIR", "adapters/medical-tinyllama-medical-qa"))
ADAPTER_REPO_ID = os.getenv("ADAPTER_REPO_ID", "RajdeepSingh-ai/medical-tinyllama-medical-qa")

MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "1024"))
# Lower defaults for CPU inference. Each token costs ~50-100ms on free HF Space;
# 220 keeps total generation under ~20s after the KV cache fix.
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "220"))
# Lower temperature => more deterministic medical answers AND slightly faster
# sampling. Combined with a tighter top_p this also cuts repetition loops.
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.35"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.15"))

# Region/Country for localized responses (can be overridden)
REGION = os.getenv("REGION", "General")  # Examples: "US", "UK", "Canada", "Australia", "India", "General"

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    f"""You are a knowledgeable and careful medical assistant. Your responses should be:
- Evidence-based and accurate
- Localized to {REGION} healthcare standards and practices where applicable
- Clear and understandable to non-medical audiences
- Appropriately cautious about when professional medical consultation is needed
- Concise but comprehensive

Always recommend consulting with a qualified healthcare professional for diagnosis and treatment decisions.""",
)


@dataclass(frozen=True)
class LoraConfigValues:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


LORA_CONFIG = LoraConfigValues()
