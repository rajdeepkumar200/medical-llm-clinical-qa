from .config import SYSTEM_PROMPT


def format_medical_example(instruction: str, input_text: str, response: str) -> str:
    user_prompt = instruction.strip()
    if input_text.strip():
        user_prompt = f"{user_prompt}\n\nContext: {input_text.strip()}"

    return (
        "<s>[INST] <<SYS>>\n"
        f"{SYSTEM_PROMPT}\n"
        "<</SYS>>\n\n"
        f"{user_prompt} [/INST] {response.strip()}</s>"
    )


def build_chat_prompt(question: str) -> str:
    return (
        "<s>[INST] <<SYS>>\n"
        f"{SYSTEM_PROMPT}\n"
        "<</SYS>>\n\n"
        f"{question.strip()} [/INST]"
    )
