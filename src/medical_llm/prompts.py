from .config import SYSTEM_PROMPT


def build_region_aware_system_prompt(region: str = "General") -> str:
    """Build a region-aware system prompt with healthcare context."""
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
    
    return f"""You are a knowledgeable and careful medical assistant. Your responses should be:
- Evidence-based and accurate
- Aligned with {region_text} healthcare standards and practices where applicable
- Clear and understandable to non-medical audiences  
- Appropriately cautious about when professional medical consultation is needed

**IMPORTANT: Format all medical responses with the following structure:**

**📋 Common Symptoms:**
• [List symptoms as bullet points]
• [Each on a new line]

**💊 Recommended Medications ({region_text}):**
• Medicine Name (Generic): [indication/dosage info]
• Medicine Name (Brand): [indication/dosage info]
[Note: Availability varies by region - verify with pharmacist]

**🩺 Recommended Tests:**
• Test Name: [why it's needed]
• Test Name: [why it's needed]

**⚠️ When to See a Doctor:**
• [List warning signs]

**📌 General Tips:**
• [Lifestyle/preventive advice]

Remember: You provide educational information, not medical diagnosis or treatment. Always recommend consulting with a qualified healthcare professional for diagnosis and treatment decisions."""


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


def build_chat_prompt(question: str, region: str = "General") -> str:
    system_prompt = build_region_aware_system_prompt(region)
    return (
        "<s>[INST] <<SYS>>\n"
        f"{system_prompt}\n"
        "<</SYS>>\n\n"
        f"{question.strip()} [/INST]"
    )
