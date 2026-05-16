"""NLP processor for symptom detection and medical term extraction."""

import re
from typing import List, Dict, Tuple

# Medical symptoms dictionary with aliases
SYMPTOMS_DB = {
    "fever": ["high temp", "temperature", "hot", "burning", "feverish"],
    "headache": ["head pain", "migraine", "throbbing head", "tension headache"],
    "cough": ["coughing", "persistent cough", "dry cough", "wet cough"],
    "cold": ["runny nose", "nasal congestion", "stuffy nose"],
    "sore throat": ["throat pain", "throat ache", "difficulty swallowing"],
    "fatigue": ["tiredness", "exhaustion", "weakness", "lethargy"],
    "nausea": ["feeling sick", "queasy", "vomiting", "throwing up"],
    "body ache": ["muscle pain", "joint pain", "body pain", "myalgia"],
    "diarrhea": ["loose stool", "loose motion", "stomach upset"],
    "constipation": ["difficulty passing stool", "hard stool"],
    "rash": ["skin irritation", "itchy skin", "skin outbreak"],
    "itching": ["itchiness", "pruritus", "skin itching"],
    "shortness of breath": ["breathing difficulty", "breathlessness", "dyspnea"],
    "chest pain": ["chest discomfort", "chest tightness"],
    "abdominal pain": ["stomach pain", "belly pain", "abdominal cramps"],
    "back pain": ["lower back pain", "backache"],
    "dizziness": ["vertigo", "lightheadedness", "feeling faint"],
    "insomnia": ["sleep disorder", "cannot sleep", "sleeping difficulty"],
    "anxiety": ["nervousness", "worry", "panic"],
    "depression": ["sadness", "low mood", "feeling down"],
}

# Disease-symptom associations
DISEASE_SYMPTOMS = {
    "common_cold": ["cough", "sore throat", "cold", "headache", "fatigue"],
    "flu": ["fever", "cough", "body ache", "fatigue", "headache"],
    "covid_19": ["fever", "cough", "shortness of breath", "fatigue", "loss of taste"],
    "allergies": ["itching", "rash", "cold", "cough"],
    "migraine": ["headache", "nausea", "dizziness"],
    "gastroenteritis": ["nausea", "diarrhea", "abdominal pain", "vomiting"],
    "anxiety": ["anxiety", "chest pain", "shortness of breath", "dizziness"],
    "asthma": ["shortness of breath", "cough", "chest tightness"],
    "bronchitis": ["cough", "chest pain", "shortness of breath"],
    "pneumonia": ["fever", "cough", "shortness of breath", "chest pain"],
}


def extract_symptoms(text: str) -> List[str]:
    """Extract medical symptoms from user text using pattern matching."""
    text_lower = text.lower()
    detected_symptoms = []
    
    # Direct symptom detection
    for symptom, aliases in SYMPTOMS_DB.items():
        if symptom in text_lower:
            detected_symptoms.append(symptom)
        else:
            # Check aliases
            for alias in aliases:
                if alias in text_lower:
                    detected_symptoms.append(symptom)
                    break
    
    return list(set(detected_symptoms))  # Remove duplicates


def extract_medical_terms(text: str) -> List[str]:
    """Extract medical terms and conditions from text."""
    medical_terms = []
    text_lower = text.lower()
    
    # Common medical conditions
    conditions = [
        "diabetes", "hypertension", "asthma", "arthritis", "cancer",
        "heart disease", "stroke", "pneumonia", "bronchitis", "covid",
        "flu", "cold", "migraine", "depression", "anxiety"
    ]
    
    for condition in conditions:
        if condition in text_lower:
            medical_terms.append(condition)
    
    return medical_terms


def suggest_diseases(symptoms: List[str]) -> Dict[str, float]:
    """Suggest possible diseases based on detected symptoms with confidence scores."""
    disease_scores = {}
    
    for disease, disease_symp_list in DISEASE_SYMPTOMS.items():
        # Calculate match percentage
        matched = sum(1 for s in symptoms if s in disease_symp_list)
        if matched > 0:
            confidence = (matched / len(disease_symp_list)) * 100
            disease_scores[disease.replace("_", " ").title()] = round(confidence, 1)
    
    # Sort by confidence
    return dict(sorted(disease_scores.items(), key=lambda x: x[1], reverse=True))


def process_user_input(text: str) -> Dict:
    """Process user input and extract medical information."""
    symptoms = extract_symptoms(text)
    terms = extract_medical_terms(text)
    suggested = suggest_diseases(symptoms)
    
    return {
        "symptoms": symptoms,
        "medical_terms": terms,
        "suggested_conditions": suggested,
        "has_medical_context": len(symptoms) > 0 or len(terms) > 0
    }


def build_context_prompt(nlp_result: Dict, original_question: str) -> str:
    """Build an enhanced prompt with NLP context for better responses."""
    context_parts = []
    
    if nlp_result["symptoms"]:
        symptoms_str = ", ".join(nlp_result["symptoms"])
        context_parts.append(f"Symptoms mentioned: {symptoms_str}")
    
    if nlp_result["medical_terms"]:
        terms_str = ", ".join(nlp_result["medical_terms"])
        context_parts.append(f"Medical context: {terms_str}")
    
    if nlp_result["suggested_conditions"]:
        top_suggestions = list(nlp_result["suggested_conditions"].items())[:3]
        suggestions_str = ", ".join([f"{cond} ({conf}%)" for cond, conf in top_suggestions])
        context_parts.append(f"Possible conditions: {suggestions_str}")
    
    if context_parts:
        context = "\n".join(context_parts)
        enhanced_prompt = f"{original_question}\n\n[Medical Context]\n{context}"
        return enhanced_prompt
    
    return original_question
