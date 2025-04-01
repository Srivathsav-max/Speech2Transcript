import re
from typing import Dict, Any, List, Tuple, Optional, Set

def extract_conversation_structure(segments: List[Dict[str, Any]],
                       text_column: str = "transcription",
                       speaker_column: str = "speaker") -> Dict[str, Any]:
    """
    Extract and structure conversation data from transcript segments.

    Args:
        segments: List of transcript segments
        text_column: Name of the column containing the transcript text
        speaker_column: Name of the column containing the speaker identifier

    Returns:
        Dictionary with extracted conversation structure for Gemini
    """
    full_text = ""
    speaker_texts = {}
    conversation = []
    speaker_times = {}

    for segment in segments:
        if text_column not in segment or speaker_column not in segment:
            continue

        speaker = segment[speaker_column]
        text = segment.get(text_column, "")

        if not text or not speaker:
            continue

        # Add to full text
        full_text += f"{speaker}: {text}\n"

        # Add to speaker-specific text
        if speaker not in speaker_texts:
            speaker_texts[speaker] = []
        speaker_texts[speaker].append(text)

        # Add to conversation timeline
        conversation.append((speaker, text))

        # Track speaking time
        if "start" in segment and "end" in segment:
            duration = segment["end"] - segment["start"]
            if speaker not in speaker_times:
                speaker_times[speaker] = 0
            speaker_times[speaker] += duration

    # Convert speaker text lists to strings
    speaker_texts = {speaker: " ".join(texts) for speaker, texts in speaker_texts.items()}

    return {
        "full_text": full_text,
        "speaker_texts": speaker_texts,
        "conversation": conversation,
        "speaker_times": speaker_times
    }

def identify_speakers(conversation: List[tuple]) -> Dict[str, str]:
    """
    Identify the likely roles of speakers (patient vs care manager).

    Args:
        conversation: List of (speaker, text) tuples

    Returns:
        Dictionary mapping speaker IDs to roles
    """
    # Count questions by speaker
    question_count = {}
    total_words = {}

    for speaker, text in conversation:
        # Count questions by speaker
        if "?" in text:
            question_count[speaker] = question_count.get(speaker, 0) + 1

        # Count total words by speaker
        words = len(text.split())
        total_words[speaker] = total_words.get(speaker, 0) + words

    # Determine care manager (person who asks more questions)
    if not question_count:
        # If no questions found, the care manager likely speaks less
        care_manager = min(total_words.items(), key=lambda x: x[1])[0] if total_words else None
    else:
        care_manager = max(question_count.items(), key=lambda x: x[1])[0]

    # Determine patient (other speakers)
    speakers = list(set(s for s, _ in conversation))
    patient = next((s for s in speakers if s != care_manager), None)

    return {
        "care_manager": care_manager,
        "patient": patient
    }

def generate_gemini_prompt(conversation_text: str, basic_info: Dict[str, Any], prompt_type: str = "summary") -> str:
    """
    Generate prompt templates for different extraction tasks with Gemini.
    
    Args:
        conversation_text: Full conversation text
        basic_info: Dictionary with basic extracted information
        prompt_type: Type of prompt to generate
        
    Returns:
        Formatted prompt for Gemini
    """
    if prompt_type == "summary":
        return f"""
Write a direct, concise medical summary of this telehealth conversation. Begin immediately with the summary content - do not include any introductory phrases like "based on the transcript" or similar preambles.

PATIENT INFO:
- Name: {basic_info['patient_name']}
- Provider: {basic_info['provider_name']}
- Conditions: {', '.join(basic_info['conditions']) if basic_info['conditions'] else 'Unknown'}
- Medications: {', '.join(basic_info['medications']) if basic_info['medications'] else 'Unknown'}

CONVERSATION:
```
{conversation_text}
```

REQUIREMENTS:
- Start immediately with "Patient [name] had a telehealth appointment with [provider]..."
- Write a continuous narrative (no bullet points or headers)
- Include: call purpose, health status, symptoms, vitals, medications, adherence
- Mention lifestyle factors (diet, exercise) if discussed
- End with follow-up plans
- Use professional, clinical tone
- 4-5 paragraphs in third-person perspective
- Be factual and precise - only include information from the conversation
"""
    elif prompt_type == "extract_entities":
        return f"""
Extract medical entities from this telehealth conversation directly into JSON format. Do not include explanatory text before or after the JSON.

CONVERSATION:
```
{conversation_text}
```

REQUIREMENTS:
- Extract only explicitly mentioned information (no assumptions)
- Include: patient name, provider name, conditions, medications, vital signs
- Always use "Unknown" for missing fields, never leave fields empty
- Assign confidence level (high/medium/low) to each entity
- Focus on precision over completeness
- Return valid, parseable JSON only
"""
    elif prompt_type == "medication_adherence":
        return f"""
Assess medication adherence from this telehealth conversation. Provide a direct analysis without preamble phrases.

CONVERSATION:
```
{conversation_text}
```

REQUIREMENTS:
- Determine if patient is taking medications as prescribed (Good/Variable/Poor/Unknown)
- Identify specific adherence challenges mentioned
- Note any medications with adherence issues
- Assess patient's attitude toward medication regimen
- Base assessment only on explicitly stated information
- Provide short, precise assessment
"""
    else:
        return f"""
Provide a factual, concise summary of this healthcare conversation directly, with no introductory phrases.

CONVERSATION:
```
{conversation_text}
```

REQUIREMENTS:
- Start immediately with the key points
- Focus only on medical information
- Use professional, clinical tone
- Be brief and factual
- Include only information directly stated in the conversation
"""

def extract_basic_info(conversation: List[tuple], speakers: Dict[str, str]) -> Dict[str, Any]:
    """
    Extract basic information from conversation to provide context for Gemini.

    Args:
        conversation: List of (speaker, text) tuples
        speakers: Dictionary mapping speaker IDs to roles

    Returns:
        Dictionary with basic extracted information
    """
    info = {
        "patient_name": "Patient",
        "provider_name": "Provider",
        "conditions": [],
        "medications": [],
        "vital_signs": {}
    }

    # Extract patient and provider names from early conversation
    early_conversation = conversation[:10]
    patient_speaker = speakers.get("patient")
    care_manager = speakers.get("care_manager")

    # Look for patient name in care manager greetings
    for speaker, text in early_conversation:
        if speaker == care_manager:
            # Check for greeting patterns with names
            name_match = re.search(r'(?:hello|hi|good\s+(?:morning|afternoon|evening))\s+(?:mr\.|mrs\.|ms\.|miss)?\s*([A-Z][a-z]+)', text)
            if name_match:
                info["patient_name"] = name_match.group(1)

            # Look for provider reference
            provider_match = re.search(r"(?:from|with)\s+(?:Dr\.|Doctor)\s+([A-Z][a-z]+)", text)
            if provider_match:
                info["provider_name"] = f"Dr. {provider_match.group(1)}"

    # Look for common conditions
    condition_patterns = {
        "diabetes": r'\b(?:diabetes|diabetic|blood\s+sugar|glucose|insulin|a1c)\b',
        "hypertension": r'\b(?:hypertension|high\s+blood\s+pressure|blood\s+pressure)\b',
        "heart disease": r'\b(?:heart\s+(?:disease|failure|problem)|cardiac)\b',
        "obesity": r'\b(?:obesity|weight\s+(?:issue|problem|loss|management))\b'
    }

    # Extract conditions
    full_text = " ".join(text for _, text in conversation).lower()
    for condition, pattern in condition_patterns.items():
        if re.search(pattern, full_text):
            info["conditions"].append(condition)

    # Extract medications
    med_patterns = [
        r'\b(?:taking|take|use|on)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b',
        r'\b(?:medication|medicine|pill|drug)\s+(?:called|named)?\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b'
    ]

    # Common medications to look for
    common_meds = ["insulin", "metformin", "ozempic", "lisinopril", "atorvastatin", "aspirin"]

    for med in common_meds:
        if re.search(r'\b' + med + r'\b', full_text, re.IGNORECASE):
            info["medications"].append(med)

    # Look for blood pressure values
    bp_matches = re.finditer(r'(\d{2,3})[/\\](\d{2,3})', full_text)
    for match in bp_matches:
        systolic = int(match.group(1))
        diastolic = int(match.group(2))
        if 80 <= systolic <= 200 and 40 <= diastolic <= 120:
            info["vital_signs"]["blood_pressure"] = f"{systolic}/{diastolic}"
            break

    # Look for glucose values
    glucose_match = re.search(r'(?:glucose|sugar)\s+(?:is|of|at|around|about)?\s*(\d{2,3})', full_text)
    if glucose_match:
        glucose = int(glucose_match.group(1))
        if 50 <= glucose <= 400:
            info["vital_signs"]["glucose"] = glucose

    return info

def prepare_gemini_extraction_prompt(conversation_text: str) -> str:
    """
    Prepare a structured prompt for Gemini to extract detailed medical information
    
    Args:
        conversation_text: The full conversation text
        
    Returns:
        A formatted prompt for entity extraction
    """
    # Create the initial part of the prompt with the conversation text
    prompt = f"""
Return only valid JSON with structured medical data from this telehealth conversation.

CONVERSATION:
```
{conversation_text}
```

REQUIREMENTS:
- Extract patient name, provider name, conditions, medications, vital signs
- Include only explicitly stated information
- Use "Unknown" for any missing information
- Assign confidence levels (high/medium/low)
- Return only valid JSON
"""

    # Add the JSON schema template separately (not as part of the f-string)
    json_schema = """
{
  "is_telemedical": true,
  "patient_name": {"value": "Patient name", "confidence": "high/medium/low"},
  "provider_name": {"value": "Provider name", "confidence": "high/medium/low"},
  "conditions": {
    "list": ["condition1", "condition2"],
    "confidence": "high/medium/low"
  },
  "medications": {
    "list": [
      {"name": "medication name", "dosage": "dosage info"}
    ],
    "adherence": "Good/Variable/Poor/Unknown",
    "confidence": "high/medium/low"
  },
  "vital_signs": {
    "blood_pressure": "systolic/diastolic",
    "glucose": 0,
    "weight": "value with units",
    "confidence": "high/medium/low"
  },
  "follow_up": {
    "next_appointment": "date/time info",
    "action_items": ["item1", "item2"],
    "confidence": "high/medium/low"
  }
}
"""

    # Combine the prompt and schema
    return prompt + json_schema