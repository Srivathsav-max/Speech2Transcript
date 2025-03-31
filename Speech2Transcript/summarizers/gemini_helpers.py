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
You are a medical summarization expert. I need you to create a comprehensive, coherent narrative summary of the following healthcare conversation between a care manager and a patient.

Patient Name: {basic_info['patient_name']}
Provider: {basic_info['provider_name']}
Key Conditions: {', '.join(basic_info['conditions']) if basic_info['conditions'] else 'Unknown'}
Medications: {', '.join(basic_info['medications']) if basic_info['medications'] else 'Unknown'}

Here's the transcript:
```
{conversation_text}
```

Please create a professional, cohesive narrative summary that:
1. Identifies who called whom and for what purpose
2. Summarizes the patient's reported health status, symptoms, and recent events
3. Covers key vitals, medications and adherence
4. Includes information about diet, exercise, and other lifestyle factors
5. Notes preventive care status and any barriers to care
6. Concludes with follow-up plans

The summary should be 4-5 paragraphs in length, written in third-person narrative style like a medical note. Use a professional tone and focus on medically relevant information. Do not use bullet points or structured headings - write it as a continuous narrative similar to a medical progress note.
"""
    elif prompt_type == "extract_entities":
        return f"""
You are a medical information extraction specialist. I need you to extract specific information from the following healthcare conversation between a care manager and a patient. Be precise and only extract what is clearly stated in the text.

Here's the transcript:
```
{conversation_text}
```

Please extract the following information in JSON format:
1. Patient name (full name if available)
2. Provider/doctor name (with title)
3. Medical conditions mentioned (list all clearly stated conditions)
4. Medications mentioned (list names and dosages if available)
5. Vital signs (include blood pressure, glucose levels, weight, or any other measurements mentioned)
6. Lifestyle information (diet, exercise, smoking status, alcohol consumption)
7. Follow-up plans (appointments, tests, calls)

IMPORTANT:
- Only include information explicitly mentioned in the transcript
- Use "Unknown" for fields where no information is provided
- Do not make assumptions or guess information
- Format your response as a clean JSON object
- Include a confidence level (high, medium, low) for each extracted entity
- Be especially careful with names and medications to avoid hallucinations

Return your answer as a structured JSON object without any additional explanations.
"""
    elif prompt_type == "medication_adherence":
        return f"""
Analyze the following healthcare conversation between a care manager and a patient to assess medication adherence. Focus only on what is explicitly stated about taking medications as prescribed.

Here's the transcript:
```
{conversation_text}
```

Based solely on the information in this transcript:
1. Is the patient taking their medications as prescribed?
2. Are there any challenges or barriers to medication adherence mentioned?
3. Are there any specific medications that the patient struggles to take regularly?
4. What is the patient's attitude toward their medication regimen?

Return a concise assessment of the patient's medication adherence using only information explicitly mentioned in the transcript. Rate the adherence as "Good", "Variable", "Poor", or "Unknown" with a brief explanation. Do not make assumptions beyond what is directly stated.
"""
    else:
        return f"""
Analyze this healthcare conversation between a care manager and a patient:
```
{conversation_text}
```

Provide a concise, factual summary focusing on the key medical information.
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
You are a medical data extraction expert. Extract structured information from this healthcare conversation, returning ONLY valid JSON.

Conversation:
```
{conversation_text}
```

Extract the following information:
1. patient_name: The patient's full name if available (default to "Patient" if unclear)
2. provider_name: The healthcare provider's name with title if mentioned
3. conditions: Array of medical conditions mentioned (empty array if none)
4. medications: Array of objects with name and dosage properties
5. vital_signs: Object with properties like blood_pressure, glucose, weight, etc.
6. lifestyle: Object with exercise, diet, smoking_status properties
7. follow_up: Object with next_appointment, action_items properties

IMPORTANT RULES:
- Only extract information that is explicitly stated in the conversation
- Use null for unknown values
- Never hallucinate or invent information
- Include a confidence property (high, medium, or low) for each extracted field
- Format your response as valid, parseable JSON without ANY explanatory text before or after
- Do not include code markup tags (like ```json)
- Focus on accuracy over comprehensiveness

Return only the following JSON structure, properly filled in:
"""

    # Add the JSON schema template separately (not as part of the f-string)
    json_schema = """
{
  "patient_name": {"value": "string", "confidence": "string"},
  "provider_name": {"value": "string", "confidence": "string"},
  "conditions": {
    "list": ["string"],
    "confidence": "string"
  },
  "medications": {
    "list": [
      {"name": "string", "dosage": "string"}
    ],
    "adherence": "string",
    "confidence": "string"
  },
  "vital_signs": {
    "blood_pressure": "string",
    "glucose": 0,
    "weight": "string",
    "other": {},
    "confidence": "string"
  },
  "lifestyle": {
    "exercise": "string",
    "diet": "string",
    "smoking_status": "string",
    "confidence": "string"
  },
  "follow_up": {
    "next_appointment": "string",
    "action_items": ["string"],
    "confidence": "string"
  }
}
"""

    # Combine the prompt and schema
    return prompt + json_schema