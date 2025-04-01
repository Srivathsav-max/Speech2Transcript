"""
HIPAA Compliance Utilities for Medical Transcription

This module provides utilities for ensuring HIPAA compliance in medical 
transcription and summary generation.
"""

import re
from typing import Dict, Any, List, Set, Tuple, Optional
import logging
import json
import random
import string

logger = logging.getLogger(__name__)

# HIPAA identifiers that need protection
HIPAA_IDENTIFIERS = {
    "names": r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',  # Names (First Last)
    "locations": r'\b(?:[A-Z][a-z]+\s+)?(?:Street|Avenue|Road|Drive|Lane|Place|Court|Blvd|Boulevard|Way|Circle|Hwy|Highway)\b',  # Addresses
    "dates": r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:[,\s]+\d{4})?\b',  # Dates
    "phone_numbers": r'\b\(?(?:\d{3})\)?[-.\s]?(?:\d{3})[-.\s]?(?:\d{4})\b',  # Phone numbers
    "emails": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
    "ids": r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',  # SSN-like identifiers
    "mrn": r'\b(?:MRN|Medical Record Number|Patient ID|Record Number)[\s:#]*\d+\b',  # Medical Record Numbers
}

# HIPAA compliance verification checklist
HIPAA_COMPLIANCE_CHECKLIST = [
    "Personally Identifiable Information (PII) should be redacted or de-identified",
    "Protected Health Information (PHI) should be properly safeguarded",
    "Only minimum necessary information should be included in summaries",
    "Patient consent should be documented before sharing information",
    "Information should be clinically accurate and professionally documented",
    "Access to patient data should be restricted to authorized personnel only"
]

def detect_hipaa_identifiers(text: str) -> Dict[str, List[str]]:
    """
    Detect potential HIPAA-protected identifiers in text.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary of categories with detected potential identifiers
    """
    results = {}
    
    for category, pattern in HIPAA_IDENTIFIERS.items():
        matches = re.findall(pattern, text)
        if matches:
            # Remove duplicates
            unique_matches = list(set(matches))
            results[category] = unique_matches
    
    return results

def redact_hipaa_identifiers(text: str, descriptive_redaction: bool = True) -> str:
    """
    Redact potential HIPAA-protected identifiers in text.
    
    Args:
        text: The text to redact
        descriptive_redaction: If True, use descriptive replacements instead of generic [REDACTED]
        
    Returns:
        Text with potential identifiers redacted
    """
    redacted_text = text
    
    # Define descriptive replacements for each category
    descriptive_replacements = {
        "names": "[PATIENT_NAME]", 
        "locations": "[ADDRESS]",
        "dates": "[DATE]",
        "phone_numbers": "[PHONE]",
        "emails": "[EMAIL]",
        "ids": "[ID_NUMBER]",
        "mrn": "[MEDICAL_RECORD_NUMBER]"
    }
    
    # Additional clinical terminology to replace with descriptive terms
    clinical_terms = {
        r'\bCC\b': "Chief Complaint",
        r'\bHPI\b': "History of Present Illness",
        r'\bROS\b': "Review of Systems",
        r'\bPMH\b': "Past Medical History",
        r'\bFH\b': "Family History",
        r'\bSH\b': "Social History",
        r'\bPE\b': "Physical Examination",
        r'\bFU\b': "Follow Up",
        r'\bD/C\b': "Discharge",
        r'\bDx\b': "Diagnosis",
        r'\bRx\b': "Prescription",
        r'\bTx\b': "Treatment",
        r'\bHx\b': "History",
        r'\bFx\b': "Fracture"
    }
    
    # First, replace PHI with appropriate descriptive terms
    for category, pattern in HIPAA_IDENTIFIERS.items():
        if descriptive_redaction:
            replacement = descriptive_replacements.get(category, "[REDACTED]")
            redacted_text = re.sub(pattern, replacement, redacted_text)
        else:
            redacted_text = re.sub(pattern, "[REDACTED]", redacted_text)
    
    # Next, replace abbreviated clinical terms with full terms if they appear redacted
    if descriptive_redaction:
        for abbrev, full_term in clinical_terms.items():
            # Look for the pattern [REDACTED] near clinical abbreviations
            redacted_pattern = r'\[REDACTED\]\s*' + abbrev
            replacement = f"{full_term}"
            redacted_text = re.sub(redacted_pattern, replacement, redacted_text)
            
            # Also check for abbreviation followed by [REDACTED]
            abbrev_redacted_pattern = abbrev + r'\s*\[REDACTED\]'
            redacted_text = re.sub(abbrev_redacted_pattern, replacement, redacted_text)
    
    return redacted_text

def generate_hipaa_compliant_id(real_name: str = None, length: int = 6) -> str:
    """
    Generate a HIPAA-compliant identifier for a patient.
    
    Args:
        real_name: Optional real name to base the ID on
        length: Length of random component
        
    Returns:
        HIPAA-compliant identifier
    """
    # Create random ID part
    random_part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
    
    # If real name provided, use initials
    if real_name:
        parts = real_name.split()
        initials = ''.join([p[0].upper() for p in parts if p])
        return f"{initials}-{random_part}"
    
    return f"PT-{random_part}"

def verify_hipaa_compliance(text: str) -> Dict[str, Any]:
    """
    Verify HIPAA compliance of a text by checking for potential issues.
    
    Args:
        text: The text to verify
        
    Returns:
        Dictionary with compliance information
    """
    identifiers = detect_hipaa_identifiers(text)
    has_identifiers = any(identifiers.values())
    
    # Recommendations if identifiers are found
    recommendations = []
    if has_identifiers:
        recommendations.append("Consider redacting or de-identifying the detected PHI")
        recommendations.append("Ensure you have proper authorization to share this information")
        recommendations.append("Review data minimization principles to share only necessary information")
    
    return {
        "compliant": not has_identifiers,
        "identifiers_found": identifiers,
        "recommendations": recommendations
    }

def generate_hipaa_compliant_prompt(conversation_text: str) -> str:
    """
    Generate a HIPAA-compliant prompt for LLM summarization.
    
    Args:
        conversation_text: The conversation text
        
    Returns:
        HIPAA-compliant prompt for LLM
    """
    prompt = f"""
You are a HIPAA-trained healthcare professional creating a concise medical summary. Follow these requirements for HIPAA compliance and clinical accuracy:

REQUIREMENTS:
1. Follow all HIPAA guidelines to protect patient privacy
2. Use only clinical terminology that a nurse or healthcare provider would use
3. Write in a professional, clear, third-person narrative style
4. Begin the summary directly with patient information - no introductory phrases
5. Focus on objective clinical observations rather than subjective impressions
6. Avoid using any directly identifying patient information
7. Document only clinically relevant details from the conversation
8. Include health status, symptoms, treatments, and follow-up plans
9. Document medication adherence and any reported issues
10. Maintain a neutral, clinical tone throughout

CONVERSATION TEXT:
```
{conversation_text}
```

SUMMARY FORMAT:
- Begin with patient identification (using "patient" rather than name) and appointment overview
- Document vital signs with their values
- List reported symptoms and their duration/severity
- Note current medications and adherence
- Include any treatment plans or recommendations
- End with follow-up instructions
- Write 4-5 paragraphs in clinical documentation style
"""
    return prompt

def preprocess_transcript_for_hipaa(transcript_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess transcript data to help ensure HIPAA compliance.
    
    Args:
        transcript_data: The transcript data dictionary
        
    Returns:
        Preprocessed transcript data with enhanced HIPAA compliance
    """
    # Create a copy to avoid modifying the original
    processed_data = transcript_data.copy()
    
    # Process segments if present
    if "segments" in processed_data:
        for segment in processed_data["segments"]:
            if "transcription" in segment:
                # Generate a random ID for each speaker instead of names
                if "speaker" in segment and segment["speaker"]:
                    # If speaker is a string that looks like a name, anonymize it
                    speaker = segment["speaker"]
                    if re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$', speaker):
                        segment["original_speaker"] = speaker
                        segment["speaker"] = f"Speaker {speaker[0]}"  # Use first initial
    
    # Add HIPAA compliance metadata
    processed_data["metadata"] = processed_data.get("metadata", {})
    processed_data["metadata"]["hipaa_processed"] = True
    processed_data["metadata"]["processing_date"] = "REDACTED"  # Don't include exact date for HIPAA compliance
    
    return processed_data
