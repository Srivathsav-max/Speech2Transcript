"""
Nurse-like Clinical Summary Generation Module

This module provides enhanced functions for generating more comprehensive,
detailed medical summaries in the style of professional nursing documentation.
"""

import re
from typing import Dict, Any, List, Tuple, Optional
import logging
import json

logger = logging.getLogger(__name__)

def generate_nurse_summary_prompt(conversation_text: str, basic_info: Dict[str, Any]) -> str:
    """
    Generate a prompt for LLM that will produce a high-quality nurse-like clinical summary.
    
    Args:
        conversation_text: The full conversation text
        basic_info: Dictionary with basic extracted information
        
    Returns:
        Formatted prompt for generating nurse-like summary
    """
    # Extract relevant information for the prompt
    patient_name = basic_info.get('patient_name', 'Patient')
    provider_name = basic_info.get('provider_name', 'Provider')
    conditions = ', '.join(basic_info.get('conditions', [])) if basic_info.get('conditions') else 'Unknown'
    medications = ', '.join(basic_info.get('medications', [])) if basic_info.get('medications') else 'Unknown'
    
    prompt = f"""
You are a professional registered nurse creating a detailed clinical summary of a telehealth appointment. Write the summary as if you were documenting the encounter in a patient's medical record following HIPAA guidelines.

PATIENT INFO:
- Name: {patient_name}
- Provider: {provider_name}
- Conditions: {conditions}
- Medications: {medications}

CONVERSATION:
```
{conversation_text}
```

CLINICAL SUMMARY REQUIREMENTS:
- Begin with "Patient presented for telehealth appointment with [provider] for [reason]..."
- Use proper medical terminology and abbreviations as appropriate (BP, HR, RR, etc.)
- Document objective clinical observations and subjective patient reports clearly
- Include comprehensive details about:
  * Chief complaint and history of present illness
  * Current medications, dosages, and adherence patterns
  * Vital signs with specific values if mentioned
  * Review of systems findings (positive and pertinent negative)
  * Physical assessment findings discussed during telehealth
  * Patient education provided
  * Treatment plan and interventions
- Note any concerning symptoms requiring follow-up
- Document lifestyle factors relevant to health (diet, exercise, sleep, stress)
- End with clear follow-up instructions and timeline
- Write in concise, professional nursing documentation style
- Use proper medical abbreviations as appropriate
- Separate assessment findings by body system where relevant
- Maintain HIPAA compliance throughout

Write a comprehensive, authoritative clinical summary as a nurse would document in a medical record. Use standard nursing documentation format with clear, professional medical terminology.
"""
    return prompt

def format_nurse_summary(summary_text: str) -> str:
    """
    Format and enhance the generated summary to make it more nurse-like.
    
    Args:
        summary_text: The generated summary text
        
    Returns:
        Formatted nurse-like summary
    """
    # Add section headers if they don't exist
    sections = [
        ("SUBJECTIVE:", ["subjective", "chief complaint", "history of present illness", "reported"]), 
        ("OBJECTIVE:", ["objective", "vital signs", "assessment", "physical exam"]),
        ("ASSESSMENT:", ["assessment", "diagnosis", "impression", "problem list"]),
        ("PLAN:", ["plan", "treatment", "recommendations", "follow-up", "medications"])
    ]
    
    # Check if summary already has section headers
    has_sections = any(section[0].lower() in summary_text.lower() for section in sections)
    
    if not has_sections:
        # Try to identify natural paragraph breaks to insert sections
        paragraphs = summary_text.split('\n\n')
        if len(paragraphs) >= 3:
            formatted_summary = ""
            
            # First paragraph is usually subjective
            formatted_summary += "SUBJECTIVE:\n" + paragraphs[0] + "\n\n"
            
            # Middle paragraphs are usually objective
            middle_paragraphs = paragraphs[1:-1]
            formatted_summary += "OBJECTIVE:\n" + "\n\n".join(middle_paragraphs) + "\n\n"
            
            # Try to split the last paragraph into assessment and plan
            last_para = paragraphs[-1]
            plan_match = re.search(r'(follow-up|advised|recommended|scheduled|plan is|will|should)', last_para, re.IGNORECASE)
            
            if plan_match:
                split_point = plan_match.start()
                assessment = last_para[:split_point].strip()
                plan = last_para[split_point:].strip()
                
                formatted_summary += "ASSESSMENT:\n" + assessment + "\n\n"
                formatted_summary += "PLAN:\n" + plan
            else:
                # Just add as plan if we can't clearly split
                formatted_summary += "PLAN:\n" + last_para
                
            return formatted_summary
    
    # If we already have sections or couldn't identify paragraphs properly, return as is
    return summary_text

def enhance_with_medical_terminology(text: str) -> str:
    """
    Enhance text with proper medical terminology and formatting.
    
    Args:
        text: Original text
        
    Returns:
        Text enhanced with medical terminology
    """
    # Common replacements to make the text more medical
    replacements = [
        (r'\bblood pressure\b', 'BP'),
        (r'\bheart rate\b', 'HR'),
        (r'\brespiratory rate\b', 'RR'),
        (r'\btemperature\b', 'temp'),
        (r'\boxygen saturation\b', 'O2 sat'),
        (r'\btwice a day\b', 'BID'),
        (r'\bthree times a day\b', 'TID'),
        (r'\bfour times a day\b', 'QID'),
        (r'\bevery day\b', 'QD'),
        (r'\bas needed\b', 'PRN'),
        (r'\bwithout\b', 's/'),
        (r'\bhistory of\b', 'h/o'),
        (r'\bshortness of breath\b', 'SOB'),
        (r'\bnausea and vomiting\b', 'N/V'),
        (r'\bnot applicable\b', 'N/A'),
    ]
    
    enhanced_text = text
    for pattern, replacement in replacements:
        enhanced_text = re.sub(pattern, replacement, enhanced_text, flags=re.IGNORECASE)
    
    return enhanced_text

def structure_clinical_summary(summary_text: str, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Structure a clinical summary into a standardized format with sections.
    
    Args:
        summary_text: The generated summary text
        extracted_info: Dictionary with extracted medical information
        
    Returns:
        Dictionary with structured clinical summary
    """
    # Create SOAP format structure
    structured_summary = {
        "subjective": "",
        "objective": "",
        "assessment": "",
        "plan": "",
        "raw_summary": summary_text,
        "extracted_info": extracted_info
    }
    
    # Try to extract sections from text
    subjective_match = re.search(r'(?:SUBJECTIVE:|SUBJECTIVE FINDINGS:|HISTORY:)(.*?)(?:OBJECTIVE:|OBJECTIVE FINDINGS:|PHYSICAL EXAM:|$)', 
                                 summary_text, re.IGNORECASE | re.DOTALL)
    if subjective_match:
        structured_summary["subjective"] = subjective_match.group(1).strip()
    
    objective_match = re.search(r'(?:OBJECTIVE:|OBJECTIVE FINDINGS:|PHYSICAL EXAM:)(.*?)(?:ASSESSMENT:|IMPRESSION:|DIAGNOSIS:|$)', 
                                summary_text, re.IGNORECASE | re.DOTALL)
    if objective_match:
        structured_summary["objective"] = objective_match.group(1).strip()
    
    assessment_match = re.search(r'(?:ASSESSMENT:|IMPRESSION:|DIAGNOSIS:)(.*?)(?:PLAN:|TREATMENT:|RECOMMENDATIONS:|$)', 
                                 summary_text, re.IGNORECASE | re.DOTALL)
    if assessment_match:
        structured_summary["assessment"] = assessment_match.group(1).strip()
    
    plan_match = re.search(r'(?:PLAN:|TREATMENT:|RECOMMENDATIONS:)(.*?)$', 
                           summary_text, re.IGNORECASE | re.DOTALL)
    if plan_match:
        structured_summary["plan"] = plan_match.group(1).strip()
    
    # If we couldn't extract sections, use heuristics to split the text
    if not any([structured_summary["subjective"], structured_summary["objective"], 
                structured_summary["assessment"], structured_summary["plan"]]):
        paragraphs = summary_text.split('\n\n')
        
        if len(paragraphs) >= 3:
            structured_summary["subjective"] = paragraphs[0]
            
            if len(paragraphs) >= 4:
                structured_summary["objective"] = paragraphs[1]
                structured_summary["assessment"] = paragraphs[2]
                structured_summary["plan"] = '\n\n'.join(paragraphs[3:])
            else:
                structured_summary["objective"] = paragraphs[1]
                structured_summary["plan"] = paragraphs[2]
    
    return structured_summary
