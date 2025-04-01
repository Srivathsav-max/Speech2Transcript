"""
Telemedical Content Detection Module

This module provides utilities for detecting if a conversation contains
telemedical or healthcare-related content, helping to optimize API usage
by only sending relevant content to expensive LLM services.
"""

import re
import logging
from typing import Dict, Any, List, Tuple, Set

# Configure logger
logger = logging.getLogger(__name__)

# Define medical pattern categories for detection
MEDICAL_PATTERN_CATEGORIES = {
    "medical_roles": [
        r'\b(?:doctor|dr\.|nurse|provider|physician|clinician|therapist|counselor|pharmacist)\b',
    ],
    "medical_terms": [
        r'\b(?:patient|symptoms|diagnosis|treatment|medication|prescription|dosage)\b',
        r'\b(?:blood pressure|glucose|vitals|heart rate|temperature|pulse|examination)\b',
    ],
    "health_conditions": [
        r'\b(?:health|medical|disease|condition|chronic|illness|disorder)\b',
        r'\b(?:pain|hospital|clinic|appointment|checkup|recovery|healing)\b',
    ],
    "medical_actions": [
        r'\b(?:prescribed|diagnosed|treated|monitored|tested|examined)\b',
        r'\b(?:surgery|operation|procedure|referral|consultation|follow-up)\b',
    ],
    "healthcare_systems": [
        r'\b(?:insurance|medicare|medicaid|coverage|referral|specialist)\b',
        r'\b(?:deductible|copay|benefits|healthcare plan|primary care|urgent care)\b',
    ]
}

# The threshold for determining telemedical content (number of categories required)
TELEMEDICAL_THRESHOLD = 2


def detect_telemedical_content(text: str) -> Tuple[bool, Dict[str, List[str]]]:
    """
    Determine if text contains telemedical content by analyzing patterns.
    
    Args:
        text: The conversation text to analyze
        
    Returns:
        Tuple containing:
        - Boolean indicating if the text is telemedical
        - Dictionary of matched categories with specific matches
    """
    # Skip detection if text is too short
    if len(text) < 50:
        logger.info("Text too short for reliable telemedical detection")
        return False, {}
    
    # Dictionary to store matched categories and specific patterns
    matched_categories = {}
    
    # Check each category of patterns
    for category, patterns in MEDICAL_PATTERN_CATEGORIES.items():
        category_matches = []
        
        # Try each pattern in the category
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Store unique matches
                unique_matches = list(set(matches))
                category_matches.extend(unique_matches)
                
        # If we found any matches in this category, store them
        if category_matches:
            matched_categories[category] = category_matches
    
    # Determine if the conversation is telemedical based on how many categories matched
    is_telemedical = len(matched_categories) >= TELEMEDICAL_THRESHOLD
    
    # Log the results
    logger.info(f"Telemedical detection: {'POSITIVE' if is_telemedical else 'NEGATIVE'}")
    logger.info(f"Matched {len(matched_categories)} of {len(MEDICAL_PATTERN_CATEGORIES)} medical categories")
    
    if matched_categories:
        for category, matches in matched_categories.items():
            logger.info(f"  Category '{category}': Found {len(matches)} matches")
    
    return is_telemedical, matched_categories


def analyze_conversation(conversation_text: str) -> Dict[str, Any]:
    """
    Analyze conversation text and provide a complete report on telemedical content.
    
    Args:
        conversation_text: The conversation text to analyze
        
    Returns:
        Dictionary containing analysis results
    """
    # Run the detection
    is_telemedical, matched_categories = detect_telemedical_content(conversation_text)
    
    # Prepare the response message
    if is_telemedical:
        message = "This appears to be a telemedical conversation containing healthcare-related content."
    else:
        message = "This conversation does not contain sufficient medical terminology or healthcare context to be classified as telemedical."
    
    # Count total matches
    total_matches = sum(len(matches) for matches in matched_categories.values())
    
    # Format matches for better readability
    formatted_matches = {}
    for category, matches in matched_categories.items():
        formatted_matches[category] = list(set([m.lower() for m in matches]))
    
    # Create and return the analysis result
    return {
        "is_telemedical": is_telemedical,
        "matched_categories": list(matched_categories.keys()),
        "match_count": total_matches,
        "category_count": len(matched_categories),
        "matches": formatted_matches,
        "message": message
    }


def get_non_telemedical_message() -> str:
    """
    Return a standardized message for non-telemedical content.
    
    Returns:
        String with explanatory message
    """
    return "This audio does not appear to contain telemedical content. The conversation lacks sufficient healthcare-related terminology or medical context typically found in telemedical consultations."
