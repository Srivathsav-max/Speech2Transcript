import re
from typing import Dict, Any, List, Tuple, Optional, Set

def extract_speakers_enhanced(conversation: List[Tuple[str, str]]) -> Dict[str, str]:
    """
    Enhanced speaker identification with improved heuristics.
    
    Args:
        conversation: List of (speaker, text) tuples
        
    Returns:
        Dictionary mapping speaker IDs to roles with confidence scores
    """
    # Track multiple speaker indicators
    indicators = {
        "question_count": {},  # Who asks more questions?
        "medical_terms": {},   # Who uses more medical terminology?
        "word_count": {},      # Total word count per speaker
        "greeting_count": {},  # Who initiates with greetings?
        "introduction_phrases": {},  # Who uses professional introduction phrases?
        "medical_inquiries": {},     # Who asks about symptoms, medications, etc.
    }
    
    # Medical terminology indicators for healthcare providers
    medical_term_patterns = [
        r'\b(?:diagnosis|prognosis|symptoms|vital|blood\s+pressure|glucose|cholesterol|hba1c)\b',
        r'\b(?:medication|prescription|dosage|treatment|therapy|regimen)\b',
        r'\b(?:referral|specialist|follow-up|appointment)\b',
        r'\b(?:laboratory|results|levels|readings|tests|screening)\b'
    ]
    
    # Professional introduction patterns
    intro_patterns = [
        r'\b(?:my name is|this is|speaking|calling from)\s+(?:dr\.|doctor|nurse|provider)',
        r'\b(?:i\'m|i am)\s+(?:from|with|at|calling from)\s+(?:the\s+)?\b(?:clinic|office|hospital|healthcare|practice|center)\b'
    ]
    
    # Medical inquiry patterns
    inquiry_patterns = [
        r'\bhow\s+(?:are|have)\s+you\s+(?:feeling|been|doing)(?:\s+today)?\b',
        r'\b(?:any|have you had any)\s+(?:pain|discomfort|symptoms|issues|problems|side effects|reactions)\b',
        r'\b(?:are you|have you been)\s+taking\s+your\s+(?:medication|medicine|pills|insulin|prescriptions)\b',
        r'\bhow\'s\s+your\s+(?:blood\s+sugar|glucose|pressure|diet|exercise|health)\b',
        r'\b(?:when was|have you had)\s+your\s+(?:last|latest|recent|previous)\s+(?:appointment|checkup|test|reading)\b'
    ]
    
    # Examine early conversation for introductions (first 20% or at least 5 exchanges)
    early_segment = conversation[:max(5, int(len(conversation) * 0.2))]
    for speaker, text in early_segment:
        # Count greetings
        if re.search(r'\b(?:hello|hi|good\s+(?:morning|afternoon|evening)|greetings)\b', text, re.IGNORECASE):
            indicators["greeting_count"][speaker] = indicators["greeting_count"].get(speaker, 0) + 1
        
        # Count professional introductions
        for pattern in intro_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                indicators["introduction_phrases"][speaker] = indicators["introduction_phrases"].get(speaker, 0) + 1
    
    # Process full conversation for other indicators
    for speaker, text in conversation:
        # Count questions
        if "?" in text:
            indicators["question_count"][speaker] = indicators["question_count"].get(speaker, 0) + 1
        
        # Count total words
        word_count = len(text.split())
        indicators["word_count"][speaker] = indicators["word_count"].get(speaker, 0) + word_count
        
        # Count medical terminology usage
        for pattern in medical_term_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                indicators["medical_terms"][speaker] = indicators["medical_terms"].get(speaker, 0) + 1
                break  # Only count once per utterance
        
        # Count medical inquiries
        for pattern in inquiry_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                indicators["medical_inquiries"][speaker] = indicators["medical_inquiries"].get(speaker, 0) + 1
                break  # Only count once per utterance
    
    # Compute total scores
    speaker_scores = {}
    speakers = set(s for s, _ in conversation)
    
    for speaker in speakers:
        score = 0
        
        # Question ratio (normalized by word count to account for verbose speakers)
        question_count = indicators["question_count"].get(speaker, 0)
        word_count = indicators["word_count"].get(speaker, 1)  # Avoid division by zero
        question_ratio = question_count / word_count
        score += question_ratio * 10  # Scale it up a bit
        
        # Medical terminology usage
        score += indicators["medical_terms"].get(speaker, 0) * 0.5
        
        # Professional introduction phrases
        score += indicators["introduction_phrases"].get(speaker, 0) * 2
        
        # Greeting initiative
        score += indicators["greeting_count"].get(speaker, 0)
        
        # Medical inquiries
        score += indicators["medical_inquiries"].get(speaker, 0)
        
        speaker_scores[speaker] = score
    
    # Determine care manager (highest score) and patient (other speakers)
    if not speaker_scores:
        return {"care_manager": None, "patient": None, "confidence": "low"}
    
    care_manager = max(speaker_scores.items(), key=lambda x: x[1])[0]
    
    # Calculate confidence based on score gap between highest and second highest
    scores = sorted(speaker_scores.values(), reverse=True)
    score_gap = scores[0] - scores[1] if len(scores) > 1 else scores[0]
    
    confidence = "low"
    if score_gap > 5:
        confidence = "high"
    elif score_gap > 2:
        confidence = "medium"
    
    # Determine patient (speakers with lowest scores)
    other_speakers = [s for s in speakers if s != care_manager]
    patient = other_speakers[0] if other_speakers else None
    
    return {
        "care_manager": care_manager,
        "patient": patient,
        "confidence": confidence
    }

def extract_patient_name(conversation: List[Tuple[str, str]], care_manager: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract patient name with improved pattern matching and confidence scoring.
    
    Args:
        conversation: List of (speaker, text) tuples
        care_manager: Speaker ID of care manager (if known)
        
    Returns:
        Dictionary with extracted patient name and confidence score
    """
    # Focus on early part of conversation where introductions happen
    early_conversation = conversation[:max(10, int(len(conversation) * 0.25))]
    
    name_patterns = [
        # Greeting patterns
        r'(?:hello|hi|hey|good\s+(?:morning|afternoon|evening))\s+(?:there\s+)?(?:mr\.|mrs\.|ms\.|miss|dr\.?)?\s*([A-Z][a-zA-Z\-\']+)(?:\s+([A-Z][a-zA-Z\-\']+))?',
        
        # Introduction patterns
        r'(?:this is|speaking with|talking to|calling for|calling)\s+(?:mr\.|mrs\.|ms\.|miss|dr\.?)?\s*([A-Z][a-zA-Z\-\']+)(?:\s+([A-Z][a-zA-Z\-\']+))?',
        
        # Confirmation patterns
        r'(?:am I speaking with|are you|is this|would this be)\s+(?:mr\.|mrs\.|ms\.|miss|dr\.?)?\s*([A-Z][a-zA-Z\-\']+)(?:\s+([A-Z][a-zA-Z\-\']+))?',
        
        # Self-introduction
        r'(?:my name is|i\'m|i am|this is)\s+(?:mr\.|mrs\.|ms\.|miss|dr\.?)?\s*([A-Z][a-zA-Z\-\']+)(?:\s+([A-Z][a-zA-Z\-\']+))?',
        
        # Direct address
        r'(?:how are you|how are you doing|how are you feeling)\s+(?:today|doing)?,?\s+(?:mr\.|mrs\.|ms\.|miss|dr\.?)?\s*([A-Z][a-zA-Z\-\']+)',
        
        # Closing patterns
        r'(?:thank you|thanks|goodbye|take care),?\s+(?:mr\.|mrs\.|ms\.|miss|dr\.?)?\s*([A-Z][a-zA-Z\-\']+)'
    ]
    
    # Track potential patient names with confidence
    name_candidates = {}
    
    # First pass: Extract from care manager's speech if known
    if care_manager:
        for speaker, text in early_conversation:
            if speaker == care_manager:
                for pattern in name_patterns:
                    matches = re.finditer(pattern, text)
                    for match in matches:
                        # Extract name components
                        first_name = match.group(1)
                        last_name = match.group(2) if len(match.groups()) >= 2 and match.group(2) else None
                        
                        # Validate it looks like a name
                        if first_name and len(first_name) > 1 and first_name[0].isupper():
                            full_name = f"{first_name} {last_name}" if last_name else first_name
                            name_candidates[full_name] = name_candidates.get(full_name, 0) + 1
    
    # Second pass: Try all patterns on all speakers if no candidates or care manager unknown
    if not name_candidates:
        for speaker, text in early_conversation:
            for pattern in name_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    # Extract name components
                    first_name = match.group(1)
                    last_name = match.group(2) if len(match.groups()) >= 2 and match.group(2) else None
                    
                    # Validate
                    if first_name and len(first_name) > 1 and first_name[0].isupper():
                        full_name = f"{first_name} {last_name}" if last_name else first_name
                        
                        # Lower confidence if not from care manager
                        if care_manager and speaker != care_manager:
                            name_candidates[full_name] = name_candidates.get(full_name, 0) + 0.5
                        else:
                            name_candidates[full_name] = name_candidates.get(full_name, 0) + 1
    
    # Check for repetition in whole conversation to increase confidence
    if name_candidates:
        for candidate in list(name_candidates.keys()):
            for _, text in conversation:
                if re.search(r'\b' + re.escape(candidate) + r'\b', text):
                    name_candidates[candidate] += 0.5
    
    # Filter out common words that might be mistaken for names
    common_words = {'patient', 'there', 'hello', 'morning', 'calling', 'speaking'}
    name_candidates = {name: score for name, score in name_candidates.items() 
                      if name.lower() not in common_words}
    
    # Return best candidate with confidence score
    if name_candidates:
        best_name = max(name_candidates.items(), key=lambda x: x[1])
        
        # Determine confidence level
        confidence = "low"
        if best_name[1] >= 3:
            confidence = "high"
        elif best_name[1] >= 1.5:
            confidence = "medium"
        
        return {"patient_name": best_name[0], "confidence": confidence}
    else:
        return {"patient_name": "Patient", "confidence": "low"}

def extract_provider_name(conversation: List[Tuple[str, str]], care_manager: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract provider/doctor name with improved pattern matching.
    
    Args:
        conversation: List of (speaker, text) tuples
        care_manager: Speaker ID of care manager (if known)
        
    Returns:
        Dictionary with extracted provider name and confidence score
    """
    # Patterns for provider name extraction
    provider_patterns = [
        r'(?:doctor|dr\.|physician|provider)\s+([A-Z][a-zA-Z\-\']+)(?:\s+([A-Z][a-zA-Z\-\']+))?',
        r'(?:from|with|see|seeing|referred by|appointment with)\s+(?:doctor|dr\.)\s+([A-Z][a-zA-Z\-\']+)(?:\s+([A-Z][a-zA-Z\-\']+))?',
        r'(?:i am|i\'m|this is|my name is)\s+(?:doctor|dr\.)\s+([A-Z][a-zA-Z\-\']+)(?:\s+([A-Z][a-zA-Z\-\']+))?',
        r'(?:s?he|you)(?:\'ll|\'re|\'s| will| are| are going to)?\s+(?:see|talk to|speak with|meet with|have an appointment with)\s+(?:doctor|dr\.)\s+([A-Z][a-zA-Z\-\']+)'
    ]
    
    # Provider self-introduction patterns (when care manager is the provider)
    self_intro_patterns = [
        r'(?:i am|i\'m|this is)\s+(?:doctor|dr\.)\s+([A-Z][a-zA-Z\-\']+)',
        r'(?:my name is|i\'m called)\s+(?:doctor|dr\.)\s+([A-Z][a-zA-Z\-\']+)'
    ]
    
    # Track potential provider names with confidence
    provider_candidates = {}
    
    # First check: Care manager self-introduction as provider
    if care_manager:
        for speaker, text in conversation[:10]:  # Check early conversation
            if speaker == care_manager:
                for pattern in self_intro_patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        name = match.group(1)
                        if name and len(name) > 1 and name[0].isupper():
                            provider_candidates[name] = provider_candidates.get(name, 0) + 2  # Higher confidence
    
    # Check entire conversation for provider references
    for _, text in conversation:
        for pattern in provider_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(1)
                if name and len(name) > 1 and name[0].isupper():
                    provider_candidates[name] = provider_candidates.get(name, 0) + 1
    
    # Determine confidence and format provider name
    if provider_candidates:
        best_provider = max(provider_candidates.items(), key=lambda x: x[1])
        
        # Format name with "Dr." prefix if not already present
        name = best_provider[0]
        if not name.lower().startswith("dr"):
            formatted_name = f"Dr. {name}"
        else:
            formatted_name = name
        
        # Determine confidence level
        confidence = "low"
        if best_provider[1] >= 3:
            confidence = "high"
        elif best_provider[1] >= 1.5:
            confidence = "medium"
        
        return {"provider_name": formatted_name, "confidence": confidence}
    else:
        return {"provider_name": "Provider", "confidence": "low"}

def extract_medical_conditions(conversation: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    Extract medical conditions with improved context validation.
    
    Args:
        conversation: List of (speaker, text) tuples
        
    Returns:
        Dictionary with extracted conditions and confidence scores
    """
    # Create full conversation text
    full_text = " ".join(text for _, text in conversation).lower()
    
    # Enhanced condition patterns with context awareness
    condition_patterns = {
        "diabetes": {
            "patterns": [
                r'\b(?:type\s+(?:1|2|one|two|ii|i)\s+)?diabetes\b',
                r'\bdiabetic\b',
                r'\b(?:blood\s+sugar|glucose)\s+(?:level|problem|issue|control)\b',
                r'\ba1c(?:\s+level)?\s+(?:of|is|was)?\s+(?:\d+\.?\d*)\b'
            ],
            "context": [
                r'\b(?:insulin|metformin|glucose|meter|reading|strip|diabetic)\b',
                r'\b(?:a1c|glycemic|sugar|hyperglycemia|hypoglycemia)\b'
            ]
        },
        "hypertension": {
            "patterns": [
                r'\bhypertension\b',
                r'\bhigh\s+blood\s+pressure\b',
                r'\bblood\s+pressure\s+(?:medication|problem|issue|elevated|high)\b'
            ],
            "context": [
                r'\b(?:systolic|diastolic|mmhg|blood\s+pressure\s+(?:medication|reading))\b',
                r'\b(?:lisinopril|amlodipine|losartan|hydrochlorothiazide|hctz)\b'
            ]
        },
        "heart disease": {
            "patterns": [
                r'\bheart\s+(?:disease|failure|problem|condition|attack)\b',
                r'\bcardiac\s+(?:issue|problem|condition|event)\b',
                r'\bcardiomyopathy\b',
                r'\bcoronary\s+artery\s+disease\b',
                r'\batrial\s+fibrillation\b',
                r'\barrhythmia\b'
            ],
            "context": [
                r'\b(?:cardiologist|ekg|ecg|echo|echocardiogram|stent|bypass)\b',
                r'\b(?:chest\s+pain|palpitation|murmur|warfarin|aspirin|statin)\b'
            ]
        },
        "kidney disease": {
            "patterns": [
                r'\bkidney\s+(?:disease|problem|failure|condition)\b',
                r'\brenal\s+(?:disease|problem|failure|condition)\b',
                r'\bdialysis\b',
                r'\bcreatinine\b'
            ],
            "context": [
                r'\b(?:nephrologist|dialysis|creatinine|egfr|bun|protein\s+in\s+urine)\b',
                r'\b(?:kidney\s+specialist|renal\s+function)\b'
            ]
        },
        "COPD": {
            "patterns": [
                r'\bcopd\b',
                r'\bchronic\s+obstructive\s+pulmonary\s+disease\b',
                r'\bemphysema\b',
                r'\bchronic\s+bronchitis\b'
            ],
            "context": [
                r'\b(?:inhaler|breathing\s+treatment|oxygen|pulmonologist|albuterol)\b',
                r'\b(?:shortness\s+of\s+breath|sob|spirometry|lung\s+(?:function|capacity))\b'
            ]
        },
        "asthma": {
            "patterns": [
                r'\basthma\b',
                r'\bwheezing\b',
                r'\binhaler\b'
            ],
            "context": [
                r'\b(?:inhaler|albuterol|fluticasone|advair|symbicort|ventolin)\b',
                r'\b(?:shortness\s+of\s+breath|wheezing|pulmonologist|peak\s+flow)\b'
            ]
        },
        "depression": {
            "patterns": [
                r'\bdepression\b',
                r'\bdepressive\s+disorder\b',
                r'\bmajor\s+depressive\s+disorder\b',
                r'\bclinical\s+depression\b'
            ],
            "context": [
                r'\b(?:antidepressant|psychiatrist|therapist|counseling|therapy)\b',
                r'\b(?:ssri|prozac|zoloft|lexapro|celexa|wellbutrin|mental\s+health)\b'
            ]
        },
        "anxiety": {
            "patterns": [
                r'\banxiety\b',
                r'\banxiety\s+disorder\b',
                r'\bpanic\s+(?:attack|disorder)\b',
                r'\bgeneralized\s+anxiety\s+disorder\b'
            ],
            "context": [
                r'\b(?:anxious|worry|stress|nervous|psychiatrist|therapist)\b',
                r'\b(?:xanax|klonopin|ativan|valium|buspar|ssri)\b'
            ]
        },
        "obesity": {
            "patterns": [
                r'\bobesity\b',
                r'\boverweight\b',
                r'\bweight\s+(?:issue|problem|management)\b',
                r'\bbmi\s+(?:of|is|was)?\s+(?:over|above)?\s*(?:30|3\d)\b'
            ],
            "context": [
                r'\b(?:diet|exercise|weight\s+loss|nutrition|bariatric)\b',
                r'\b(?:bmi|body\s+mass\s+index|weight\s+management|ozempic|wegovy)\b'
            ]
        }
    }
    
    # Results with confidence
    results = {
        "conditions": [],
        "confidence": {}
    }
    
    # Extract conditions with context validation
    for condition, pattern_data in condition_patterns.items():
        # Check for condition patterns
        pattern_matches = [
            re.search(pattern, full_text) is not None
            for pattern in pattern_data["patterns"]
        ]
        
        # Count how many pattern matches we have
        match_count = sum(1 for m in pattern_matches if m)
        
        if match_count > 0:
            # Check context for validation
            context_matches = [
                re.search(context, full_text) is not None
                for context in pattern_data.get("context", [])
            ]
            
            # Count context matches
            context_count = sum(1 for m in context_matches if m)
            
            # Determine confidence based on pattern and context matches
            confidence = "low"
            if match_count >= 2 or (match_count >= 1 and context_count >= 1):
                # Strong confidence if multiple patterns match or pattern with context
                confidence = "high" if context_count >= 2 else "medium"
                results["conditions"].append(condition)
                results["confidence"][condition] = confidence
            elif match_count == 1:
                # Low confidence with single pattern match and no context
                results["conditions"].append(condition)
                results["confidence"][condition] = confidence
    
    return results

def extract_medications(conversation: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    Extract medications with improved context validation.
    
    Args:
        conversation: List of (speaker, text) tuples
        
    Returns:
        Dictionary with extracted medications and confidence scores
    """
    # Create full conversation text
    full_text = " ".join(text for _, text in conversation).lower()
    
    # Common medications with patterns and brand names
    common_meds = {
        "metformin": {
            "patterns": [r'\bmetformin\b', r'\bglucophage\b'],
            "context": [r'\bdiabetes\b', r'\bglucose\b', r'\bdiabetic\b', r'\bblood\s+sugar\b']
        },
        "insulin": {
            "patterns": [
                r'\binsulin\b', r'\blantus\b', r'\bnovolog\b', r'\bhumalog\b', 
                r'\btoujeo\b', r'\btresiba\b', r'\bbasaglar\b'
            ],
            "context": [r'\bdiabetes\b', r'\bdiabetic\b', r'\bblood\s+sugar\b', r'\binjection\b']
        },
        "lisinopril": {
            "patterns": [r'\blisinopril\b', r'\bzestril\b', r'\bprinivil\b'],
            "context": [r'\bblood\s+pressure\b', r'\bhypertension\b', r'\bace\s+inhibitor\b']
        },
        "amlodipine": {
            "patterns": [r'\bamlodipine\b', r'\bnorvasc\b'],
            "context": [r'\bblood\s+pressure\b', r'\bhypertension\b', r'\bcalcium\s+channel\s+blocker\b']
        },
        "atorvastatin": {
            "patterns": [r'\batorvastatin\b', r'\blipitor\b', r'\bstatin\b'],
            "context": [r'\bcholesterol\b', r'\blipid\b', r'\bldl\b', r'\bheart\b']
        },
        "levothyroxine": {
            "patterns": [r'\blevothyroxine\b', r'\bsynthroid\b', r'\bthyroid\s+medication\b'],
            "context": [r'\bthyroid\b', r'\bhypothyroidism\b', r'\btsh\b']
        },
        "albuterol": {
            "patterns": [r'\balbuterol\b', r'\bventolin\b', r'\bproair\b', r'\brescue\s+inhaler\b'],
            "context": [r'\basthma\b', r'\binhaler\b', r'\bbreathing\b', r'\bcopd\b']
        },
        "omeprazole": {
            "patterns": [r'\bomeprazole\b', r'\bprilosec\b', r'\bacid\s+reducer\b', r'\bppi\b'],
            "context": [r'\bacid\s+reflux\b', r'\bgerd\b', r'\bheartburn\b', r'\bstomach\b']
        },
        "gabapentin": {
            "patterns": [r'\bgabapentin\b', r'\bneurontin\b'],
            "context": [r'\bpain\b', r'\bnerve\s+pain\b', r'\bneuropathy\b']
        },
        "hydrochlorothiazide": {
            "patterns": [r'\bhydrochlorothiazide\b', r'\bhctz\b', r'\bdiuretic\b'],
            "context": [r'\bblood\s+pressure\b', r'\bhypertension\b', r'\bwater\s+pill\b']
        },
        "metoprolol": {
            "patterns": [r'\bmetoprolol\b', r'\blopressor\b', r'\btoprol\b', r'\bbeta\s+blocker\b'],
            "context": [r'\bblood\s+pressure\b', r'\bheart\s+rate\b', r'\bheart\b']
        },
        "losartan": {
            "patterns": [r'\blosartan\b', r'\bcozaar\b', r'\barb\b'],
            "context": [r'\bblood\s+pressure\b', r'\bhypertension\b']
        },
        "fluoxetine": {
            "patterns": [r'\bfluoxetine\b', r'\bprozac\b', r'\bantidepressant\b', r'\bssri\b'],
            "context": [r'\bdepression\b', r'\banxiety\b', r'\bmood\b', r'\bmental\s+health\b']
        },
        "sertraline": {
            "patterns": [r'\bsertraline\b', r'\bzoloft\b', r'\bantidepressant\b', r'\bssri\b'],
            "context": [r'\bdepression\b', r'\banxiety\b', r'\bmood\b', r'\bmental\s+health\b']
        },
        "warfarin": {
            "patterns": [r'\bwarfarin\b', r'\bcoumadin\b', r'\bblood\s+thinner\b', r'\banticoagulant\b'],
            "context": [r'\bclot\b', r'\bafib\b', r'\batrial\s+fibrillation\b', r'\bstroke\b']
        },
        "furosemide": {
            "patterns": [r'\bfurosemide\b', r'\blasix\b', r'\bdiuretic\b', r'\bwater\s+pill\b'],
            "context": [r'\bswelling\b', r'\bedema\b', r'\bheart\s+failure\b']
        },
        "ozempic": {
            "patterns": [r'\bozempic\b', r'\bsemaglutide\b'],
            "context": [r'\bdiabetes\b', r'\bweight\s+loss\b', r'\bglp-?1\b']
        },
        "jardiance": {
            "patterns": [r'\bjardiance\b', r'\bempagliflozin\b', r'\bsglt2\b'],
            "context": [r'\bdiabetes\b', r'\bkidney\b', r'\bheart\s+failure\b']
        }
    }
    
    # Medication context indicators
    medication_contexts = [
        r'\b(?:take|taking|took|prescribed|on)\b',
        r'\b(?:medication|medicine|pill|capsule|tablet)\b',
        r'\b(?:mg|milligram|mcg|unit)\b',
        r'\b(?:daily|twice|once|every|day|morning|evening|night|week)\b',
        r'\b(?:dose|dosage|refill|prescription)\b'
    ]
    
    # Results with confidence
    results = {
        "medications": [],
        "confidence": {},
        "dosage": {}
    }
    
    # Extract medications with context validation
    for med, data in common_meds.items():
        # Check if any pattern matches
        pattern_matches = any(re.search(pattern, full_text) for pattern in data["patterns"])
        
        if pattern_matches:
            # Look for medication pattern with surrounding context
            med_context = False
            
            # Check if medication appears in medication context
            for pattern in data["patterns"]:
                for context in medication_contexts:
                    # Look for pattern before context within 5 words
                    med_before_context = re.search(
                        r'(?:' + pattern + r')\s+(?:\w+\s+){0,5}' + context, 
                        full_text
                    )
                    
                    # Look for context before pattern within 5 words
                    context_before_med = re.search(
                        context + r'\s+(?:\w+\s+){0,5}(?:' + pattern + r')',
                        full_text
                    )
                    
                    if med_before_context or context_before_med:
                        med_context = True
                        break
                        
                if med_context:
                    break
            
            # Check for condition context
            condition_context = any(re.search(context, full_text) for context in data.get("context", []))
            
            # Determine confidence level
            confidence = "low"
            if med_context and condition_context:
                confidence = "high"
            elif med_context or condition_context:
                confidence = "medium"
            
            # Extract dosage if available
            dosage = None
            # Pattern to find dosage near medication mentions
            for pattern in data["patterns"]:
                # Look for dosage after medication name (common pattern)
                dosage_match = re.search(
                    pattern + r'\s+(?:\w+\s+){0,3}(\d+(?:\.\d+)?)\s*(?:mg|mcg|milligram|microgram|unit)', 
                    full_text
                )
                
                if dosage_match:
                    dosage = dosage_match.group(1) + " " + (dosage_match.group(2) if len(dosage_match.groups()) > 1 else "mg")
                    break
            
            results["medications"].append(med)
            results["confidence"][med] = confidence
            if dosage:
                results["dosage"][med] = dosage
    
    return results

def extract_vital_signs(conversation: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    Extract vital signs with improved pattern matching and validation.
    
    Args:
        conversation: List of (speaker, text) tuples
        
    Returns:
        Dictionary with extracted vital signs and confidence scores
    """
    # Create full conversation text
    full_text = " ".join(text for _, text in conversation).lower()
    
    # Results with confidence
    results = {
        "vital_signs": {},
        "confidence": {}
    }
    
    # 1. Blood Pressure
    bp_patterns = [
        # Standard format
        r'(?:blood\s+pressure|bp)\s+(?:is|was|of|reading|at|about|around)?\s*(\d{2,3})[/\\](\d{2,3})',
        # Separate systolic/diastolic
        r'(?:systolic|sys)\s+(?:of|is|was|at)?\s*(\d{2,3})(?:\s+(?:and|with)\s+)?(?:diastolic|dia)\s+(?:of|is|was|at)?\s*(\d{2,3})',
        # BP mentioned after values
        r'(\d{2,3})[/\\](\d{2,3})\s+(?:for|as|the|your)?\s+(?:blood\s+pressure|bp)',
    ]
    
    for pattern in bp_patterns:
        bp_matches = re.finditer(pattern, full_text)
        for match in bp_matches:
            try:
                systolic = int(match.group(1))
                diastolic = int(match.group(2))
                # Validate reasonable BP ranges
                if 80 <= systolic <= 200 and 40 <= diastolic <= 120:
                    results["vital_signs"]["blood_pressure"] = f"{systolic}/{diastolic}"
                    
                    # Determine BP confidence based on context
                    bp_context = re.search(r'(?:check|measure|monitor|read|take).{0,30}(?:blood\s+pressure|bp)', full_text)
                    results["confidence"]["blood_pressure"] = "high" if bp_context else "medium"
                    break
            except (ValueError, IndexError):
                continue
    
    # 2. Glucose
    glucose_patterns = [
        r'(?:glucose|sugar|reading)\s+(?:level|is|was|of|at|about|around)?\s*(\d{2,3})(?:\s+mg/dl)?',
        r'(\d{2,3})(?:\s+mg/dl)?\s+(?:glucose|blood\s+sugar|sugar)',
        r'(?:checked|tested|measured)\s+(?:my|your|the)?\s+(?:sugar|glucose|blood\s+sugar)\s+(?:and|it\s+was|it\'s|level\s+is)?\s*(\d{2,3})',
    ]
    
    for pattern in glucose_patterns:
        glucose_matches = re.finditer(pattern, full_text)
        for match in glucose_matches:
            try:
                glucose = int(match.group(1))
                # Validate reasonable glucose range
                if 50 <= glucose <= 400:
                    results["vital_signs"]["glucose"] = glucose
                    
                    # Determine glucose confidence based on context
                    glucose_context = re.search(r'(?:diabetes|diabetic|meter|reading|check).{0,30}(?:glucose|sugar)', full_text)
                    results["confidence"]["glucose"] = "high" if glucose_context else "medium"
                    break
            except (ValueError, IndexError):
                continue
    
    # 3. A1C
    a1c_patterns = [
        r'(?:a1c|hba1c|hemoglobin\s+a1c)\s+(?:level|is|was|of|at|about|around)?\s*(\d+\.?\d*)(?:\s*%|\s+percent)?',
        r'(\d+\.?\d*)\s+(?:percent|%)\s+(?:for|as|the|your)?\s+(?:a1c|hba1c)',
    ]
    
    for pattern in a1c_patterns:
        a1c_matches = re.finditer(pattern, full_text)
        for match in a1c_matches:
            try:
                a1c = float(match.group(1))
                # Validate reasonable A1C range
                if 4.0 <= a1c <= 14.0:
                    results["vital_signs"]["a1c"] = a1c
                    
                    # Determine A1C confidence
                    a1c_context = re.search(r'(?:diabetes|diabetic|control).{0,30}(?:a1c|hemoglobin)', full_text)
                    results["confidence"]["a1c"] = "high" if a1c_context else "medium"
                    break
            except (ValueError, IndexError):
                continue
    
    # 4. Weight
    weight_patterns = [
        r'(?:weight|weigh)\s+(?:is|was|of|at|about|around)?\s*(\d{2,3})\s+(pound|lb|kg|kilo)',
        r'(\d{2,3})\s+(pound|lb|kg|kilo)s?(?:\s+(?:body\s+)?weight)?',
    ]
    
    for pattern in weight_patterns:
        weight_matches = re.finditer(pattern, full_text)
        for match in weight_matches:
            try:
                value = float(match.group(1))
                unit = match.group(2).lower()
                
                if unit in ['pound', 'lb'] and 50 <= value <= 500:
                    results["vital_signs"]["weight_lbs"] = value
                    results["confidence"]["weight_lbs"] = "medium"
                    break
                elif unit in ['kg', 'kilo'] and 20 <= value <= 200:
                    results["vital_signs"]["weight_kg"] = value
                    results["confidence"]["weight_kg"] = "medium"
                    break
            except (ValueError, IndexError):
                continue
    
    # 5. BMI
    bmi_patterns = [
        r'(?:bmi|body\s+mass\s+index)\s+(?:is|was|of|at|about|around)?\s*(\d{1,2}\.?\d*)',
        r'(?:body\s+mass\s+index|bmi)\s+(?:of|is|was)?\s*(\d{1,2}\.?\d*)',
    ]
    
    for pattern in bmi_patterns:
        bmi_matches = re.finditer(pattern, full_text)
        for match in bmi_matches:
            try:
                bmi = float(match.group(1))
                if 10 <= bmi <= 50:
                    results["vital_signs"]["bmi"] = bmi
                    results["confidence"]["bmi"] = "medium" 
                    break
            except (ValueError, IndexError):
                continue
                
    # 6. Temperature
    temp_patterns = [
        r'(?:temperature|temp)\s+(?:is|was|of|at|about|around)?\s*(\d{2})\.?(\d+)?(?:\s*°?(?:F|C))?',
        r'(\d{2})\.?(\d+)?\s*°?(?:F|C)(?:\s+(?:temperature|temp))?',
    ]
    
    for pattern in temp_patterns:
        temp_matches = re.finditer(pattern, full_text)
        for match in temp_matches:
            try:
                temp = float(match.group(1) + (f".{match.group(2)}" if match.group(2) else ""))
                # Determine temp unit and validate (Fahrenheit is common in US medical)
                if '°c' in match.group(0).lower() or 'c' in match.group(0).lower():
                    if 35.0 <= temp <= 41.0:  # Celsius range
                        results["vital_signs"]["temperature_c"] = temp
                        results["confidence"]["temperature_c"] = "medium"
                        break
                else:  # Assume Fahrenheit
                    if 95.0 <= temp <= 105.0:  # Fahrenheit range
                        results["vital_signs"]["temperature_f"] = temp
                        results["confidence"]["temperature_f"] = "medium"
                        break
            except (ValueError, IndexError):
                continue
                
    # 7. Heart Rate / Pulse
    hr_patterns = [
        r'(?:heart\s+rate|pulse|hr)\s+(?:is|was|of|at|about|around)?\s*(\d{2,3})(?:\s+(?:bpm|beats\s+per\s+minute))?',
        r'(\d{2,3})\s+(?:bpm|beats\s+per\s+minute)(?:\s+(?:heart\s+rate|pulse|hr))?',
    ]
    
    for pattern in hr_patterns:
        hr_matches = re.finditer(pattern, full_text)
        for match in hr_matches:
            try:
                hr = int(match.group(1))
                if 40 <= hr <= 200:  # Reasonable heart rate range
                    results["vital_signs"]["heart_rate"] = hr
                    results["confidence"]["heart_rate"] = "medium"
                    break
            except (ValueError, IndexError):
                continue
    
    return results

def extract_basic_info_enhanced(conversation: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    Enhanced information extraction with confidence scores.
    Combines the individual extraction functions into a comprehensive result.
    
    Args:
        conversation: List of (speaker, text) tuples
        
    Returns:
        Dictionary with extracted information and confidence scores
    """
    # First identify speakers
    speakers = extract_speakers_enhanced(conversation)
    care_manager = speakers.get("care_manager")
    
    # Extract patient name
    patient_info = extract_patient_name(conversation, care_manager)
    
    # Extract provider name
    provider_info = extract_provider_name(conversation, care_manager)
    
    # Extract medical conditions
    conditions_info = extract_medical_conditions(conversation)
    
    # Extract medications
    medications_info = extract_medications(conversation)
    
    # Extract vital signs
    vitals_info = extract_vital_signs(conversation)
    
    # Combine all results
    combined_info = {
        "patient_name": patient_info["patient_name"],
        "provider_name": provider_info["provider_name"],
        "conditions": conditions_info["conditions"],
        "medications": medications_info["medications"],
        "vital_signs": vitals_info["vital_signs"],
        "dosage": medications_info.get("dosage", {}),
        "confidence": {
            "patient_name": patient_info["confidence"],
            "provider_name": provider_info["confidence"],
            "speaker_roles": speakers["confidence"]
        }
    }
    
    # Add condition confidences
    for condition in conditions_info["conditions"]:
        if condition in conditions_info.get("confidence", {}):
            combined_info["confidence"][f"condition:{condition}"] = conditions_info["confidence"][condition]
    
    # Add medication confidences
    for med in medications_info["medications"]:
        if med in medications_info.get("confidence", {}):
            combined_info["confidence"][f"medication:{med}"] = medications_info["confidence"][med]
    
    # Add vital sign confidences
    for vital, conf in vitals_info.get("confidence", {}).items():
        combined_info["confidence"][f"vital:{vital}"] = conf
    
    return combined_info