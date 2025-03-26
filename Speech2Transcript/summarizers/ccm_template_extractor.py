"""
CCM (Chronic Care Management) Template Extractor

This module provides specialized extraction of medical information for CCM templates
using advanced NLP techniques rather than simple regex matching.
"""
import re
import torch
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import spacy
from transformers import pipeline
from .base_extractor import BaseExtractor

class CCMTemplateExtractor(BaseExtractor):
    """
    Specialized extractor for CCM (Chronic Care Management) information.
    
    This class processes medical transcripts to extract information required for
    CCM billing and documentation using NLP techniques:
    - Named Entity Recognition with medical-specific models
    - Contextual understanding of medical terms
    - Negation detection
    - Temporal relationship extraction
    """
    
    def __init__(
        self,
        ner_model: str = "emilyalsentzer/Bio_ClinicalBERT", 
        device: str = None,
        confidence_threshold: float = 0.65,
        logger = None
    ):
        """
        Initialize the CCM template extractor.
        
        Args:
            ner_model: Model name for medical NER
            device: Computation device ('cuda', 'cpu', or None for auto-detect)
            confidence_threshold: Minimum confidence for entity extraction
            logger: Optional logger for messages
        """
        super().__init__(logger)
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize NLP components
        self._initialize_nlp_components(ner_model)
        
        # Initialize medical knowledge bases
        self._initialize_medical_knowledge()
        
    def _initialize_nlp_components(self, ner_model: str) -> None:
        """Initialize NLP models and components."""
        try:
            # Load spaCy model for general NLP tasks
            self._log("Loading spaCy model...")
            try:
                self.nlp = spacy.load("en_core_web_md")
                # Add negation detection
                self._log("Adding negation detection to spaCy pipeline...")
                try:
                    from negspacy.negation import Negex
                    self.nlp.add_pipe("negex", config={"ent_types": ["CONDITION", "DISEASE", "SYMPTOM"]})
                    self._log("Negation detection added successfully")
                except ImportError:
                    self._log("negspacy not available, negation detection will be limited", level="warning")
            except OSError:
                self._log("Could not load spaCy model, falling back to regex", level="warning")
                self.nlp = None
            
            # Try to load transformers model for NER if available
            self._log("Loading medical NER model...")
            try:
                self.ner = pipeline(
                    "ner", 
                    model=ner_model,
                    device=0 if self.device == "cuda" else -1,
                    aggregation_strategy="simple"
                )
                self._log("Medical NER model loaded successfully")
            except (ImportError, OSError) as e:
                self._log(f"Error loading NER model: {e}. Will proceed without specialized NER.", level="warning")
                self.ner = None
                
        except Exception as e:
            self._log(f"Error initializing NLP components: {e}", level="error")
            self.nlp = None
            self.ner = None
    
    def _initialize_medical_knowledge(self) -> None:
        """Initialize medical knowledge bases."""
        # Chronic conditions commonly monitored in CCM
        self.chronic_conditions = {
            "diabetes": ["diabetes", "diabetic", "type 1", "type 2", "blood sugar", "glucose", "a1c"],
            "hypertension": ["hypertension", "high blood pressure", "htn", "elevated bp"],
            "chronic heart failure": ["heart failure", "chf", "congestive heart failure"],
            "copd": ["copd", "chronic obstructive", "emphysema", "chronic bronchitis"],
            "asthma": ["asthma", "asthmatic"],
            "alzheimer's": ["alzheimer", "dementia"],
            "depression": ["depression", "depressive disorder"],
            "osteoarthritis": ["osteoarthritis", "arthritis"],
            "chronic kidney disease": ["kidney disease", "ckd", "renal disease"],
            "obesity": ["obesity", "obese", "bmi"]
        }
        
        # Common vital signs monitored in CCM
        self.vital_signs = {
            "blood pressure": ["blood pressure", "bp", "systolic", "diastolic", "mmhg"],
            "heart rate": ["heart rate", "pulse", "bpm"],
            "temperature": ["temperature", "temp", "fever"],
            "respiratory rate": ["respiratory rate", "breathing rate"],
            "oxygen saturation": ["oxygen", "o2", "saturation", "sats"],
            "weight": ["weight", "pounds", "lbs", "kilograms", "kg"],
            "glucose": ["glucose", "blood sugar", "mg/dl"]
        }
        
        # Common medication information in CCM
        self.medication_info = {
            "adherence terms": ["taking", "took", "skipped", "missed", "forgot", "regimen", "regularly", "schedule"],
            "side effect terms": ["side effect", "reaction", "symptom", "experiencing", "caused by", "result of"],
            "common medications": ["metformin", "lisinopril", "atorvastatin", "levothyroxine", "amlodipine", 
                                  "albuterol", "omeprazole", "losartan", "gabapentin", "hydrochlorothiazide"]
        }
        
        # Time tracking terms for CCM billing
        self.time_tracking = {
            "duration terms": ["minutes", "time spent", "duration", "took", "spent"],
            "ccm codes": ["99490", "99487", "99489"],
            "rpm codes": ["99453", "99454", "99457", "99458"]
        }

    def extract(self, text: str, entities: List = None, speaker_segments: List = None) -> Dict[str, Any]:
        """
        Extract CCM-specific information from transcript.
        
        Args:
            text: Full transcript text
            entities: Pre-extracted entities (optional)
            speaker_segments: Transcript segments with speaker information (optional)
            
        Returns:
            Dictionary with structured CCM information
        """
        self._log("Extracting CCM-specific information")
        
        # Run NER if not provided
        if not entities and self.ner is not None:
            entities = self._extract_medical_entities(text)
        
        # Process text with spaCy for linguistic features if available
        doc = None
        if self.nlp:
            doc = self.nlp(text)
            
        # Extract all required CCM components
        return {
            "chronic_conditions": self._extract_chronic_conditions(text, entities, doc),
            "care_coordination": self._extract_care_coordination(text, doc),
            "medication_management": self._extract_medication_management(text, entities, doc),
            "care_plan_updates": self._extract_care_plan_updates(text, doc),
            "patient_goals": self._extract_patient_goals(text, doc),
            "social_determinants": self._extract_social_determinants(text, doc),
            "time_spent": self._extract_time_spent(text, doc),
            "billing_codes": self._extract_billing_codes(text, doc)
        }
    
    def _extract_medical_entities(self, text: str) -> List[Dict]:
        """Extract medical entities using the medical NER model."""
        if not self.ner:
            return []
            
        try:
            # Handle long texts by chunking
            max_len = 512  # Maximum length for transformer model
            chunks = [text[i:i+max_len] for i in range(0, len(text), max_len)]
            
            # Process each chunk
            all_entities = []
            for chunk in chunks:
                chunk_entities = self.ner(chunk)
                
                # Add context to each entity
                for entity in chunk_entities:
                    entity_word = entity.get("word", "")
                    entity_pos = chunk.find(entity_word)
                    if entity_pos >= 0:
                        start_pos = max(0, entity_pos - 25)
                        end_pos = min(len(chunk), entity_pos + len(entity_word) + 25)
                        entity["context"] = chunk[start_pos:end_pos]
                
                all_entities.extend(chunk_entities)
            
            # Filter entities by confidence
            entities = [e for e in all_entities if e.get("score", 0) >= self.confidence_threshold]
            
            self._log(f"Extracted {len(entities)} medical entities")
            return entities
            
        except Exception as e:
            self._log(f"Error extracting medical entities: {e}", level="error")
            return []
    
    def _extract_chronic_conditions(self, text: str, entities: List = None, doc = None) -> List[Dict]:
        """Extract chronic conditions using NLP and medical knowledge base validation."""
        self._log("Extracting chronic conditions")
        conditions = []
        
        # Method 1: Use spaCy with negation detection if available
        if doc is not None:
            for ent in doc.ents:
                # Look for medical conditions
                if ent.label_ in ["DISEASE", "CONDITION"]:
                    # Check for negation
                    is_negated = getattr(ent, "_.negex", False)
                    
                    # Verify if it's a chronic condition using medical knowledge base
                    condition_match = self._match_chronic_condition(ent.text)
                    if condition_match and not is_negated:
                        conditions.append({
                            "name": condition_match,
                            "mention": ent.text,
                            "context": text[max(0, ent.start_char-30):min(len(text), ent.end_char+30)],
                            "confidence": 0.9
                        })
        
        # Method 2: Use pre-extracted entities from medical NER model
        if entities:
            for entity in entities:
                if entity.get("entity") in ["DISEASE", "B-PROBLEM", "I-PROBLEM", "CONDITION"]:
                    # Verify if it's a chronic condition 
                    condition_match = self._match_chronic_condition(entity.get("word", ""))
                    if condition_match:
                        # Check context for negation
                        context = entity.get("context", "")
                        if not self._is_negated(context):
                            conditions.append({
                                "name": condition_match,
                                "mention": entity.get("word", ""),
                                "context": context,
                                "confidence": entity.get("score", 0.8)
                            })
        
        # Method 3: Fallback to pattern matching with medical terminology validation
        if not conditions:
            for condition_name, terms in self.chronic_conditions.items():
                for term in terms:
                    pattern = fr'\b{re.escape(term)}\b'
                    matches = re.finditer(pattern, text.lower())
                    for match in matches:
                        # Get context and check for negation
                        start_pos = max(0, match.start() - 30)
                        end_pos = min(len(text), match.end() + 30)
                        context = text[start_pos:end_pos]
                        
                        if not self._is_negated(context):
                            # Check if already added
                            if not any(c["name"] == condition_name for c in conditions):
                                conditions.append({
                                    "name": condition_name,
                                    "mention": match.group(0),
                                    "context": context,
                                    "confidence": 0.7  # Lower confidence for regex matches
                                })
        
        # De-duplicate and sort by confidence
        unique_conditions = {}
        for condition in conditions:
            name = condition["name"]
            if name not in unique_conditions or condition["confidence"] > unique_conditions[name]["confidence"]:
                unique_conditions[name] = condition
                
        self._log(f"Extracted {len(unique_conditions)} chronic conditions")
        return list(unique_conditions.values())
    
    def _match_chronic_condition(self, text: str) -> Optional[str]:
        """Match text to known chronic conditions."""
        text_lower = text.lower()
        
        for condition_name, terms in self.chronic_conditions.items():
            # Direct match with condition name
            if condition_name.lower() in text_lower:
                return condition_name
                
            # Match with condition terms
            for term in terms:
                if term.lower() in text_lower:
                    return condition_name
                    
        return None
        
    def _is_negated(self, text: str) -> bool:
        """Check if a condition mention is negated using pattern matching."""
        negation_patterns = [
            r'no\s+(?:\w+\s+){0,2}' + r'(conditions?|diseases?|problems?|symptoms?)',
            r'(?:denies|deny|denied)\s+(?:\w+\s+){0,2}' + r'(conditions?|diseases?|problems?|symptoms?)',
            r'not\s+(?:been)?\s*(?:diagnosed|confirmed)',
            r'doesn\'t\s+have',
            r'don\'t\s+have',
            r'does\s+not\s+have',
            r'do\s+not\s+have',
            r'never\s+had',
            r'free\s+of',
            r'absence\s+of',
            r'rule\s+out',
            r'ruled\s+out',
            r'negative\s+for'
        ]
        
        text_lower = text.lower()
        
        # Check for negation patterns
        for pattern in negation_patterns:
            if re.search(pattern, text_lower):
                return True
                
        return False
    
    def _extract_medication_management(self, text: str, entities: List = None, doc = None) -> Dict[str, Any]:
        """Extract medication management information using NLP."""
        self._log("Extracting medication management information")
        
        medications = []
        adherence = None
        side_effects = None
        
        # Method 1: Use spaCy if available
        if doc is not None:
            for ent in doc.ents:
                if ent.label_ in ["MEDICATION", "TREATMENT"]:
                    # Check if it matches known medications
                    matched_med = self._is_known_medication(ent.text)
                    if matched_med:
                        med_info = {
                            "name": matched_med,
                            "mention": ent.text,
                            "dosage": self._extract_dosage(text, ent.text, ent.start_char, ent.end_char),
                            "frequency": self._extract_frequency(text, ent.start_char, ent.end_char),
                            "context": text[max(0, ent.start_char-30):min(len(text), ent.end_char+30)],
                        }
                        medications.append(med_info)
        
        # Method 2: Use pre-extracted entities
        if entities:
            for entity in entities:
                if entity.get("entity") in ["MEDICATION", "TREATMENT", "B-DRUG", "I-DRUG"]:
                    med_name = entity.get("word", "")
                    matched_med = self._is_known_medication(med_name)
                    if matched_med:
                        medications.append({
                            "name": matched_med,
                            "mention": med_name,
                            "context": entity.get("context", ""),
                            "dosage": self._extract_dosage_from_context(entity.get("context", "")),
                            "frequency": self._extract_frequency_from_context(entity.get("context", ""))
                        })
        
        # Method 3: Fallback to pattern matching for medications
        if not medications:
            # Use regex patterns to find medication mentions
            med_patterns = [
                r'\b(?:taking|takes|took|prescribed|on)\s+(\w+)\b',
                r'\b(metformin|lisinopril|atorvastatin|aspirin|ibuprofen|acetaminophen)\b'
            ]
            
            for pattern in med_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    med_name = match.group(1) if '(' in pattern else match.group(0)
                    matched_med = self._is_known_medication(med_name)
                    if matched_med:
                        start_pos = match.start()
                        end_pos = match.end()
                        context = text[max(0, start_pos-30):min(len(text), end_pos+30)]
                        
                        med_info = {
                            "name": matched_med,
                            "mention": med_name,
                            "dosage": self._extract_dosage_from_context(context),
                            "frequency": self._extract_frequency_from_context(context),
                            "context": context
                        }
                        medications.append(med_info)
        
        # Extract adherence information
        adherence = self._extract_adherence_info(text, doc)
        
        # Extract side effects information
        side_effects = self._extract_side_effects(text, doc)
        
        # De-duplicate medications by name
        unique_meds = {}
        for med in medications:
            name = med["name"]
            if name not in unique_meds:
                unique_meds[name] = med
        
        self._log(f"Extracted {len(unique_meds)} medications")
        return {
            "medications": list(unique_meds.values()),
            "adherence": adherence,
            "side_effects": side_effects
        }
    
    def _is_known_medication(self, text: str) -> Optional[str]:
        """Check if text refers to a known medication."""
        text_lower = text.lower()
        
        # Check against common medications list
        for med in self.medication_info["common medications"]:
            if med.lower() in text_lower:
                return med
                
        # Could be extended to check against an external medication database
        return text.strip()
    
    def _extract_dosage(self, text: str, med_name: str, start_pos: int, end_pos: int) -> Optional[str]:
        """Extract medication dosage."""
        # Look for dosage pattern after medication mention
        context = text[end_pos:min(len(text), end_pos + 50)]
        
        dosage_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:mg|mcg|g|ml)',
            r'(\d+(?:\.\d+)?)\s*(?:milligrams?|micrograms?|grams?|milliliters?)'
        ]
        
        for pattern in dosage_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(0)
                
        return None
    
    def _extract_frequency(self, text: str, start_pos: int, end_pos: int) -> Optional[str]:
        """Extract medication frequency."""
        # Look for frequency patterns near medication
        context = text[max(0, start_pos - 20):min(len(text), end_pos + 50)]
        
        frequency_patterns = [
            r'(once|twice|three times|four times)\s+(?:a|per)\s+day',
            r'(\d+)\s+times\s+(?:a|per)\s+day',
            r'(daily|weekly|monthly|every\s+\w+\s+(?:hours?|days?|weeks?|months?))',
            r'(in the morning|at night|with meals?|before meals?|after meals?)'
        ]
        
        for pattern in frequency_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(0)
                
        return None
    
    def _extract_dosage_from_context(self, context: str) -> Optional[str]:
        """Extract medication dosage from context."""
        dosage_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:mg|mcg|g|ml)',
            r'(\d+(?:\.\d+)?)\s*(?:milligrams?|micrograms?|grams?|milliliters?)'
        ]
        
        for pattern in dosage_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(0)
                
        return None
    
    def _extract_frequency_from_context(self, context: str) -> Optional[str]:
        """Extract medication frequency from context."""
        frequency_patterns = [
            r'(once|twice|three times|four times)\s+(?:a|per)\s+day',
            r'(\d+)\s+times\s+(?:a|per)\s+day',
            r'(daily|weekly|monthly|every\s+\w+\s+(?:hours?|days?|weeks?|months?))',
            r'(in the morning|at night|with meals?|before meals?|after meals?)'
        ]
        
        for pattern in frequency_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(0)
                
        return None
    
    def _extract_adherence_info(self, text: str, doc=None) -> str:
        """Extract medication adherence information."""
        # Look for adherence issues
        adherence_issue_patterns = [
            r'(?:not|hasn\'t been|haven\'t been|isn\'t|aren\'t)\s+taking\s+(?:medication|med|meds|medicine)',
            r'(?:missed|skipped|forgot|forgetting)\s+(?:to take|taking|doses?|medications?|meds?)',
            r'(?:adherence|compliance)\s+(?:issue|problem|concern)',
            r'(?:difficulty|problems?|issues?|challenges?)\s+(?:with|taking|remembering)\s+(?:medication|med|meds|medicine)',
            r'(?:stopped|discontinue|quit)\s+taking'
        ]
        
        adherence_good_patterns = [
            r'(?:taking|takes|took)\s+(?:all|)\s*(?:medication|med|meds|medicine)\s+(?:as prescribed|as directed)',
            r'(?:good|great|excellent)\s+(?:adherence|compliance)',
            r'(?:no|not having|doesn\'t have|don\'t have)\s+(?:trouble|difficulty|problems?|issues?)\s+(?:taking|with)\s+(?:medication|med|meds|medicine)',
            r'(?:regularly|consistently)\s+(?:taking|takes|took)\s+(?:medication|med|meds|medicine)'
        ]
        
        # Check for adherence issues
        text_lower = text.lower()
        for pattern in adherence_issue_patterns:
            match = re.search(pattern, text_lower)
            if match:
                context_start = max(0, match.start() - 30)
                context_end = min(len(text_lower), match.end() + 30)
                return f"Adherence issues: {text_lower[context_start:context_end]}"
        
        # Check for good adherence
        for pattern in adherence_good_patterns:
            if re.search(pattern, text_lower):
                return "Taking all medications as prescribed"
                
        return "No specific adherence information found"
    
    def _extract_side_effects(self, text: str, doc=None) -> str:
        """Extract medication side effect information."""
        side_effect_patterns = [
            r'(?:side|adverse)\s+effects?',
            r'(?:causing|causes|caused)\s+(?:me|him|her|them|patient)\s+to\s+(?:feel|have|experience)',
            r'(?:problem|issue|trouble)\s+with\s+(?:the|my|his|her|their)\s+(?:medication|med|meds|medicine)',
            r'(?:after|since|from)\s+taking\s+(?:medication|med|meds|medicine)\s+(?:I|he|she|they|patient)\s+(?:feel|feels|felt|have|has|had)'
        ]
        
        no_side_effects_patterns = [
            r'(?:no|not|doesn\'t|don\'t|didn\'t)\s+(?:have|having|experience|experiencing|report|feel|notice)\s+(?:any|)\s*(?:side|adverse)\s+effects',
            r'(?:tolerate|tolerating|tolerates|tolerated)\s+(?:medication|med|meds|medicine)\s+(?:well|fine|okay)'
        ]
        
        # Check for side effect mentions
        text_lower = text.lower()
        for pattern in side_effect_patterns:
            match = re.search(pattern, text_lower)
            if match:
                context_start = max(0, match.start() - 30)
                context_end = min(len(text_lower), match.end() + 30)
                return f"Side effects reported: {text_lower[context_start:context_end]}"
        
        # Check for explicit mentions of no side effects
        for pattern in no_side_effects_patterns:
            if re.search(pattern, text_lower):
                return "No side effects reported"
                
        return "No specific side effect information found"
    
    def _extract_care_coordination(self, text: str, doc=None) -> List[Dict]:
        """Extract care coordination information."""
        self._log("Extracting care coordination information")
        
        coordination_activities = []
        
        # Patterns for different coordination activities
        coordination_patterns = {
            "provider_communication": [
                r'(?:contacted|called|spoke to|talked to|communicated with|reached out to)\s+(?:Dr\.|Doctor|provider|specialist)',
                r'(?:Dr\.|Doctor|provider|specialist)\s+(?:was contacted|was informed|will be informed|will be updated)',
                r'(?:send|sent|forward|forwarded)\s+(?:information|record|report|results)\s+to\s+(?:Dr\.|Doctor|provider|specialist)'
            ],
            "referral": [
                r'(?:referr?ed|referr?al)\s+to\s+(?:specialist|Dr\.|Doctor|provider|cardiology|neurology|podiatry)',
                r'(?:appointment|visit|consultation)\s+with\s+(?:specialist|Dr\.|Doctor|provider|cardiology|neurology|podiatry)'
            ],
            "prescription_management": [
                r'(?:prescription|rx)\s+(?:refill|renewal|updated|sent|ordered)',
                r'(?:call|called|contact|contacted)\s+(?:pharmacy|pharmacist)',
                r'(?:new|changed)\s+(?:prescription|medication regimen|dosage)'
            ],
            "care_transition": [
                r'(?:discharged from|admitted to)\s+(?:hospital|facility|er|emergency)',
                r'(?:transition|follow-up)\s+(?:care|services|appointment|visit)',
                r'(?:after|post)\s+(?:hospitalization|hospital stay|discharge)'
            ]
        }
        
        # Extract coordination activities
        for activity_type, patterns in coordination_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    context_start = max(0, match.start() - 30)
                    context_end = min(len(text), match.end() + 30)
                    context = text[context_start:context_end]
                    
                    coordination_activities.append({
                        "type": activity_type,
                        "text": match.group(0),
                        "context": context
                    })
        
        self._log(f"Extracted {len(coordination_activities)} care coordination activities")
        return coordination_activities
    
    def _extract_care_plan_updates(self, text: str, doc=None) -> Dict[str, Any]:
        """Extract care plan updates."""
        self._log("Extracting care plan updates")
        
        updates = {
            "monitoring_changes": [],
            "medication_changes": [],
            "lifestyle_recommendations": [],
            "referrals": [],
            "other_updates": []
        }
        
        # Patterns for different types of care plan updates
        update_patterns = {
            "monitoring_changes": [
                r'(?:monitor|check|record|track|log)\s+(?:blood pressure|glucose|weight|symptoms)\s+(?:more frequently|daily|twice daily|weekly|every day)',
                r'(?:increase|decrease|change)\s+(?:frequency|monitoring|checking|tracking)\s+of\s+(?:blood pressure|glucose|weight|symptoms)'
            ],
            "medication_changes": [
                r'(?:increase|decrease|adjust|change)\s+(?:dose|dosage|medication|medicine|prescription)',
                r'(?:start|begin|initiate|discontinue|stop)\s+(?:taking|using)\s+(?:medication|medicine|drug|treatment)',
                r'(?:switch|change)\s+from\s+\w+\s+to\s+\w+'
            ],
            "lifestyle_recommendations": [
                r'(?:recommend|advised|suggested|encouraged)\s+(?:more|increased|regular|daily)\s+(?:exercise|activity|walking)',
                r'(?:diet|nutrition|meal)\s+(?:changes|modifications|adjustments|plan)',
                r'(?:reduce|limit|restrict|avoid)\s+(?:sodium|salt|sugar|carbs|fat|alcohol)'
            ],
            "referrals": [
                r'(?:refer|referral|send)\s+to\s+(?:specialist|Dr\.|Doctor|nutritionist|dietitian|physical therapy)',
                r'(?:schedule|set up|arrange)\s+(?:appointment|consultation|evaluation)\s+with\s+(?:specialist|Dr\.|Doctor)'
            ]
        }
        
        # Extract care plan updates
        for update_type, patterns in update_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    context_start = max(0, match.start() - 30)
                    context_end = min(len(text), match.end() + 30)
                    context = text[context_start:context_end]
                    
                    updates[update_type].append({
                        "text": match.group(0),
                        "context": context
                    })
        
        # Count total updates
        total_updates = sum(len(updates[key]) for key in updates)
        self._log(f"Extracted {total_updates} care plan updates")
        return updates
    
    def _extract_patient_goals(self, text: str, doc=None) -> List[Dict]:
        """Extract patient goals."""
        self._log("Extracting patient goals")
        
        goals = []
        
        # Patterns for patient goals
        goal_patterns = [
            r'(?:goal|aim|target|objective|plan)\s+(?:is|are|to|for)\s+(?:to\s+)?([^.,;:]+)',
            r'(?:want|wants|wanted|hoping|hope|would like)\s+to\s+([^.,;:]+)',
            r'(?:trying|going|going to|plan|planning)\s+to\s+([^.,;:]+)',
            r'(?:my|his|her|their|patient\'s)\s+(?:goal|aim|target|objective)\s+(?:is|are|to|for)\s+(?:to\s+)?([^.,;:]+)'
        ]
        
        # Extract goals
        for pattern in goal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Get the content of the goal
                goal_text = match.group(1).strip() if match.groups() else match.group(0)
                
                # Get context
                context_start = max(0, match.start() - 30)
                context_end = min(len(text), match.end() + 30)
                context = text[context_start:context_end]
                
                # Categorize goal
                category = self._categorize_goal(goal_text)
                
                goals.append({
                    "text": goal_text,
                    "category": category,
                    "context": context
                })
        
        self._log(f"Extracted {len(goals)} patient goals")
        return goals
    
    def _categorize_goal(self, goal_text: str) -> str:
        """Categorize a patient goal."""
        goal_lower = goal_text.lower()
        
        # Categories and related terms
        categories = {
            "weight_management": ["weight", "lose", "pounds", "diet", "slim", "fat"],
            "exercise": ["exercise", "walk", "run", "gym", "active", "workout", "activity"],
            "medication": ["medication", "medicine", "pills", "prescription", "dose", "regimen"],
            "nutrition": ["eat", "food", "diet", "nutrition", "meal", "vegetable", "fruit"],
            "symptom_management": ["symptom", "pain", "discomfort", "feel better", "manage", "control"],
            "lifestyle": ["smoke", "drinking", "alcohol", "sleep", "rest", "stress"]
        }
        
        # Check each category
        for category, terms in categories.items():
            for term in terms:
                if term in goal_lower:
                    return category
        
        return "other"
    
    def _extract_social_determinants(self, text: str, doc=None) -> Dict[str, Any]:
        """Extract social determinants of health."""
        self._log("Extracting social determinants of health")
        
        sdoh = {
            "housing": self._extract_sdoh_housing(text),
            "transportation": self._extract_sdoh_transportation(text),
            "food_security": self._extract_sdoh_food(text),
            "financial": self._extract_sdoh_financial(text),
            "social_support": self._extract_sdoh_social(text),
            "other": []
        }
        
        # Count total SDOH elements
        total_sdoh = sum(1 for key in sdoh if sdoh[key])
        self._log(f"Extracted {total_sdoh} social determinants of health")
        return sdoh
    
    def _extract_sdoh_housing(self, text: str) -> Optional[Dict]:
        """Extract housing-related social determinants."""
        housing_patterns = [
            r'(?:housing|home|living situation|residence|apartment|house)\s+(?:issue|problem|concern|insecurity|unstable)',
            r'(?:homeless|evict|eviction|foreclosure)',
            r'(?:trouble|difficulty|problem|issue)\s+(?:paying|with|affording)\s+(?:rent|mortgage|housing)'
        ]
        
        text_lower = text.lower()
        for pattern in housing_patterns:
            match = re.search(pattern, text_lower)
            if match:
                context_start = max(0, match.start() - 30)
                context_end = min(len(text_lower), match.end() + 30)
                return {
                    "identified": True,
                    "text": match.group(0),
                    "context": text_lower[context_start:context_end]
                }
                
        return None
    
    def _extract_sdoh_transportation(self, text: str) -> Optional[Dict]:
        """Extract transportation-related social determinants."""
        transportation_patterns = [
            r'(?:transportation|travel|getting to|getting there)\s+(?:issue|problem|barrier|concern)',
            r'(?:can\'t|cannot|couldn\'t|unable to|no way to|difficulty|trouble)\s+(?:get to|make|attend|go to)\s+(?:appointment|doctor|clinic|hospital)',
            r'(?:don\'t|doesn\'t|no)\s+(?:have|access to)\s+(?:car|transportation|bus|ride)'
        ]
        
        text_lower = text.lower()
        for pattern in transportation_patterns:
            match = re.search(pattern, text_lower)
            if match:
                context_start = max(0, match.start() - 30)
                context_end = min(len(text_lower), match.end() + 30)
                return {
                    "identified": True,
                    "text": match.group(0),
                    "context": text_lower[context_start:context_end]
                }
                
        return None
    
    def _extract_sdoh_food(self, text: str) -> Optional[Dict]:
        """Extract food security-related social determinants."""
        food_patterns = [
            r'(?:food|nutrition|meal|grocery)\s+(?:insecurity|security|issue|problem|concern)',
            r'(?:can\'t|cannot|couldn\'t|unable to|difficulty|trouble)\s+(?:afford|buy|get|access)\s+(?:food|groceries|meals)',
            r'(?:worried|worry|concerned)\s+about\s+(?:running out of|not having enough)\s+food',
            r'(?:hungry|go hungry|skip meals|skipping meals|food bank|food pantry)'
        ]
        
        text_lower = text.lower()
        for pattern in food_patterns:
            match = re.search(pattern, text_lower)
            if match:
                context_start = max(0, match.start() - 30)
                context_end = min(len(text_lower), match.end() + 30)
                return {
                    "identified": True,
                    "text": match.group(0),
                    "context": text_lower[context_start:context_end]
                }
                
        return None
    
    def _extract_sdoh_financial(self, text: str) -> Optional[Dict]:
        """Extract financial-related social determinants."""
        financial_patterns = [
            r'(?:financial|money|economic|income|bill|bills)\s+(?:issue|problem|concern|difficulty|hardship|trouble)',
            r'(?:can\'t|cannot|couldn\'t|unable to|difficulty|trouble)\s+(?:afford|pay for|cover cost of)\s+(?:medication|medicine|treatment|healthcare|prescription)',
            r'(?:lost|job loss|unemployed|unemployment|laid off|without work)',
            r'(?:insurance|coverage)\s+(?:issue|problem|concern|expired|ended|terminated)'
        ]
        
        text_lower = text.lower()
        for pattern in financial_patterns:
            match = re.search(pattern, text_lower)
            if match:
                context_start = max(0, match.start() - 30)
                context_end = min(len(text_lower), match.end() + 30)
                return {
                    "identified": True,
                    "text": match.group(0),
                    "context": text_lower[context_start:context_end]
                }
                
        return None
    
    def _extract_sdoh_social(self, text: str) -> Optional[Dict]:
        """Extract social support-related social determinants."""
        support_patterns = [
            r'(?:lives|living|stay|staying)\s+(?:alone|by (?:myself|himself|herself|themselves))',
            r'(?:no|limited|little|lack of)\s+(?:family|support|friends|social support|help|assistance)',
            r'(?:isolated|isolation|lonely|loneliness)',
            r'(?:caregiver|caregiving)\s+(?:burden|stress|strain|issue|problem|concern)'
        ]
        
        support_positive_patterns = [
            r'(?:good|great|strong|excellent)\s+(?:family|social)\s+(?:support|help|assistance)',
            r'(?:family|daughter|son|spouse|partner|friend)\s+(?:helps|helping|assist|assisting|supporting)',
            r'(?:not|isn\'t|aren\'t)\s+(?:alone|isolated|lonely)'
        ]
        
        text_lower = text.lower()
        
        # Check for support issues
        for pattern in support_patterns:
            match = re.search(pattern, text_lower)
            if match:
                context_start = max(0, match.start() - 30)
                context_end = min(len(text_lower), match.end() + 30)
                return {
                    "identified": True,
                    "has_support": False,
                    "text": match.group(0),
                    "context": text_lower[context_start:context_end]
                }
        
        # Check for positive support
        for pattern in support_positive_patterns:
            match = re.search(pattern, text_lower)
            if match:
                context_start = max(0, match.start() - 30)
                context_end = min(len(text_lower), match.end() + 30)
                return {
                    "identified": True,
                    "has_support": True,
                    "text": match.group(0),
                    "context": text_lower[context_start:context_end]
                }
                
        return None
    
    def _extract_time_spent(self, text: str, doc=None) -> Dict[str, Any]:
        """
        Extract time spent on CCM activities, critical for CCM billing requirements.
        """
        self._log("Extracting time spent information")
        
        time_info = {
            "total_minutes": None,
            "coordination_minutes": None,
            "medication_review_minutes": None,
            "has_20min_minimum": False,
            "time_mentions": []
        }
        
        # Look for time mentions
        time_patterns = [
            r'(?:spent|took|duration of|call lasted)\s+(\d+)(?:\s+|-)(?:min|minute)s?',
            r'(\d+)(?:\s+|-)(?:min|minute)s?(?:\s+|-)(?:call|session|appointment|conversation)',
            r'(?:call|session|appointment|conversation)(?:\s+|-)(?:of|lasted|took)(?:\s+|-)(\d+)(?:\s+|-)(?:min|minute)s?',
            r'(?:time spent|time of call|duration)(?:\s*:?\s*)(\d+)(?:\s+|-)(?:min|minute)s?'
        ]
        
        # Extract all time mentions
        text_lower = text.lower()
        for pattern in time_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                try:
                    minutes = int(match.group(1))
                    if 1 <= minutes <= 120:  # Sanity check
                        context_start = max(0, match.start() - 30)
                        context_end = min(len(text_lower), match.end() + 30)
                        context = text_lower[context_start:context_end]
                        
                        time_mention = {
                            "minutes": minutes,
                            "text": match.group(0),
                            "context": context
                        }
                        
                        time_info["time_mentions"].append(time_mention)
                        
                        # Categorize time based on context
                        if any(term in context for term in ["medication", "medicine", "prescription"]):
                            time_info["medication_review_minutes"] = minutes
                        elif any(term in context for term in ["coordination", "coordinating", "communicate", "provider", "contact"]):
                            time_info["coordination_minutes"] = minutes
                except ValueError:
                    continue
        
        # Determine total time from mentions
        if time_info["time_mentions"]:
            # Take the largest time as total if we have multiple mentions
            time_info["total_minutes"] = max(mention["minutes"] for mention in time_info["time_mentions"])
            time_info["has_20min_minimum"] = time_info["total_minutes"] >= 20
        
        self._log(f"Extracted {len(time_info['time_mentions'])} time mentions")
        return time_info
    
    def _extract_billing_codes(self, text: str, doc=None) -> Dict[str, Any]:
        """Extract relevant billing codes for CCM services."""
        self._log("Extracting billing code information")
        
        billing_info = {
            "suggested_codes": [],
            "time_based_code": None,
            "complexity_code": None
        }
        
        # Extract any explicit mentions of billing codes
        code_patterns = [
            r'(?:billing|bill|claim|code|CPT)\s+(?:code|with)?\s+(99490|99487|99489|99453|99454|99457|99458)',
            r'(99490|99487|99489|99453|99454|99457|99458)'
        ]
        
        text_lower = text.lower()
        for pattern in code_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                code = match.group(1)
                if code not in billing_info["suggested_codes"]:
                    billing_info["suggested_codes"].append(code)
        
        # Suggest billing code based on time if available
        if not billing_info["suggested_codes"]:
            # Check for 20+ minutes (99490)
            if any("20" in mention["text"] for mention in self._extract_time_spent(text)["time_mentions"]):
                billing_info["suggested_codes"].append("99490")
                billing_info["time_based_code"] = "99490"
            
            # Check for 60+ minutes (99487)
            if any(mention["minutes"] >= 60 for mention in self._extract_time_spent(text)["time_mentions"]):
                billing_info["suggested_codes"].append("99487")
                billing_info["time_based_code"] = "99487"
        
        self._log(f"Suggested billing codes: {', '.join(billing_info['suggested_codes']) if billing_info['suggested_codes'] else 'None'}")
        return billing_info