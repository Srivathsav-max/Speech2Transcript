"""
Medical condition and symptom extraction module.
"""
import re
from typing import Dict, List, Any
from .base_extractor import BaseExtractor

class ConditionSymptomExtractor(BaseExtractor):
    """
    Specialized extractor for medical conditions, symptoms, and health concerns
    from medical conversations.
    """
    
    def __init__(self, logger=None):
        """Initialize the condition and symptom extractor."""
        super().__init__(logger)
        self._initialize_medical_data()
    
    def _initialize_medical_data(self):
        """Initialize condition and symptom-related data."""
        # Common chronic conditions to look for explicitly
        self.common_conditions = [
            {"name": "diabetes", "keywords": ["diabetes", "diabetic", "blood sugar", "glucose", "a1c"]},
            {"name": "hypertension", "keywords": ["hypertension", "high blood pressure", "blood pressure"]},
            {"name": "hyperlipidemia", "keywords": ["cholesterol", "lipids", "triglycerides"]},
            {"name": "heart disease", "keywords": ["heart disease", "cardiovascular", "coronary", "cardiac"]},
            {"name": "asthma", "keywords": ["asthma", "wheezing", "inhaler"]},
            {"name": "COPD", "keywords": ["copd", "emphysema", "chronic obstructive"]},
            {"name": "arthritis", "keywords": ["arthritis", "joint pain", "rheumatoid"]},
            {"name": "obesity", "keywords": ["obesity", "weight", "bmi"]},
            {"name": "kidney disease", "keywords": ["kidney", "renal"]},
            {"name": "depression", "keywords": ["depression", "depressed", "mood"]},
            {"name": "anxiety", "keywords": ["anxiety", "anxious", "panic"]}
        ]
        
        # Symptom terms for better detection
        self.symptom_terms = [
            "pain", "discomfort", "ache", "headache", "nausea", "vomiting", "dizzy", "dizziness",
            "fatigue", "tired", "exhausted", "weakness", "cough", "shortness of breath", "sob",
            "fever", "chills", "sweating", "rash", "itching", "numbness", "tingling", "swelling",
            "burning", "cramping", "stiffness", "blurry vision", "palpitations", "chest pain",
            "difficulty", "problem", "trouble", "issue"
        ]
        
        # Negation terms for detecting denied symptoms
        self.negation_terms = [
            "no", "don't", "not", "hasn't", "hadn't", "haven't", 
            "doesn't", "didn't", "wouldn't", "isn't", "aren't"
        ]
        
        # Severity terms for symptom characterization
        self.severity_terms = {
            "mild": ["mild", "slight", "minor", "little", "somewhat"],
            "moderate": ["moderate", "medium", "somewhat", "noticeable"],
            "severe": ["severe", "serious", "extreme", "bad", "worst", "very", "really"]
        }
        
        # Duration terms for symptom characterization
        self.duration_terms = {
            "acute": ["today", "yesterday", "day", "recent", "just", "started"],
            "subacute": ["week", "weeks", "month", "months", "last few"],
            "chronic": ["year", "years", "chronic", "always", "long time", "ongoing"]
        }
    
    def extract(self, text: str, entities: List[Dict] = None) -> Dict[str, Any]:
        """
        Extract medical conditions and symptoms from text.
        
        Args:
            text: Text to analyze
            entities: Optional list of entities from NER
            
        Returns:
            Dictionary with conditions and symptoms
        """
        self._log("Extracting medical conditions and symptoms")
        
        # Extract medical conditions and symptoms
        conditions = self._extract_conditions(text, entities)
        symptoms = self._extract_symptoms(text, entities)
        
        # Summarize symptoms in natural language
        symptom_summary = self._summarize_symptoms(symptoms)
        
        return {
            "conditions": conditions,
            "symptoms": symptoms,
            "has_symptoms": len(symptoms) > 0,
            "symptom_text": symptom_summary
        }
    
    def _extract_conditions(self, text: str, entities: List[Dict] = None) -> List[Dict[str, Any]]:
        """
        Extract medical conditions from text.
        
        Args:
            text: Text to analyze
            entities: Optional list of entities from NER
            
        Returns:
            List of condition dictionaries
        """
        text_lower = text.lower()
        conditions = []
        
        # Method 1: Check for common conditions in text
        for condition in self.common_conditions:
            for keyword in condition["keywords"]:
                if keyword.lower() in text_lower:
                    # Find context for severity assessment
                    keyword_pos = text_lower.find(keyword.lower())
                    if keyword_pos >= 0:
                        context = self._get_context(text, keyword_pos, len(keyword))
                        
                        # Assess control/severity
                        severity = self._assess_condition_severity(context)
                            
                        # Check if condition is active
                        is_active = True
                        if re.search(r"(used to have|had|history of|resolved|cured)", context, re.IGNORECASE):
                            # Look for indications it's still present
                            if not re.search(r"(still|current|now|manage|control)", context, re.IGNORECASE):
                                is_active = False
                        
                        # Only add condition if we don't already have it
                        condition_entry = {
                            "name": condition["name"],
                            "context": context,
                            "severity": severity,
                            "is_active": is_active,
                            "confidence": 0.9  # High confidence for common conditions
                        }
                        
                        if not any(c["name"] == condition["name"] for c in conditions):
                            conditions.append(condition_entry)
                        break  # Found one keyword, no need to check others
        
        # Method 2: Use entities if provided
        if entities:
            for entity in entities:
                # Skip if not a condition entity
                if not (entity.get("type") == "CONDITION" or 
                        entity.get("entity") == "CONDITION" or
                        entity.get("custom_type") == "CONDITION"):
                    continue
                    
                cond_name = entity.get("word", "").strip()
                
                # Skip if empty, too short, or already found
                if not cond_name or len(cond_name) < 3 or any(c["name"] == cond_name for c in conditions):
                    continue
                
                # Skip common words that might be misidentified
                if cond_name.lower() in ["good", "well", "ok", "okay", "fine", "normal"]:
                    continue
                
                # Get context
                context = entity.get("context", "")
                if not context:
                    cond_pos = text_lower.find(cond_name.lower())
                    if cond_pos >= 0:
                        context = self._get_context(text, cond_pos, len(cond_name))
                
                # Add the condition
                conditions.append({
                    "name": cond_name,
                    "context": context,
                    "severity": self._assess_condition_severity(context),
                    "is_active": not re.search(r"(used to have|had|history of|resolved|cured)", context, re.IGNORECASE),
                    "confidence": entity.get("score", 0.7)
                })
        
        return conditions
    
    def _extract_symptoms(self, text: str, entities: List[Dict] = None) -> List[Dict[str, Any]]:
        """
        Extract symptoms from text.
        
        Args:
            text: Text to analyze
            entities: Optional list of entities from NER
            
        Returns:
            List of symptom dictionaries
        """
        text_lower = text.lower()
        symptoms = []
        
        # Method 1: Check for common symptom terms
        for term in self.symptom_terms:
            if term in text_lower:
                # Find the context
                term_pos = text_lower.find(term)
                if term_pos >= 0:
                    context = self._get_context(text, term_pos, len(term))
                    
                    # Check if symptom is denied
                    is_denied = self._is_symptom_denied(context, term)
                    
                    # Only add if not denied and not already in list
                    if not is_denied and not any(s["symptom"] == term for s in symptoms):
                        symptom_entry = {
                            "symptom": term,
                            "context": context,
                            "severity": self._assess_symptom_severity(context),
                            "duration": self._assess_symptom_duration(context),
                            "confidence": 0.8
                        }
                        symptoms.append(symptom_entry)
        
        # Method 2: Use entities if provided
        if entities:
            for entity in entities:
                # Skip if not a symptom entity
                if not (entity.get("type") == "SYMPTOM" or 
                        entity.get("entity") == "SYMPTOM" or
                        entity.get("custom_type") == "SYMPTOM"):
                    continue
                    
                symptom_name = entity.get("word", "").strip()
                
                # Skip if empty, too short, or already found
                if not symptom_name or len(symptom_name) < 3 or any(s["symptom"] == symptom_name for s in symptoms):
                    continue
                
                # Get context
                context = entity.get("context", "")
                if not context:
                    symptom_pos = text_lower.find(symptom_name.lower())
                    if symptom_pos >= 0:
                        context = self._get_context(text, symptom_pos, len(symptom_name))
                
                # Check if symptom is denied
                if self._is_symptom_denied(context, symptom_name):
                    continue
                
                # Add the symptom
                symptoms.append({
                    "symptom": symptom_name,
                    "context": context,
                    "severity": self._assess_symptom_severity(context),
                    "duration": self._assess_symptom_duration(context),
                    "confidence": entity.get("score", 0.7)
                })
        
        return symptoms
    
    def _assess_condition_severity(self, context: str) -> str:
        """
        Assess the severity or control level of a medical condition.
        
        Args:
            context: Text context for the condition
            
        Returns:
            Severity assessment or None
        """
        context_lower = context.lower()
        
        if re.search(r"(well|good|excellent)\s+(?:control|managed)", context_lower):
            return "well-controlled"
        elif re.search(r"(poor|bad|not well|un)\s*(?:control|managed)", context_lower):
            return "poorly-controlled"
        
        return None
    
    def _assess_symptom_severity(self, context: str) -> str:
        """
        Assess the severity of a symptom.
        
        Args:
            context: Text context for the symptom
            
        Returns:
            Severity assessment or None
        """
        context_lower = context.lower()
        
        for severity, terms in self.severity_terms.items():
            if any(term in context_lower for term in terms):
                return severity
        
        return None
    
    def _assess_symptom_duration(self, context: str) -> str:
        """
        Assess the duration of a symptom.
        
        Args:
            context: Text context for the symptom
            
        Returns:
            Duration assessment or None
        """
        context_lower = context.lower()
        
        for duration, terms in self.duration_terms.items():
            if any(term in context_lower for term in terms):
                return duration
        
        return None
    
    def _is_symptom_denied(self, context: str, symptom: str) -> bool:
        """
        Check if a symptom is being denied.
        
        Args:
            context: Text context for the symptom
            symptom: The symptom term
            
        Returns:
            True if symptom is denied, False otherwise
        """
        context_lower = context.lower()
        
        # Check for negation before the symptom
        for neg in self.negation_terms:
            neg_pattern = rf"{neg}\s+(?:\w+\s+){{0,3}}{re.escape(symptom)}"
            if re.search(neg_pattern, context_lower):
                return True
        
        # Check for common denial phrases
        denial_phrases = [
            "don't have", "doesn't bother", "not experiencing", 
            "no problem with", "not an issue", "no issues with"
        ]
        
        for phrase in denial_phrases:
            if phrase in context_lower:
                return True
        
        return False
    
    def _summarize_symptoms(self, symptoms: List[Dict[str, Any]]) -> str:
        """
        Create a readable summary of symptoms.
        
        Args:
            symptoms: List of symptom dictionaries
            
        Returns:
            Natural language summary of symptoms
        """
        if not symptoms:
            return "No unusual symptoms reported"
            
        if len(symptoms) == 1:
            symptom = symptoms[0]
            summary = f"Reports {symptom['symptom']}"
            if symptom.get('severity'):
                summary = f"Reports {symptom.get('severity')} {symptom['symptom']}"
            return summary
            
        symptom_texts = []
        for symptom in symptoms:
            symptom_text = symptom['symptom']
            if symptom.get('severity'):
                symptom_text = f"{symptom.get('severity')} {symptom_text}"
            symptom_texts.append(symptom_text)
            
        return "Reports " + ", ".join(symptom_texts[:-1]) + f" and {symptom_texts[-1]}"
    
    def _get_context(self, text: str, position: int, length: int) -> str:
        """Helper method to extract context around a position in text."""
        start_pos = max(0, position - 50)
        end_pos = min(len(text), position + length + 50)
        return text[start_pos:end_pos]
