"""
Medication extraction module for identifying medications, dosages, and adherence information.
"""
import re
from typing import Dict, List, Any
from .base_extractor import BaseExtractor

class MedicationExtractor(BaseExtractor):
    """
    Specialized extractor for medications, dosages, frequencies, and adherence
    information from medical conversations.
    """
    
    def __init__(self, logger=None):
        """Initialize the medication extractor."""
        super().__init__(logger)
        self._initialize_medication_data()
        self._compile_patterns()
    
    def _initialize_medication_data(self):
        """Initialize medication-related data and knowledge base."""
        # Common medications for better detection
        self.common_medications = [
            "insulin", "metformin", "ozempic", "zempek", "januvia", "jardiance", "victoza",
            "trulicity", "rybelsus", "glipizide", "glimepiride", "actos", "farxiga", "invokana",
            "lisinopril", "losartan", "valsartan", "amlodipine", "metoprolol", "atenolol",
            "hydrochlorothiazide", "furosemide", "spironolactone", "atorvastatin", "simvastatin",
            "rosuvastatin", "pravastatin", "crestor", "lipitor", "zocor", "aspirin", "plavix",
            "warfarin", "eliquis", "xarelto", "coumadin", "synthroid", "levothyroxine"
        ]
        
        # Medication categories for better understanding
        self.medication_categories = {
            "diabetes": [
                "insulin", "metformin", "glipizide", "glimepiride", "ozempic", "zempek", 
                "januvia", "jardiance", "trulicity", "rybelsus", "farxiga", "invokana"
            ],
            "hypertension": [
                "lisinopril", "losartan", "valsartan", "amlodipine", "metoprolol", 
                "atenolol", "hydrochlorothiazide", "furosemide", "spironolactone"
            ],
            "cholesterol": [
                "atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", 
                "crestor", "lipitor", "zocor"
            ],
            "anticoagulant": [
                "aspirin", "plavix", "warfarin", "eliquis", "xarelto", "coumadin"
            ],
            "thyroid": [
                "synthroid", "levothyroxine"
            ]
        }
        
        # Common medication suffixes for recognition
        self.medication_suffixes = [
            "pril", "sartan", "olol", "dipine", "statin", "mab", "mide", "zole", 
            "mycin", "cycline", "formin", "prazole", "vir", "tide", "dronate"
        ]
    
    def _compile_patterns(self):
        """Compile regex patterns for medication extraction."""
        self.patterns = {
            # Medication frequency patterns
            "medication_freq": re.compile(r'(\d+)\s+times\s+(?:a|per)\s+(day|week|month)'),
            
            # Medication dosage patterns
            "medication_dose": re.compile(r'(\d+(?:\.\d+)?)\s*(?:mg|mcg|mL|g|unit)'),
            
            # Medication taking patterns
            "taking_meds": re.compile(r'(?:tak(?:e|ing)|us(?:e|ing))\s+(?:all|my|the)\s+(?:medications|meds)'),
            "med_adherence": re.compile(r'(?:tak(?:e|ing)|us(?:e|ing))\s+(?:medications|meds)\s+(?:as|like)\s+(?:prescribed|directed|told)')
        }
    
    def extract(self, text: str, entities: List[Dict] = None) -> Dict[str, Any]:
        """
        Extract medication information from text.
        
        Args:
            text: Text to analyze
            entities: Optional list of entities from NER
            
        Returns:
            Dictionary with medication information
        """
        self._log("Extracting medication information")
        
        # Initialize results structure
        results = {
            "medications": self._extract_medications(text, entities),
            "adherence": self._extract_adherence_info(text),
            "side_effects": self._extract_side_effects(text)
        }
        
        return results
    
    def _extract_medications(self, text: str, entities: List[Dict] = None) -> List[Dict[str, Any]]:
        """
        Extract detailed medication information from text.
        
        Args:
            text: Text to analyze
            entities: Optional entities from NER
            
        Returns:
            List of medication dictionaries
        """
        medications = []
        text_lower = text.lower()
        
        # Method 1: Extract from common medication names
        for med_name in self.common_medications:
            if med_name.lower() in text_lower:
                med_pos = text_lower.find(med_name.lower())
                if med_pos >= 0:
                    # Get surrounding context
                    context = self._get_context(text, med_pos, len(med_name))
                    
                    # Don't add duplicates
                    if not any(med["name"].lower() == med_name.lower() for med in medications):
                        med_info = self._extract_medication_details(med_name, context)
                        medications.append(med_info)
        
        # Method 2: Look for medication-taking patterns
        med_patterns = [
            # Look for medication names followed by dosage
            r'\b((?:[A-Za-z]+(?:in|ol|ide|one|ine|pril|sartan|statin|mab|zole|mycin|cycline|formin|prazole))\b)\s*(?:\d+\s*(?:mg|mcg|mL|units))?',
            
            # Look for phrases indicating medication usage
            r'(?:taking|on|prescribed|using|start(?:ed|ing)?)\s+([A-Za-z]+(?:in|ol|ide|one|ine)\b)'
        ]
        
        # Find all potential medications
        for pattern in med_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                med_name = match.group(1)
                
                # Skip if already found or too short
                if len(med_name) < 4 or any(med["name"].lower() == med_name.lower() for med in medications):
                    continue
                    
                # Get context
                context = self._get_context(text, match.start(), len(match.group(0)))
                
                # Add to medications list
                med_info = self._extract_medication_details(med_name, context)
                medications.append(med_info)
        
        # Method 3: Use NER entities if provided
        if entities:
            for entity in entities:
                # Skip if not a medication entity
                if not (entity.get("type") == "MEDICATION" or 
                        entity.get("entity") == "MEDICATION" or
                        entity.get("custom_type") == "MEDICATION"):
                    continue
                    
                med_name = entity.get("word", "").strip()
                
                # Skip if empty, too short, or already found
                if not med_name or len(med_name) < 4 or any(med["name"].lower() == med_name.lower() for med in medications):
                    continue
                
                # Get context
                context = entity.get("context", "")
                if not context:
                    med_pos = text_lower.find(med_name.lower())
                    if med_pos >= 0:
                        context = self._get_context(text, med_pos, len(med_name))
                
                # Add to medications list
                med_info = self._extract_medication_details(med_name, context)
                med_info["confidence"] = entity.get("score", 0.7)
                medications.append(med_info)
        
        # Determine medication categories
        for med in medications:
            med["category"] = self._determine_category(med["name"])
        
        return medications
    
    def _extract_medication_details(self, med_name: str, context: str) -> Dict[str, Any]:
        """
        Extract detailed information about a medication from its context.
        
        Args:
            med_name: Name of the medication
            context: Surrounding text context
            
        Returns:
            Dictionary with medication details
        """
        context_lower = context.lower()
        
        # Initialize medication details
        med_info = {
            "name": med_name,
            "context": context,
            "dosage": None,
            "frequency": None,
            "is_active": True,
            "category": None,
            "confidence": 0.8
        }
        
        # Extract dosage
        dose_match = self.patterns["medication_dose"].search(context)
        if dose_match:
            med_info["dosage"] = dose_match.group(0)
        
        # Extract frequency
        freq_match = self.patterns["medication_freq"].search(context)
        if freq_match:
            med_info["frequency"] = f"{freq_match.group(1)} times per {freq_match.group(2)}"
        elif "daily" in context_lower:
            med_info["frequency"] = "daily"
        elif "twice a day" in context_lower:
            med_info["frequency"] = "twice daily"
        elif "once a week" in context_lower:
            med_info["frequency"] = "weekly"
        
        # Check if medication is discontinued or not active
        if re.search(r"(stopped|quit|don't take|not (?:on|taking)|discontinued)", context_lower):
            med_info["is_active"] = False
        
        return med_info
    
    def _determine_category(self, med_name: str) -> str:
        """
        Determine the category of a medication.
        
        Args:
            med_name: Name of the medication
            
        Returns:
            Category name or None
        """
        med_lower = med_name.lower()
        
        # Check each category
        for category, meds in self.medication_categories.items():
            if any(med_lower == med.lower() for med in meds):
                return category
        
        # Try to guess based on medication suffix
        for suffix in self.medication_suffixes:
            if med_lower.endswith(suffix):
                if suffix == "pril" or suffix == "sartan":
                    return "hypertension"
                elif suffix == "statin":
                    return "cholesterol"
                elif suffix == "formin":
                    return "diabetes"
        
        return None
    
    def _extract_adherence_info(self, text: str) -> str:
        """
        Extract medication adherence information from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Adherence information as a string
        """
        text_lower = text.lower()
        
        # Look for explicit adherence statements
        adherence_patterns = [
            r"(tak(?:e|ing)|using|follow)\s+(?:all|my|the)\s+(?:medications|meds)\s+(?:as|like)\s+(?:prescribed|directed|told)",
            r"(?:yes|yeah|correct),?\s+(?:I\s+)?(tak(?:e|ing)|using)\s+(?:all|my|the|those)",
            r"(?:good|great|excellent)\s+(?:with|about)\s+(?:medications|meds)"
        ]
        
        for pattern in adherence_patterns:
            if re.search(pattern, text_lower):
                return "Patient reports taking medications as prescribed"
                
        # Look for non-adherence statements
        non_adherence_patterns = [
            r"(?:miss(?:ed|ing)|skip(?:ped|ping)|forgot|don't take)\s+(?:sometimes|occasionally|often|my|the)\s+(?:medications|meds)",
            r"(?:not|haven't been)\s+(?:tak(?:e|ing)|using)\s+(?:all|some|my|the)\s+(?:medications|meds)",
            r"(?:stopped|quit|don't take)\s+(?:medications|meds)"
        ]
        
        for pattern in non_adherence_patterns:
            if re.search(pattern, text_lower):
                return "Patient reports issues with medication adherence"
        
        # Default response based on general impression
        if re.search(r"(tak(?:e|ing)|using)\s+(?:medications|meds)", text_lower):
            return "Patient appears to be taking medications as prescribed"
        
        return "Medication adherence status unclear from conversation"
    
    def _extract_side_effects(self, text: str) -> str:
        """
        Extract medication side effects information from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Side effects information as a string
        """
        text_lower = text.lower()
        
        # Look for explicit side effect discussion
        if re.search(r"(no|not|don't have|haven't had)\s+(side effects|problems|issues)", text_lower):
            return "No side effects reported"
            
        side_effect_patterns = [
            r"(?:having|have|experiencing)\s+(?:some|a few|several)?\s+(?:side effects|problems)",
            r"(?:medication|med|drug)\s+(?:makes|making|caused|causing)\s+me\s+(?:feel|have)"
        ]
        
        for pattern in side_effect_patterns:
            match = re.search(pattern + r".*?(?:\.|$)", text_lower)
            if match:
                return match.group(0)
                
        return "No side effects mentioned"
    
    def _get_context(self, text: str, position: int, length: int) -> str:
        """Helper method to extract context around a position in text."""
        start_pos = max(0, position - 50)
        end_pos = min(len(text), position + length + 50)
        return text[start_pos:end_pos]
