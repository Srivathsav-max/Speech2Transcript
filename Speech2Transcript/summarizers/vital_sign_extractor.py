"""
Vital sign extraction module for identifying and extracting vital measurements from medical conversations.
"""
import re
from typing import Dict, Any, List
from .base_extractor import BaseExtractor

class VitalSignExtractor(BaseExtractor):
    """
    Specialized extractor for vital signs (blood pressure, glucose, weight, etc.)
    from medical conversations.
    """
    
    def __init__(self, logger=None):
        """Initialize the vital sign extractor."""
        super().__init__(logger)
        self._compile_extraction_patterns()
    
    def _compile_extraction_patterns(self):
        """Compile comprehensive regex patterns for extracting vital measurements."""
        self.patterns = {
            # Blood pressure patterns with more variants
            "blood_pressure": re.compile(r'(\d{2,3})[\s/,.-]{1,2}(\d{2,3})(?:\s*mm\s*Hg)?'),
            
            # Glucose patterns with unit variants
            "glucose": re.compile(r'(\d{2,3})(?:(?:\s*[-–]\s*\d{2,3})?(?:\s*mg/dL)?|(?:\s*mmol/L)?)'),
            
            # Weight patterns (both absolute and change)
            "weight": re.compile(r'(\d{2,3})(?:\s*(?:pounds|lbs|kg|kilos))'),
            "weight_change": re.compile(r'(?:lost|gained)\s+(\d+)\s+(?:pounds|lbs|kg|kilos)'),
            
            # Time patterns (for timing of measurements)
            "time_pattern": re.compile(r'\b(\d{1,2}):(\d{2})\s*(am|pm)?\b', re.IGNORECASE),
        }
        
        # Define vital sign terms for context recognition
        self.vital_terms = {
            "blood_pressure": ["blood pressure", "bp", "pressure", "hypertension"],
            "blood_glucose": ["glucose", "sugar", "blood sugar", "diabetes", "diabetic", "glycemia"],
            "weight": ["weight", "pounds", "lbs", "kg", "kilos"],
            "temperature": ["temp", "temperature", "fever"]
        }
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract vital sign information from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of vital sign measurements
        """
        self._log("Extracting vital signs")
        
        vitals = {
            "blood_pressure": [],
            "glucose": [],
            "weight": None,
            "weight_change": None
        }
        
        # Extract blood pressure measurements
        vitals["blood_pressure"] = self._extract_blood_pressure(text)
        
        # Extract glucose measurements
        vitals["glucose"] = self._extract_glucose(text)
        
        # Extract weight information
        weight_info = self._extract_weight_info(text)
        vitals["weight"] = weight_info.get("weight")
        vitals["weight_change"] = weight_info.get("weight_change")
        
        return vitals
    
    def _extract_blood_pressure(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract blood pressure measurements from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of blood pressure measurements
        """
        bp_results = []
        
        # Enhanced blood pressure extraction (format: 120/80, BP 120/80, etc.)
        bp_contexts = []
        for bp_term in self.vital_terms["blood_pressure"]:
            for match in re.finditer(r'\b' + bp_term + r'.*?(\d{2,3})[\s/,.-]{1,2}(\d{2,3})', text, re.IGNORECASE):
                bp_contexts.append(match.group(0))
                
        # Also look for standalone BP values (e.g., "120/80")
        for match in re.finditer(r'\b(\d{2,3})[\s/,.-]{1,2}(\d{2,3})\b', text):
            # Check if this might be a BP reading (not a date, etc.)
            context_start = max(0, match.start() - 30)
            context_end = min(len(text), match.end() + 30)
            context = text[context_start:context_end].lower()
            
            if not any(excluded in context for excluded in ["date", "time", "clock", "ratio", "score"]):
                bp_contexts.append(match.group(0))
                
        # Process all potential BP readings
        for context in bp_contexts:
            bp_match = re.search(r'(\d{2,3})[\s/,.-]{1,2}(\d{2,3})', context)
            if bp_match:
                try:
                    systolic, diastolic = int(bp_match.group(1)), int(bp_match.group(2))
                    # Check if values are in a reasonable range
                    if 80 <= systolic <= 200 and 40 <= diastolic <= 130:
                        bp_results.append({
                            "systolic": systolic,
                            "diastolic": diastolic,
                            "full": f"{systolic}/{diastolic}",
                            "context": context
                        })
                except ValueError:
                    pass
        
        return bp_results
    
    def _extract_glucose(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract glucose measurements from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of glucose measurements
        """
        glucose_results = []
        
        # Enhanced glucose extraction with context analysis
        glucose_contexts = []
        for glucose_term in self.vital_terms["blood_glucose"]:
            for match in re.finditer(r'\b' + glucose_term + r'.*?(\d{2,3})', text, re.IGNORECASE):
                glucose_contexts.append(match.group(0))
                
        # Process all potential glucose readings
        for context in glucose_contexts:
            glucose_match = re.search(r'(\d{2,3})', context)
            if glucose_match:
                try:
                    glucose_val = int(glucose_match.group(1))
                    if 70 <= glucose_val <= 300:  # Valid glucose range
                        glucose_results.append({
                            "value": glucose_val,
                            "unit": "mg/dL",
                            "context": context
                        })
                except ValueError:
                    pass
        
        # Look for specific values mentioned in morning/fasting contexts
        fasting_contexts = re.finditer(r'(morning|fasting|empty stomach).*?(\d{2,3})', text, re.IGNORECASE)
        for match in fasting_contexts:
            try:
                glucose_val = int(match.group(2))
                if 70 <= glucose_val <= 300:
                    glucose_results.append({
                        "value": glucose_val,
                        "unit": "mg/dL",
                        "context": match.group(0),
                        "type": "fasting"
                    })
            except ValueError:
                pass
        
        return glucose_results
    
    def _extract_weight_info(self, text: str) -> Dict[str, Any]:
        """
        Extract weight and weight change information from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with weight and weight change information
        """
        result = {
            "weight": None,
            "weight_change": None
        }
        
        # Look for absolute weight mentions
        weight_matches = re.finditer(r'(\d{2,3})\s*(?:pound|lb|kg|kilo)', text, re.IGNORECASE)
        for match in weight_matches:
            try:
                weight_val = int(match.group(1))
                if 50 <= weight_val <= 500:  # Reasonable weight range
                    result["weight"] = {
                        "value": weight_val,
                        "unit": "pounds",
                        "context": self._get_context(text, match)
                    }
                    break
            except ValueError:
                pass
        
        # Weight change detection with improved pattern recognition
        weight_change_contexts = []
        for weight_term in ["lost", "gained", "lose", "gain", "dropped"]:
            for match in re.finditer(r'\b' + weight_term + r'.*?(\d{1,3})\s*(?:pound|lb|kg|kilo)', text, re.IGNORECASE):
                weight_change_contexts.append(match.group(0))
                
        # Process all potential weight change mentions
        for context in weight_change_contexts:
            weight_match = re.search(r'(\d{1,3})', context)
            if weight_match:
                try:
                    weight_val = int(weight_match.group(1))
                    if 1 <= weight_val <= 100:  # Reasonable weight change
                        is_loss = any(term in context.lower() for term in ["lost", "lose", "dropped"])
                        result["weight_change"] = {
                            "value": weight_val,
                            "direction": "loss" if is_loss else "gain",
                            "context": context
                        }
                except ValueError:
                    pass
        
        return result
    
    def _get_context(self, text: str, match) -> str:
        """Helper method to extract context around a regex match."""
        start_pos = max(0, match.start() - 30)
        end_pos = min(len(text), match.end() + 30)
        return text[start_pos:end_pos]
