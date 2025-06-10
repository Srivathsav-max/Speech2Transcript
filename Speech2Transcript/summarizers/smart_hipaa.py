import re
import logging
from typing import Dict, Any, List, Set, Tuple, Optional
import hashlib
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class SmartHIPAAProcessor:
    # PHI categories based on HIPAA Safe Harbor method (18 identifiers)
    PHI_CATEGORIES = {
        "names": {
            "patterns": [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',  # Full names
                r'(?i)\b(?:mr|mrs|ms|miss|dr)\.\s+[a-z]+\b',  # Names with titles
            ],
            "placeholder": "Patient",
            "context_placeholders": {
                "doctor": "Dr. [Provider]",
                "nurse": "[Nurse]",
                "patient": "[Patient]"
            },
            "common_medical_names": [
                "patient", "provider", "doctor", "nurse", "physician", "clinician", 
                "practitioner", "therapist", "specialist", "pharmacist", "caregiver"
            ]
        },
        "locations": {
            "patterns": [
                r'\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|Avenue|Road|Drive|Lane|Place|Court|Blvd|Boulevard|Way|Circle|Hwy|Highway)\b',
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?\b'  # City, State ZIP
            ],
            "placeholder": "[Address]"
        },
        "dates": {
            "patterns": [
                r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:[,\s]+\d{4})?\b',
                r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'  # MM/DD/YYYY
            ],
            "placeholder": "[Date]",
            "preserved_pattern": r'\b(?:today|yesterday|tomorrow|last\s+(?:week|month|year)|next\s+(?:week|month|year))\b'
        },
        "times": {
            "patterns": [
                r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)\b',
                r'\b\d{1,2}\s*(?:AM|PM|am|pm)\b'
            ],
            "placeholder": "[Time]"
        },
        "phone_numbers": {
            "patterns": [
                r'\b\(?(?:\d{3})\)?[-.\s]?(?:\d{3})[-.\s]?(?:\d{4})\b'
            ],
            "placeholder": "[Phone]"
        },
        "emails": {
            "patterns": [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            "placeholder": "[Email]"
        },
        "ssn": {
            "patterns": [
                r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b'
            ],
            "placeholder": "[SSN]"
        },
        "mrn": {
            "patterns": [
                r'\b(?:MRN|Medical Record Number|Patient ID|Record Number)[\s:#]*\d+\b',
                r'\b\d{5,10}\b'  # Potential MRN (only in context)
            ],
            "placeholder": "[MRN]"
        },
        "account_numbers": {
            "patterns": [
                r'\b(?:Account|Acct)[\s:#]*\d+\b'
            ],
            "placeholder": "[Account]"
        },
        "age": {
            "patterns": [
                r'\b(?:age[d\s:]+|is\s+)(\d{1,3})(?:\s+years?\s+old)?\b'
            ],
            "placeholder": "[Age]",
            "preserve_ranges": True,
            "range_replacement": "adult"
        }
    }
    
    # Clinical terms to preserve or maintain in a standard format
    CLINICAL_TERMS = {
        "abbreviations": {
            # Common medical abbreviations
            r'\bCC\b': "Chief Complaint",
            r'\bHPI\b': "History of Present Illness",
            r'\bROS\b': "Review of Systems",
            r'\bPE\b': "Physical Examination",
            r'\bA/P\b': "Assessment and Plan",
            r'\bHTN\b': "hypertension",
            r'\bDM\b': "diabetes mellitus",
            r'\bCAD\b': "coronary artery disease",
            r'\bCHF\b': "congestive heart failure",
            r'\bCOPD\b': "chronic obstructive pulmonary disease",
            r'\bBP\b': "blood pressure",
            r'\bHR\b': "heart rate",
            r'\bRR\b': "respiratory rate",
            r'\bO2\s+sat\b': "oxygen saturation",
            r'\bT\b': "temperature",
            r'\bJVD\b': "jugular venous distention",
            r'\bBID\b': "twice daily",
            r'\bTID\b': "three times daily",
            r'\bQID\b': "four times daily",
            r'\bPRN\b': "as needed",
            r'\bPO\b': "by mouth",
            r'\bIV\b': "intravenous",
            r'\bIM\b': "intramuscular",
            r'\bSC\b': "subcutaneous",
            r'\bq\.\s*d\.\b': "daily",
            r'\bg\b': "gram",
            r'\bmg\b': "milligram",
            r'\bmcg\b': "microgram",
            r'\bmL\b': "milliliter",
            r'\bL\b': "liter",
            r'\bng\b': "nanogram",
            r'\bU\b': "unit",
            r'\bcm\b': "centimeter",
            r'\bmm\b': "millimeter",
            r'\bin\b': "inch",
            r'\bft\b': "feet",
            r'\blb\b': "pound",
            r'\bkg\b': "kilogram"
        },
        "section_headers": {
            # Standard section headers to preserve
            r'\b(?:SUBJECTIVE|SUBJ|S)[:;.\-\s]*': "SUBJECTIVE:",
            r'\b(?:OBJECTIVE|OBJ|O)[:;.\-\s]*': "OBJECTIVE:",
            r'\b(?:ASSESSMENT|ASS|A)[:;.\-\s]*': "ASSESSMENT:",
            r'\b(?:PLAN|P)[:;.\-\s]*': "PLAN:",
            r'\b(?:CHIEF COMPLAINT|CC)[:;.\-\s]*': "CHIEF COMPLAINT:",
            r'\b(?:HISTORY OF PRESENT ILLNESS|HPI)[:;.\-\s]*': "HISTORY OF PRESENT ILLNESS:",
            r'\b(?:REVIEW OF SYSTEMS|ROS)[:;.\-\s]*': "REVIEW OF SYSTEMS:",
            r'\b(?:PHYSICAL EXAMINATION|PE)[:;.\-\s]*': "PHYSICAL EXAMINATION:",
            r'\b(?:VITAL SIGNS|VS)[:;.\-\s]*': "VITAL SIGNS:",
            r'\b(?:ALLERGIES|ALL)[:;.\-\s]*': "ALLERGIES:",
            r'\b(?:MEDICATIONS|MEDS)[:;.\-\s]*': "MEDICATIONS:",
            r'\b(?:PAST MEDICAL HISTORY|PMH)[:;.\-\s]*': "PAST MEDICAL HISTORY:",
            r'\b(?:FAMILY HISTORY|FH)[:;.\-\s]*': "FAMILY HISTORY:",
            r'\b(?:SOCIAL HISTORY|SH)[:;.\-\s]*': "SOCIAL HISTORY:",
            r'\b(?:IMPRESSION|IMP)[:;.\-\s]*': "IMPRESSION:",
            r'\b(?:DIAGNOSIS|DX)[:;.\-\s]*': "DIAGNOSIS:",
            r'\b(?:FOLLOW-UP|FU)[:;.\-\s]*': "FOLLOW-UP:"
        }
    }
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the HIPAA processor.
        
        Args:
            logger: Optional logger for messages
        """
        self.logger = logger
        
    def _log(self, message: str, level: str = "info") -> None:
        """
        Log messages if logger is available.
        
        Args:
            message: Message to log
            level: Log level (info, error, warning)
        """
        if self.logger:
            if level == "info":
                self.logger.info(message)
            elif level == "error":
                self.logger.error(message)
            elif level == "warning":
                self.logger.warning(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def process_document(self, document: Dict[str, Any], content_field: str = "summary") -> Dict[str, Any]:
        """
        Process a document for HIPAA compliance.
        
        Args:
            document: The document dictionary
            content_field: The field containing the content to process
            
        Returns:
            Processed document with HIPAA-compliant content
        """
        if content_field not in document:
            self._log(f"Field {content_field} not found in document", level="error")
            return document
        
        # Create a copy to avoid modifying the original
        processed_doc = document.copy()
        
        # Extract content
        content = processed_doc[content_field]
        
        # Apply smart redaction
        processed_content = self.apply_smart_redaction(content)
        
        # Update the document
        processed_doc[content_field] = processed_content
        
        # Add HIPAA compliance metadata
        processed_doc["metadata"] = processed_doc.get("metadata", {})
        processed_doc["metadata"]["hipaa_processed"] = True
        processed_doc["metadata"]["hipaa_processing_timestamp"] = datetime.now().isoformat()
        
        return processed_doc
    
    def apply_smart_redaction(self, text: str) -> str:
        """
        Apply intelligent, context-aware HIPAA redaction.
        
        Args:
            text: Text to redact
            
        Returns:
            Redacted text that preserves clinical readability
        """
        if not text:
            return text
        
        # First, standardize clinical terms and abbreviations
        standardized_text = self._standardize_clinical_terms(text)
        
        # Then apply smart PHI redaction
        redacted_text = self._redact_phi(standardized_text)
        
        # Apply final cleanup
        redacted_text = self._cleanup_redacted_text(redacted_text)
        
        return redacted_text
    
    def _standardize_clinical_terms(self, text: str) -> str:
        """
        Standardize clinical terms and abbreviations.
        
        Args:
            text: Text to standardize
            
        Returns:
            Text with standardized clinical terms
        """
        standardized = text
        
        # Standardize section headers
        for pattern, replacement in self.CLINICAL_TERMS["section_headers"].items():
            standardized = re.sub(pattern, replacement, standardized)
        
        # Preserve clinical abbreviations that don't need to be expanded
        preserve_patterns = [r'\bBP\b', r'\bHR\b', r'\bRR\b', r'\bT\b', r'\bO2\b']
        preserve_sections = {}
        
        for i, pattern in enumerate(preserve_patterns):
            matches = list(re.finditer(pattern, standardized))
            for j, match in enumerate(matches):
                placeholder = f"__PRESERVED_TERM_{i}_{j}__"
                preserve_sections[placeholder] = match.group(0)
                standardized = standardized.replace(match.group(0), placeholder, 1)
        
        # Expand medical abbreviations
        for pattern, replacement in self.CLINICAL_TERMS["abbreviations"].items():
            standardized = re.sub(pattern, replacement, standardized)
        
        # Restore preserved sections
        for placeholder, original in preserve_sections.items():
            standardized = standardized.replace(placeholder, original)
        
        return standardized
    
    def _redact_phi(self, text: str) -> str:
        """
        Redact PHI with intelligent, context-aware replacements.
        
        Args:
            text: Text to redact
            
        Returns:
            Redacted text
        """
        redacted = text
        
        # Process each PHI category
        for category, config in self.PHI_CATEGORIES.items():
            patterns = config["patterns"]
            default_placeholder = config["placeholder"]
            
            # Find all matches for this category
            all_matches = []
            for pattern in patterns:
                matches = list(re.finditer(pattern, redacted))
                all_matches.extend(matches)
            
            # Sort matches by position (to process from end to beginning)
            all_matches.sort(key=lambda m: m.start(), reverse=True)
            
            # Track processed spans to avoid double-redacting
            processed_spans = set()
            
            for match in all_matches:
                # Skip if already processed
                span = (match.start(), match.end())
                if any(s[0] <= span[0] and span[1] <= s[1] for s in processed_spans):
                    continue
                
                # Get the matching text
                match_text = match.group(0)
                
                # Skip if this is a standard clinical term that should be preserved
                if category == "names" and match_text.lower() in config.get("common_medical_names", []):
                    continue
                
                # Determine appropriate placeholder based on context
                placeholder = self._get_context_aware_placeholder(match, redacted, category, config)
                
                # Replace the match with the placeholder
                redacted = redacted[:match.start()] + placeholder + redacted[match.end():]
                
                # Mark as processed
                processed_spans.add(span)
        
        return redacted
    
    def _get_context_aware_placeholder(self, match, text: str, category: str, config: Dict) -> str:
        """
        Determine the most appropriate placeholder based on context.
        
        Args:
            match: The regex match object
            text: The full text
            category: PHI category
            config: Category configuration
            
        Returns:
            Context-appropriate placeholder
        """
        default_placeholder = config["placeholder"]
        
        # Special handling for names
        if category == "names":
            # Check if it's referring to a doctor
            pre_context = text[max(0, match.start() - 20):match.start()]
            if re.search(r'(?i)(doctor|dr|physician|provider|clinician)', pre_context):
                return config["context_placeholders"]["doctor"]
            
            # Check if it's referring to a nurse
            if re.search(r'(?i)(nurse|rn|lpn|nursing)', pre_context):
                return config["context_placeholders"]["nurse"]
            
            # Default to patient name
            return config["context_placeholders"]["patient"]
        
        # Special handling for ages
        if category == "age" and config.get("preserve_ranges", False):
            age_value = int(match.group(1)) if hasattr(match, "group") and match.group(1).isdigit() else 0
            if age_value:
                if age_value < 18:
                    return "minor"
                elif age_value >= 65:
                    return "elderly adult"
                else:
                    return "adult"
        
        # For other categories, use default placeholder
        return default_placeholder
    
    def _cleanup_redacted_text(self, text: str) -> str:
        """
        Clean up the redacted text to improve readability.
        
        Args:
            text: Redacted text
            
        Returns:
            Cleaned up text
        """
        cleaned = text
        
        # Fix repeated placeholders in sequence
        placeholder_patterns = [
            (r'\[Patient\]\s+\[Patient\]', '[Patient]'),
            (r'\[Date\]\s+\[Date\]', '[Date]'),
            (r'\[Time\]\s+\[Time\]', '[Time]')
        ]
        
        for pattern, replacement in placeholder_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        # Fix capitalization issues
        cleaned = re.sub(r'(?<=\.)\s+([a-z])', lambda m: ' ' + m.group(1).upper(), cleaned)
        
        # Fix spacing around placeholders
        cleaned = re.sub(r'\s+\[', ' [', cleaned)
        cleaned = re.sub(r'\]\s+', '] ', cleaned)
        
        # Fix duplicate punctuation
        cleaned = re.sub(r'\.+', '.', cleaned)
        cleaned = re.sub(r'\,+', ',', cleaned)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Fix spacing after periods
        cleaned = re.sub(r'\.(?=[A-Za-z])', '. ', cleaned)
        
        return cleaned.strip()
    
    def verify_hipaa_compliance(self, text: str) -> Dict[str, Any]:
        """
        Verify HIPAA compliance of a text.
        
        Args:
            text: Text to verify
            
        Returns:
            Dictionary with compliance information
        """
        potentially_identifiable = False
        potential_phi = {}
        
        # Check each PHI category
        for category, config in self.PHI_CATEGORIES.items():
            patterns = config["patterns"]
            
            for pattern in patterns:
                matches = list(re.finditer(pattern, text))
                if matches:
                    potentially_identifiable = True
                    potential_phi[category] = [m.group(0) for m in matches]
        
        recommendations = []
        if potentially_identifiable:
            recommendations.append("Apply smart redaction to ensure HIPAA compliance")
            recommendations.append("Review and validate automated redaction results")
            recommendations.append("Consider manual review for sensitive information")
        
        return {
            "compliant": not potentially_identifiable,
            "potential_phi": potential_phi,
            "recommendations": recommendations
        }
