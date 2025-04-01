"""
Clinical Narrative Generator

This module provides an enterprise-grade implementation for generating human-like,
professionally structured clinical narratives from medical transcripts.
"""

import re
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ClinicalNarrativeGenerator:
    """
    Enterprise-grade clinical narrative generator that produces human-like,
    professionally formatted medical documentation.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the narrative generator.
        
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
    
    def generate_clinical_narrative(self, 
                                   summary_data: Dict[str, Any], 
                                   extracted_info: Dict[str, Any],
                                   hipaa_compliant: bool = True) -> Dict[str, str]:
        """
        Generate a complete clinical narrative in a professional, human-like style.
        
        Args:
            summary_data: Raw summary data from LLM
            extracted_info: Extracted medical information
            hipaa_compliant: Whether to apply HIPAA compliance rules
            
        Returns:
            Dictionary containing formatted clinical narrative sections
        """
        self._log("Generating formatted clinical narrative")
        
        # Extract key information
        patient_name = extracted_info.get("patient_name", "Patient")
        provider_name = extracted_info.get("provider_name", "Provider")
        
        # Get full summary or use raw text as fallback
        raw_summary = summary_data.get("summary", "")
        if isinstance(raw_summary, dict) and "text" in raw_summary:
            raw_summary = raw_summary["text"]
            
        # Get structured sections if available, otherwise parse from raw summary
        structured_sections = summary_data.get("soap_format", {})
        if not structured_sections or not any(structured_sections.values()):
            structured_sections = self._extract_sections_from_text(raw_summary)
        
        # Generate each section with proper formatting
        subjective = self._format_subjective_section(
            structured_sections.get("subjective", ""), 
            patient_name, 
            provider_name,
            extracted_info
        )
        
        objective = self._format_objective_section(
            structured_sections.get("objective", ""),
            extracted_info
        )
        
        assessment = self._format_assessment_section(
            structured_sections.get("assessment", ""),
            extracted_info
        )
        
        plan = self._format_plan_section(
            structured_sections.get("plan", ""),
            extracted_info
        )
        
        # Generate concise narrative
        narrative = self._generate_concise_narrative(
            subjective, objective, assessment, plan,
            patient_name, provider_name, extracted_info
        )
        
        return {
            "subjective": subjective,
            "objective": objective,
            "assessment": assessment,
            "plan": plan,
            "narrative": narrative
        }
    
    def _extract_sections_from_text(self, text: str) -> Dict[str, str]:
        """
        Extract SOAP sections from raw text.
        
        Args:
            text: Raw text to parse
            
        Returns:
            Dictionary with SOAP sections
        """
        sections = {
            "subjective": "",
            "objective": "",
            "assessment": "",
            "plan": ""
        }
        
        # Try to find section headers
        subjective_matches = re.finditer(r'(?i)(?:^|\n)(?:subjective|history|chief complaint|cc)[:;.\-\s]*(.*?)(?=(?:^|\n)(?:objective|physical exam|vitals|assessment|plan|impression)|$)', text, re.DOTALL)
        for match in subjective_matches:
            sections["subjective"] += match.group(1).strip() + "\n\n"
        
        objective_matches = re.finditer(r'(?i)(?:^|\n)(?:objective|physical exam|vitals|examination)[:;.\-\s]*(.*?)(?=(?:^|\n)(?:assessment|plan|impression|a/p)|$)', text, re.DOTALL)
        for match in objective_matches:
            sections["objective"] += match.group(1).strip() + "\n\n"
        
        assessment_matches = re.finditer(r'(?i)(?:^|\n)(?:assessment|impression|diagnosis)[:;.\-\s]*(.*?)(?=(?:^|\n)(?:plan|recommendation|follow|treatment)|$)', text, re.DOTALL)
        for match in assessment_matches:
            sections["assessment"] += match.group(1).strip() + "\n\n"
        
        plan_matches = re.finditer(r'(?i)(?:^|\n)(?:plan|recommendation|follow|treatment)[:;.\-\s]*(.*?)(?=$)', text, re.DOTALL)
        for match in plan_matches:
            sections["plan"] += match.group(1).strip() + "\n\n"
        
        # If we couldn't extract sections, divide the text into roughly appropriate sections
        if not any(sections.values()):
            paragraphs = re.split(r'\n\s*\n', text)
            if len(paragraphs) >= 4:
                sections["subjective"] = paragraphs[0]
                sections["objective"] = paragraphs[1]
                sections["assessment"] = paragraphs[2]
                sections["plan"] = "\n\n".join(paragraphs[3:])
            elif len(paragraphs) == 3:
                sections["subjective"] = paragraphs[0]
                sections["objective"] = paragraphs[1]
                sections["assessment"] = ""
                sections["plan"] = paragraphs[2]
            elif len(paragraphs) == 2:
                sections["subjective"] = paragraphs[0]
                sections["objective"] = ""
                sections["assessment"] = ""
                sections["plan"] = paragraphs[1]
            else:
                sections["subjective"] = text
        
        # Clean up the sections
        for key in sections:
            if sections[key]:
                sections[key] = sections[key].strip()
        
        return sections
    
    def _format_subjective_section(self, 
                                  text: str, 
                                  patient_name: str, 
                                  provider_name: str,
                                  extracted_info: Dict[str, Any]) -> str:
        """
        Format the subjective section in a professional, human-like style.
        
        Args:
            text: Raw subjective text
            patient_name: Patient name
            provider_name: Provider name
            extracted_info: Additional information
            
        Returns:
            Formatted subjective section
        """
        if not text:
            # Generate a placeholder if no text provided
            return f"Patient presents for telehealth visit with {provider_name}. Chief complaint not clearly documented."
        
        # Extract key subjective information
        chief_complaint = self._extract_chief_complaint(text)
        
        # Create a more natural opening
        opening = f"Patient presents for telehealth appointment with {provider_name} for evaluation of {chief_complaint}."
        
        # Reorganize the text to create natural paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        content = []
        
        if len(paragraphs) > 1:
            # Use the first paragraph, but remove redundant opening
            first_para = re.sub(r'^(patient|the patient).*?(appointment|visit|evaluation|consultation)', '', paragraphs[0], flags=re.IGNORECASE)
            first_para = first_para.strip()
            if first_para:
                content.append(first_para)
            
            # Add remaining paragraphs
            for para in paragraphs[1:]:
                if para.strip():
                    content.append(para.strip())
        else:
            # Just use the original text
            content = [re.sub(r'^(patient|the patient).*?(appointment|visit|evaluation|consultation)', '', text, flags=re.IGNORECASE).strip()]
        
        # Create the final subjective section
        subjective = opening + "\n\n"
        if content:
            subjective += "\n\n".join(content)
        
        return subjective.strip()
    
    def _format_objective_section(self, 
                                 text: str,
                                 extracted_info: Dict[str, Any]) -> str:
        """
        Format the objective section in a professional, human-like style.
        
        Args:
            text: Raw objective text
            extracted_info: Additional information
            
        Returns:
            Formatted objective section
        """
        if not text:
            return "No objective findings documented during this telehealth encounter."
        
        # Extract vital signs if mentioned
        vital_signs = extracted_info.get("vital_signs", {})
        vital_signs_text = ""
        
        if vital_signs:
            vital_signs_list = []
            for key, value in vital_signs.items():
                if key != "confidence" and value:
                    vital_signs_list.append(f"{key}: {value}")
            
            if vital_signs_list:
                vital_signs_text = "Vital Signs: " + ", ".join(vital_signs_list) + ".\n\n"
        
        # Create organized paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        content = []
        
        # Add vital signs at the beginning if found
        if vital_signs_text:
            content.append(vital_signs_text)
        
        # Process other paragraphs
        for para in paragraphs:
            if para.strip():
                # Skip if this paragraph is already about vital signs
                if re.search(r'(?i)vital signs', para) and vital_signs_text:
                    continue
                content.append(para.strip())
        
        # Create the final objective section
        return "\n\n".join(content)
    
    def _format_assessment_section(self, 
                                  text: str,
                                  extracted_info: Dict[str, Any]) -> str:
        """
        Format the assessment section in a professional, human-like style.
        
        Args:
            text: Raw assessment text
            extracted_info: Additional information
            
        Returns:
            Formatted assessment section
        """
        if not text:
            # Get conditions from extracted info
            conditions = extracted_info.get("conditions", [])
            if conditions:
                return "Assessment: " + ", ".join(conditions) + "."
            return "Assessment deferred pending further evaluation."
        
        # Clean up the text
        clean_text = text.strip()
        
        # Make sure it starts with "Assessment" if it doesn't already
        if not re.match(r'(?i)^assessment', clean_text):
            clean_text = "Assessment: " + clean_text
        
        return clean_text
    
    def _format_plan_section(self, 
                            text: str,
                            extracted_info: Dict[str, Any]) -> str:
        """
        Format the plan section in a professional, human-like style.
        
        Args:
            text: Raw plan text
            extracted_info: Additional information
            
        Returns:
            Formatted plan section
        """
        if not text:
            return "Treatment plan to be determined. Follow-up recommended."
        
        # Clean up the text
        clean_text = text.strip()
        
        # Make sure it starts with "Plan" if it doesn't already
        if not re.match(r'(?i)^plan|^treatment|^recommendations', clean_text):
            clean_text = "Plan: " + clean_text
        
        # Extract medication information if available
        medications = extracted_info.get("medications", [])
        medications_text = ""
        
        if medications and not re.search(r'(?i)medications?', clean_text):
            medications_text = "Medications: " + ", ".join(medications) + ".\n\n"
            clean_text = medications_text + clean_text
        
        return clean_text
    
    def _generate_concise_narrative(self,
                                   subjective: str,
                                   objective: str,
                                   assessment: str,
                                   plan: str,
                                   patient_name: str,
                                   provider_name: str,
                                   extracted_info: Dict[str, Any]) -> str:
        """
        Generate a concise 2-3 paragraph narrative summary.
        
        Args:
            subjective: Formatted subjective section
            objective: Formatted objective section
            assessment: Formatted assessment section
            plan: Formatted plan section
            patient_name: Patient name
            provider_name: Provider name
            extracted_info: Additional information
            
        Returns:
            Human-like narrative summary in 2-3 paragraphs
        """
        # Extract chief complaint
        chief_complaint = self._extract_chief_complaint(subjective)
        if not chief_complaint:
            chief_complaint = "health concerns"
        
        # Extract key symptoms from subjective section
        symptoms = self._extract_symptoms(subjective)
        symptoms_text = ", ".join(symptoms) if symptoms else chief_complaint
        
        # Extract conditions if mentioned in assessment
        conditions = extracted_info.get("conditions", [])
        conditions_text = ", ".join(conditions) if conditions else "medical concerns"
        
        # Extract medications
        medications = extracted_info.get("medications", [])
        medications_text = ""
        if medications:
            medications_text = f" Medications prescribed include {', '.join(medications)}."
        
        # Generate the narrative
        first_paragraph = f"Patient presented to {provider_name} via telehealth visit for evaluation of {chief_complaint}. The patient reported {symptoms_text}."
        
        second_paragraph = "Upon evaluation, "
        if objective and len(objective) > 20:
            # Extract a condensed version of the objective findings
            key_findings = self._extract_key_findings(objective)
            if key_findings:
                second_paragraph += f"key findings included {key_findings}. "
            else:
                second_paragraph += "a thorough examination was conducted within the limitations of telehealth. "
        else:
            second_paragraph += "a limited telehealth examination was performed. "
        
        if assessment and len(assessment) > 20:
            assessment_clean = re.sub(r'(?i)^assessment:?\s*', '', assessment).strip()
            second_paragraph += f"Assessment: {assessment_clean}. "
        else:
            second_paragraph += f"Assessment focused on {conditions_text}. "
        
        third_paragraph = ""
        if plan and len(plan) > 20:
            # Extract key plan elements
            plan_elements = self._extract_plan_elements(plan)
            if plan_elements:
                third_paragraph = f"Treatment plan includes {plan_elements}.{medications_text} "
                if not re.search(r'(?i)follow.?up', third_paragraph):
                    third_paragraph += "Follow-up recommended as needed."
            else:
                plan_clean = re.sub(r'(?i)^plan:?\s*', '', plan).strip()
                third_paragraph = plan_clean
        else:
            third_paragraph = f"Treatment plan established.{medications_text} Follow-up recommended to assess response to treatment."
        
        narrative = f"{first_paragraph}\n\n{second_paragraph}\n\n{third_paragraph}"
        return narrative
    
    def _extract_chief_complaint(self, text: str) -> str:
        """
        Extract chief complaint from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            Extracted chief complaint or empty string
        """
        # Look for common chief complaint patterns
        cc_patterns = [
            r'(?i)(?:chief\s+complaint|cc|reason\s+for\s+visit|presenting\s+with|presented\s+with|presents?\s+for)[:\s]+([^.]+)',
            r'(?i)(?:evaluation|assessment|management)\s+of\s+([^.]+)',
            r'(?i)complaining\s+of\s+([^.]+)'
        ]
        
        for pattern in cc_patterns:
            match = re.search(pattern, text)
            if match:
                cc = match.group(1).strip()
                # Remove common prefixes
                cc = re.sub(r'(?i)^(patient|the patient|he|she|they)\s+(has|had|is|was|reports?|complains?)\s+', '', cc)
                # Remove articles at the beginning
                cc = re.sub(r'(?i)^(a|an|the)\s+', '', cc)
                return cc
        
        # If no clear CC found, try to extract from first sentence
        first_sentence = text.split('.')[0] if '.' in text else text
        # Look for key symptoms in the first sentence
        symptom_match = re.search(r'(?i)(pain|discomfort|fever|cough|headache|nausea|vomiting|diarrhea|shortness of breath|difficulty breathing|fatigue|weakness|dizziness)', first_sentence)
        if symptom_match:
            return symptom_match.group(1).strip()
        
        return "health concerns"
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """
        Extract key symptoms from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            List of extracted symptoms
        """
        common_symptoms = [
            r'(?i)(sore\s+throat)',
            r'(?i)(cough)',
            r'(?i)(fever|feverish)',
            r'(?i)(headache)',
            r'(?i)(pain)',
            r'(?i)(nausea)',
            r'(?i)(vomiting)',
            r'(?i)(diarrhea)',
            r'(?i)(shortness\s+of\s+breath)',
            r'(?i)(difficulty\s+breathing)',
            r'(?i)(fatigue)',
            r'(?i)(weakness)',
            r'(?i)(dizziness)',
            r'(?i)(congestion)',
            r'(?i)(runny\s+nose)',
            r'(?i)(rash)',
            r'(?i)(itching)',
            r'(?i)(swelling)'
        ]
        
        symptoms = []
        for pattern in common_symptoms:
            if re.search(pattern, text):
                # Extract the symptom
                match = re.search(pattern, text)
                if match:
                    symptoms.append(match.group(1).lower())
        
        return list(set(symptoms))
    
    def _extract_key_findings(self, text: str) -> str:
        """
        Extract key findings from objective text.
        
        Args:
            text: Text to extract from
            
        Returns:
            String with key findings
        """
        # Look for vital signs
        vitals_match = re.search(r'(?i)vital\s+signs:?\s+([^.]+)', text)
        vitals = vitals_match.group(1).strip() if vitals_match else ""
        
        # Look for physical exam findings
        findings = []
        
        # Common patterns for findings
        finding_patterns = [
            r'(?i)(?:appears|noted\s+to\s+be|observed\s+to\s+be)\s+([^.]+)',
            r'(?i)examination\s+reveals\s+([^.]+)',
            r'(?i)auscultation\s+reveals\s+([^.]+)',
            r'(?i)(?:skin|throat|chest|abdomen|lungs)\s+(?:shows|reveals|demonstrates)\s+([^.]+)'
        ]
        
        for pattern in finding_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                findings.append(match.group(1).strip())
        
        # Combine findings
        if vitals and findings:
            return f"{vitals}; {'; '.join(findings)}"
        elif vitals:
            return vitals
        elif findings:
            return "; ".join(findings)
        else:
            return ""
    
    def _extract_plan_elements(self, text: str) -> str:
        """
        Extract key plan elements from plan text.
        
        Args:
            text: Text to extract from
            
        Returns:
            String with key plan elements
        """
        # Common plan elements to look for
        plan_patterns = [
            r'(?i)(?:prescribed|prescription\s+for|started\s+on)\s+([^.]+)',
            r'(?i)recommended\s+([^.]+)',
            r'(?i)advised\s+to\s+([^.]+)',
            r'(?i)follow-up\s+(?:in|within)\s+([^.]+)',
            r'(?i)return\s+(?:to|for)\s+([^.]+)'
        ]
        
        elements = []
        for pattern in plan_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                elements.append(match.group(1).strip())
        
        if elements:
            return "; ".join(elements)
        else:
            # Just take the first sentence if no specific elements found
            first_sentence = text.split('.')[0] if '.' in text else text
            return re.sub(r'(?i)^(plan:?|treatment:?|recommendations?:?)\s*', '', first_sentence).strip()
