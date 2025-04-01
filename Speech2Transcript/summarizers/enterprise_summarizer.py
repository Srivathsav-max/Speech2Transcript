"""
Enterprise Medical Summarizer

An enterprise-grade integration module that combines advanced NLP, HIPAA compliance,
and professional clinical documentation to produce high-quality medical summaries.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

# Import specialized modules
from .base_summarizer import BaseSummarizer
from .smart_hipaa import SmartHIPAAProcessor
from .clinical_narrative import ClinicalNarrativeGenerator

# Import Gemini integration
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class EnterpriseSummarizer(BaseSummarizer):
    """
    Enterprise-grade medical summarizer that integrates multiple advanced
    capabilities including LLM-powered analysis, HIPAA compliance, and
    professional clinical documentation.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-pro",
        logger: Optional[logging.Logger] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
        enforce_hipaa: bool = True,
        clinical_format: bool = True
    ):
        """
        Initialize the enterprise summarizer.
        
        Args:
            api_key: API key for LLM service
            model_name: Model name to use
            logger: Optional logger
            temperature: Temperature setting for generation (lower = more deterministic)
            max_output_tokens: Maximum tokens to generate
            enforce_hipaa: Whether to enforce HIPAA compliance
            clinical_format: Whether to use clinical formatting
        """
        super().__init__(logger)
        
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.enforce_hipaa = enforce_hipaa
        self.clinical_format = clinical_format
        
        # Initialize specialized processors
        self.hipaa_processor = SmartHIPAAProcessor(logger=logger)
        self.narrative_generator = ClinicalNarrativeGenerator(logger=logger)
        
        # Initialize LLM client
        self.llm_available = False
        if GEMINI_AVAILABLE and self.api_key:
            try:
                self._log(f"Initializing LLM client with model: {model_name}")
                self.client = genai.Client(api_key=self.api_key)
                self.llm_available = True
            except Exception as e:
                self._log(f"Error initializing LLM client: {e}", level="error")
    
    def extract_text_from_segments(self, 
                                   segments: List[Dict[str, Any]],
                                   text_column: str = "transcription",
                                   speaker_column: str = "speaker") -> Dict[str, Any]:
        """
        Extract and format text from transcript segments.
        
        Args:
            segments: List of transcript segments
            text_column: Name of the column containing the transcript text
            speaker_column: Name of the column containing the speaker identifier
            
        Returns:
            Dictionary with extracted text information
        """
        full_text = ""
        speaker_texts = {}
        conversation = []
        speaker_times = {}
        
        for segment in segments:
            if text_column not in segment or speaker_column not in segment:
                continue
                
            speaker = segment[speaker_column]
            text = segment.get(text_column, "")
            
            if not text or not speaker:
                continue
                
            # Add to full text
            full_text += f"{speaker}: {text}\n"
            
            # Add to speaker-specific text
            if speaker not in speaker_texts:
                speaker_texts[speaker] = []
            speaker_texts[speaker].append(text)
            
            # Add to conversation timeline
            conversation.append((speaker, text))
            
            # Track speaking time
            if "start" in segment and "end" in segment:
                duration = segment["end"] - segment["start"]
                if speaker not in speaker_times:
                    speaker_times[speaker] = 0
                speaker_times[speaker] += duration
        
        # Convert speaker text lists to strings
        speaker_texts = {speaker: " ".join(texts) for speaker, texts in speaker_texts.items()}
        
        return {
            "full_text": full_text,
            "speaker_texts": speaker_texts,
            "conversation": conversation,
            "speaker_times": speaker_times
        }
    
    def process_transcript(self,
                          transcript_path: Optional[str] = None,
                          transcript_data: Optional[Dict[str, Any]] = None,
                          output_path: Optional[str] = None,
                          text_column: str = "transcription",
                          speaker_column: str = "speaker",
                          force_process: bool = False,
                          **kwargs) -> Dict[str, Any]:
        """
        Process a transcript and generate a comprehensive, professional medical summary.
        
        Args:
            transcript_path: Path to transcript file (JSON)
            transcript_data: Alternatively, provide transcript data directly
            output_path: Optional path to save results
            text_column: Column name containing the transcription text
            speaker_column: Column name containing the speaker ID
            force_process: If True, process even when content is not telemedical
            **kwargs: Additional options
            
        Returns:
            Dictionary with the summary and extracted information
        """
        # Update options with kwargs
        enforce_hipaa = kwargs.get("enforce_hipaa", self.enforce_hipaa)
        clinical_format = kwargs.get("clinical_format", self.clinical_format)
        custom_prompt = kwargs.get("custom_prompt", None)
        
        # Load transcript
        data = self.load_transcript(transcript_path, transcript_data)
        if not data:
            self._log("No transcript data available", level="error")
            return {"error": "No transcript data available"}
            
        # Extract segments
        segments = data.get("segments", [])
        if not segments:
            self._log("No segments found in transcript", level="error")
            return {"error": "No segments found in transcript"}
            
        self._log(f"Processing transcript with {len(segments)} segments")
        
        # Extract text from segments
        text_data = self.extract_text_from_segments(segments, text_column, speaker_column)
        
        # Identify speaker roles (patient vs provider)
        speaker_roles = self._identify_speaker_roles(text_data["conversation"])
        
        # Extract medical information
        self._log("Extracting medical information")
        extracted_info = self._extract_medical_information(text_data, speaker_roles)
        
        # Generate LLM summary
        self._log("Generating initial summary")
        llm_summary = self._generate_llm_summary(text_data["full_text"], extracted_info, custom_prompt)
        
        # Apply HIPAA compliance if needed
        if enforce_hipaa:
            self._log("Applying HIPAA compliance processing")
            summary_data = {"summary": llm_summary}
            hipaa_compliant_data = self.hipaa_processor.process_document(summary_data)
            llm_summary = hipaa_compliant_data["summary"]
        
        # Generate clinical narrative if requested
        if clinical_format:
            self._log("Generating clinical narrative")
            narrative_data = self.narrative_generator.generate_clinical_narrative(
                {"summary": llm_summary},
                extracted_info,
                hipaa_compliant=enforce_hipaa
            )
            
            # Structure the result
            result = {
                "summary": narrative_data["narrative"],
                "sections": {
                    "subjective": narrative_data["subjective"],
                    "objective": narrative_data["objective"],
                    "assessment": narrative_data["assessment"],
                    "plan": narrative_data["plan"]
                },
                "extracted_info": extracted_info,
                "hipaa_compliant": enforce_hipaa,
                "processed_at": datetime.now().isoformat()
            }
        else:
            # Just return the processed summary
            result = {
                "summary": llm_summary,
                "extracted_info": extracted_info,
                "hipaa_compliant": enforce_hipaa,
                "processed_at": datetime.now().isoformat()
            }
        
        # Save results if output path provided
        if output_path:
            self.save_results(result, output_path)
            
        return result
    
    def _identify_speaker_roles(self, conversation: List[tuple]) -> Dict[str, str]:
        """
        Identify the likely roles of speakers (patient vs provider).
        
        Args:
            conversation: List of (speaker, text) tuples
            
        Returns:
            Dictionary mapping speaker IDs to roles
        """
        # Count questions by speaker
        question_count = {}
        total_words = {}
        medical_terms_count = {}
        
        # Regex for common medical terms
        medical_terms_regex = re.compile(r'\b(?:diagnosis|prescription|symptoms|treatment|medication|dose|dosage|assessment|labs|vitals|follow-up|referral)\b', re.IGNORECASE)
        
        for speaker, text in conversation:
            # Count questions by speaker
            if "?" in text:
                question_count[speaker] = question_count.get(speaker, 0) + 1
                
            # Count total words by speaker
            words = len(text.split())
            total_words[speaker] = total_words.get(speaker, 0) + words
            
            # Count medical terms used by speaker
            medical_terms = len(medical_terms_regex.findall(text))
            medical_terms_count[speaker] = medical_terms_count.get(speaker, 0) + medical_terms
        
        # Combine heuristics to determine roles
        speaker_scores = {}
        
        for speaker in set(s for s, _ in conversation):
            # Higher score = more likely to be provider
            score = 0
            
            # Providers ask more questions
            if speaker in question_count:
                score += question_count[speaker] * 2
                
            # Providers use more medical terminology
            if speaker in medical_terms_count:
                score += medical_terms_count[speaker]
                
            # Providers often speak less overall than patients in medical conversations
            if speaker in total_words and sum(total_words.values()) > 0:
                # Lower percentage of total words = higher score
                word_percentage = total_words[speaker] / sum(total_words.values())
                if word_percentage < 0.4:  # Provider speaks less than 40% of words
                    score += 5
                    
            speaker_scores[speaker] = score
        
        # Assign roles based on scores
        if not speaker_scores:
            return {"provider": None, "patient": None}
            
        # Speaker with highest score is likely the provider
        provider = max(speaker_scores.items(), key=lambda x: x[1])[0]
        
        # Other speakers are patients or family members
        speakers = list(set(s for s, _ in conversation))
        patient = next((s for s in speakers if s != provider), None)
        
        return {
            "provider": provider,
            "patient": patient
        }
    
    def _extract_medical_information(self, text_data: Dict[str, Any], speaker_roles: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract structured medical information from conversation.
        
        Args:
            text_data: Extracted text data
            speaker_roles: Speaker role mapping
            
        Returns:
            Dictionary with extracted medical information
        """
        # Extract from full conversation
        full_text = text_data["full_text"]
        extracted_info = {}
        
        # Add speaker role information
        extracted_info["speaker_roles"] = speaker_roles
        
        # Extract patient name and provider name with basic heuristics
        patient_speaker = speaker_roles.get("patient")
        provider_speaker = speaker_roles.get("provider")
        
        # Set default names
        extracted_info["patient_name"] = patient_speaker or "Patient"
        extracted_info["provider_name"] = provider_speaker or "Provider"
        
        # Look for formal name references (e.g., "Mr. Smith", "Dr. Jones")
        name_pattern = re.compile(r'\b(?:Mr\.|Mrs\.|Ms\.|Miss|Dr\.)\s+([A-Z][a-z]+)', re.IGNORECASE)
        name_matches = name_pattern.findall(full_text)
        
        if name_matches:
            # Assume first name mentioned with title is patient
            extracted_info["patient_name"] = f"{name_matches[0]}"
            
            # If there's a second distinct name, assume it's provider
            if len(name_matches) > 1 and name_matches[1] != name_matches[0]:
                extracted_info["provider_name"] = f"Dr. {name_matches[1]}"
        
        # Extract medications using regex
        med_pattern = re.compile(r'\b(?:taking|prescribed|on|using)\s+([A-Za-z]+(?:\s+\d+\s*(?:mg|mcg|g|mL|tablet|cap|pill))?)', re.IGNORECASE)
        med_matches = med_pattern.findall(full_text)
        extracted_info["medications"] = list(set(med_matches)) if med_matches else []
        
        # Extract conditions using regex
        condition_pattern = re.compile(r'\b(?:diagnosed with|suffering from|has|had|experiencing)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?(?:\s+(?:disease|disorder|condition|syndrome|infection))?)', re.IGNORECASE)
        condition_matches = condition_pattern.findall(full_text)
        extracted_info["conditions"] = list(set(condition_matches)) if condition_matches else []
        
        # Extract vital signs
        vitals = {}
        
        # Blood pressure pattern
        bp_pattern = re.compile(r'\b(?:blood pressure|BP)[:\s]*(\d{2,3})[/\\](\d{2,3})', re.IGNORECASE)
        bp_match = bp_pattern.search(full_text)
        if bp_match:
            systolic, diastolic = bp_match.groups()
            vitals["blood_pressure"] = f"{systolic}/{diastolic}"
            
        # Heart rate pattern
        hr_pattern = re.compile(r'\b(?:heart rate|HR|pulse)[:\s]*(\d{2,3})', re.IGNORECASE)
        hr_match = hr_pattern.search(full_text)
        if hr_match:
            vitals["heart_rate"] = hr_match.group(1)
            
        # Temperature pattern
        temp_pattern = re.compile(r'\b(?:temperature|temp)[:\s]*(\d{2,3}(?:\.\d)?)', re.IGNORECASE)
        temp_match = temp_pattern.search(full_text)
        if temp_match:
            vitals["temperature"] = temp_match.group(1)
            
        # Respiratory rate pattern
        rr_pattern = re.compile(r'\b(?:respiratory rate|RR)[:\s]*(\d{1,2})', re.IGNORECASE)
        rr_match = rr_pattern.search(full_text)
        if rr_match:
            vitals["respiratory_rate"] = rr_match.group(1)
            
        # Oxygen saturation pattern
        o2_pattern = re.compile(r'\b(?:oxygen saturation|O2 sat|SpO2)[:\s]*(\d{1,3})(?:\s*%)?', re.IGNORECASE)
        o2_match = o2_pattern.search(full_text)
        if o2_match:
            vitals["oxygen_saturation"] = o2_match.group(1)
            
        extracted_info["vital_signs"] = vitals
        
        return extracted_info
    
    def _generate_llm_summary(self, 
                             conversation_text: str, 
                             extracted_info: Dict[str, Any],
                             custom_prompt: Optional[str] = None) -> str:
        """
        Generate a summary using LLM.
        
        Args:
            conversation_text: Full conversation text
            extracted_info: Extracted medical information
            custom_prompt: Optional custom prompt
            
        Returns:
            Generated summary
        """
        if not self.llm_available:
            self._log("LLM not available, using template-based summary", level="warning")
            return self._generate_fallback_summary(conversation_text, extracted_info)
            
        try:
            # Prepare the prompt
            prompt = custom_prompt if custom_prompt else self._prepare_default_prompt(conversation_text, extracted_info)
            
            self._log("Sending prompt to LLM")
            
            # Generate content
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            # Extract the text
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'parts'):
                parts_text = ''.join([part.text for part in response.parts if hasattr(part, 'text')])
                if parts_text:
                    return parts_text
                    
            # Fall back to string representation
            return str(response)
            
        except Exception as e:
            self._log(f"Error generating LLM summary: {e}", level="error")
            return self._generate_fallback_summary(conversation_text, extracted_info)
    
    def _prepare_default_prompt(self, conversation_text: str, extracted_info: Dict[str, Any]) -> str:
        """
        Prepare a default prompt for LLM summary generation.
        
        Args:
            conversation_text: Full conversation text
            extracted_info: Extracted medical information
            
        Returns:
            Formatted prompt
        """
        patient_name = extracted_info.get("patient_name", "Patient")
        provider_name = extracted_info.get("provider_name", "Provider")
        
        medications = extracted_info.get("medications", [])
        medications_str = ", ".join(medications) if medications else "Unknown"
        
        conditions = extracted_info.get("conditions", [])
        conditions_str = ", ".join(conditions) if conditions else "Unknown"
        
        prompt = f"""
You are an experienced medical scribe creating a professionally formatted clinical note from a telehealth conversation. Produce a note that follows standard medical documentation practices and formats.

PATIENT INFORMATION:
- Patient: {patient_name}
- Provider: {provider_name}
- Medications: {medications_str}
- Conditions: {conditions_str}

CONVERSATION TRANSCRIPT:
```
{conversation_text}
```

DOCUMENTATION REQUIREMENTS:
1. Format the note in SOAP format (Subjective, Objective, Assessment, Plan)
2. Write in professional medical style using appropriate clinical terminology
3. Focus only on medically relevant information
4. Include:
   - Chief complaint and history of present illness
   - Clear documentation of objective findings
   - Assessment of the patient's condition
   - Detailed treatment plan and follow-up recommendations
5. Use specific values for vital signs and measurements when available
6. Document medications, dosages, and instructions with precision
7. Include only information explicitly mentioned in the conversation
8. Format as a continuous narrative (2-3 paragraphs per section)
9. Use third-person clinical perspective throughout
10. Maintain professional, clinical tone

Your clinical note should read as if written by a healthcare professional with clear organization and appropriate usage of medical terminology.
"""
        return prompt
    
    def _generate_fallback_summary(self, conversation_text: str, extracted_info: Dict[str, Any]) -> str:
        """
        Generate a fallback summary when LLM is not available.
        
        Args:
            conversation_text: Full conversation text
            extracted_info: Extracted medical information
            
        Returns:
            Fallback summary
        """
        patient_name = extracted_info.get("patient_name", "Patient")
        provider_name = extracted_info.get("provider_name", "Provider")
        
        medications = extracted_info.get("medications", [])
        medications_str = ", ".join(medications) if medications else "not documented"
        
        conditions = extracted_info.get("conditions", [])
        conditions_str = ", ".join(conditions) if conditions else "not clearly documented"
        
        vitals = extracted_info.get("vital_signs", {})
        vitals_list = []
        for key, value in vitals.items():
            vitals_list.append(f"{key}: {value}")
        vitals_str = ", ".join(vitals_list) if vitals_list else "not documented during this encounter"
        
        # Create a basic summary
        summary = f"""
SUBJECTIVE:
Patient {patient_name} presented for telehealth appointment with {provider_name}. Chief complaint and reason for visit based on available information appears to be {conditions_str}. The patient's current medications are {medications_str}.

OBJECTIVE:
Vital signs were {vitals_str}. Physical examination limited by telehealth format.

ASSESSMENT:
Assessment based on presentation suggests {conditions_str}. Further evaluation may be needed.

PLAN:
Treatment plan to be determined. Follow-up recommended to monitor progress and response to treatment.
"""
        return summary.strip()
