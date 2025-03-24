"""
Main medical transcript processor that coordinates all extraction and generation components.
"""
import os
import json
import torch
from typing import Dict, List, Any, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)

from .speaker_identifier import SpeakerIdentifier
from .vital_sign_extractor import VitalSignExtractor
from .medication_extractor import MedicationExtractor
from .condition_symptom_extractor import ConditionSymptomExtractor
from .note_generator import NoteGenerator
from .base_extractor import BaseExtractor

class MedicalTranscriptProcessor(BaseExtractor):
    """
    Main coordinator class that processes medical transcripts by:
    1. Identifying speakers and their roles
    2. Extracting medical entities using specialized extractors
    3. Generating formatted medical notes
    
    This modular approach allows for easy maintenance and extension of functionality.
    """
    
    def __init__(
        self,
        ner_model: str = "emilyalsentzer/Bio_ClinicalBERT",
        device: str = None,
        compute_type: str = "float16",
        cache_dir: str = None,
        confidence_threshold: float = 0.65,
        logger = None
    ):
        """
        Initialize the medical transcript processor.
        
        Args:
            ner_model: Model name for NER
            device: Computation device ('cuda', 'cpu', or None for auto-detect)
            compute_type: Computation precision ("float16", "float32", "int8")
            cache_dir: Directory to cache downloaded models
            confidence_threshold: Minimum confidence for entity extraction
            logger: Optional logger for messages
        """
        super().__init__(logger)
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type
        
        # Initialize NER model if available
        self.ner_model = None
        self.ner_tokenizer = None
        try:
            self._log("Loading NER model...")
            self.ner_tokenizer = AutoTokenizer.from_pretrained(ner_model, cache_dir=cache_dir)
            self.ner_model = pipeline(
                "ner",
                model=ner_model,
                tokenizer=self.ner_tokenizer,
                device=0 if self.device == "cuda" else -1,
                aggregation_strategy="simple"
            )
            self._log("NER model loaded successfully")
        except Exception as e:
            self._log(f"Error loading NER model: {e}. Will proceed without NER.", level="warning")
        
        # Initialize specialized extractors
        self.speaker_identifier = SpeakerIdentifier(logger)
        self.vital_sign_extractor = VitalSignExtractor(logger)
        self.medication_extractor = MedicationExtractor(logger)
        self.condition_symptom_extractor = ConditionSymptomExtractor(logger)
        self.note_generator = NoteGenerator(logger)
        
        self._log(f"Initialized Medical Transcript Processor (device: {self.device})")
    
    def process_transcript(
        self,
        transcript_path: str = None,
        transcript_data: Dict = None,
        output_path: str = None,
        text_column: str = "transcription",
        speaker_column: str = "speaker",
        template_path: str = None
    ) -> Dict[str, Any]:
        """
        Process a conversation transcript to extract medical information and generate notes.
        
        Args:
            transcript_path: Path to transcript file (JSON)
            transcript_data: Alternatively, provide transcript data directly
            output_path: Optional path to save results
            text_column: Column name containing the transcription text
            speaker_column: Column name containing the speaker ID
            template_path: Optional path to a telehealth note template
            
        Returns:
            Dictionary with extracted information and generated notes
        """
        # Load transcript
        if transcript_path and not transcript_data:
            try:
                with open(transcript_path, 'r') as f:
                    transcript_data = json.load(f)
            except Exception as e:
                self._log(f"Error loading transcript: {e}", level="error")
                return {}
        
        if not transcript_data:
            self._log("No transcript data provided", level="error")
            return {}
        
        # Extract segments
        segments = transcript_data.get("segments", [])
        if not segments:
            self._log("No segments found in transcript", level="error")
            return {}
        
        self._log(f"Processing transcript with {len(segments)} segments")
        
        # Step 1: Identify speakers
        speaker_info = self.speaker_identifier.extract(segments, text_column, speaker_column)
        
        # Extract combined text for analysis
        all_text = self._combine_text(segments, text_column)
        
        # Step 2: Perform NER if model is available
        entities = []
        if self.ner_model:
            entities = self._extract_entities(all_text)
        
        # Step 3: Extract vital signs
        vital_signs = self.vital_sign_extractor.extract(all_text)
        
        # Step 4: Extract medications
        medication_info = self.medication_extractor.extract(all_text, entities)
        
        # Step 5: Extract conditions and symptoms
        condition_symptom_info = self.condition_symptom_extractor.extract(all_text, entities)
        
        # Step 6: Extract lifestyle information
        lifestyle_info = self._extract_lifestyle_info(all_text)
        
        # Step 7: Extract preventive care information
        preventive_care = self._extract_preventive_care(all_text)
        
        # Step 8: Extract follow-up information
        follow_up = self._extract_follow_up(all_text)
        
        # Combine all extracted information
        results = {
            "patient_info": {
                "patient_name": speaker_info["names"]["patient_name"],
                "care_manager_name": speaker_info["names"]["care_manager_name"]
            },
            "speakers": speaker_info["speakers"],
            "health_status": {
                **condition_symptom_info,
                "vital_signs": vital_signs
            },
            "medications": medication_info,
            "lifestyle": lifestyle_info,
            "preventive_care": preventive_care,
            "plan": {
                "follow_up": follow_up
            },
            "raw_entities": [{"word": e["word"], "type": e.get("entity"), "score": e.get("score")}
                           for e in entities] if entities else []
        }
        
        # Generate SOAP note
        soap_note = self.note_generator.generate_soap_note(results)
        results["soap_note"] = soap_note
        
        # Generate telehealth note
        telehealth_note = self.note_generator.generate_telehealth_note(results, template_path)
        results["telehealth_note"] = telehealth_note
        
        # Generate narrative summary
        results["narrative_summary"] = self._generate_summary(results)
        
        # Save results if output path provided
        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save main results
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                self._log(f"Saved results to {output_path}")
                
                # Save text outputs
                summary_path = output_path.replace('.json', '_summary.txt')
                with open(summary_path, 'w') as f:
                    f.write(results["narrative_summary"])
                    f.write("\n\n--- SOAP NOTE ---\n\n")
                    for section, content in soap_note.items():
                        f.write(f"{section}:\n{content}\n\n")
                
                telehealth_path = output_path.replace('.json', '_telehealth_note.txt')
                with open(telehealth_path, 'w') as f:
                    f.write(telehealth_note)
                
                self._log(f"Saved text outputs to {os.path.dirname(output_path)}")
                
            except Exception as e:
                self._log(f"Error saving results: {e}", level="error")
        
        return results
    
    def _combine_text(self, segments: List[Dict], text_column: str) -> str:
        """Combine all text segments for analysis."""
        return " ".join([segment.get(text_column, "") for segment in segments if text_column in segment])
    
    def _extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities using the NER model."""
        self._log("Extracting named entities")
        
        try:
            # Handle long texts by chunking
            max_len = 512  # Maximum length for transformer model
            chunks = []
            
            # Simple chunking by length
            for i in range(0, len(text), max_len):
                chunks.append(text[i:i+max_len])
            
            # Process each chunk
            all_entities = []
            for chunk in chunks:
                chunk_entities = self.ner_model(chunk)
                
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
            
            self._log(f"Extracted {len(entities)} entities")
            return entities
            
        except Exception as e:
            self._log(f"Error in entity extraction: {e}", level="error")
            return []
    
    def _extract_lifestyle_info(self, text: str) -> Dict[str, Any]:
        """Extract lifestyle information (exercise, diet, smoking, alcohol, etc.)."""
        self._log("Extracting lifestyle information")
        
        # This is a placeholder - in a real implementation, this would be a specialized extractor
        # For now, we'll return a simple structure with empty values
        return {
            "exercise": {},
            "diet": {},
            "smoking": {},
            "alcohol": {},
            "social_support": {}
        }
    
    def _extract_preventive_care(self, text: str) -> Dict[str, str]:
        """Extract preventive care information (colonoscopy, vaccines, etc.)."""
        self._log("Extracting preventive care information")
        
        preventive_care = {}
        text_lower = text.lower()
        
        # Check for colonoscopy
        if "colonoscopy" in text_lower:
            if "last year" in text_lower and "colonoscopy" in text_lower:
                preventive_care["colonoscopy"] = "Completed last year"
            elif re.search(r"(\d+)\s+(?:year|month)s?\s+ago", text_lower):
                time_match = re.search(r"(\d+)\s+(?:year|month)s?\s+ago", text_lower)
                if time_match:
                    time_value = time_match.group(1)
                    time_unit = "year" if "year" in time_match.group(0) else "month"
                    preventive_care["colonoscopy"] = f"Completed {time_value} {time_unit}{'s' if int(time_value) > 1 else ''} ago"
        
        # Check for vaccines
        if "vaccine" in text_lower or "vaccination" in text_lower:
            if "up to date" in text_lower:
                preventive_care["vaccines"] = "Up to date"
        
        # Check for flu shot
        if "flu shot" in text_lower or "flu vaccine" in text_lower:
            preventive_care["flu_shot"] = "Received"
        
        # Check for pneumonia vaccine
        if "pneumonia vaccine" in text_lower:
            preventive_care["pneumonia_vaccine"] = "Received"
        
        return preventive_care
    
    def _extract_follow_up(self, text: str) -> Dict[str, Any]:
        """Extract follow-up information."""
        self._log("Extracting follow-up information")
        
        follow_up = {
            "timeframe": None,
            "type": None,
            "with_who": None,
            "complete_text": None
        }
        
        # Look for follow-up mentions
        text_lower = text.lower()
        
        # Check for timeframe mentions
        timeframe_match = re.search(r"(?:in|after|next)\s+(\d+)\s+(days|weeks|months)", text_lower)
        if timeframe_match:
            timeframe_value = timeframe_match.group(1)
            timeframe_unit = timeframe_match.group(2)
            follow_up["timeframe"] = f"{timeframe_value} {timeframe_unit}"
            
            # Get surrounding context
            pos = timeframe_match.start()
            context_start = max(0, pos - 30)
            context_end = min(len(text_lower), pos + len(timeframe_match.group(0)) + 30)
            follow_up["complete_text"] = text_lower[context_start:context_end]
        
        # Check for type of follow-up
        if "call" in text_lower and follow_up["timeframe"]:
            follow_up["type"] = "phone"
        elif ("appointment" in text_lower or "visit" in text_lower) and follow_up["timeframe"]:
            follow_up["type"] = "in-person"
        
        # Check for provider
        if follow_up["timeframe"]:
            provider_match = re.search(r"(?:with|see)\s+(dr\.?\s+\w+|doctor\s+\w+)", text_lower)
            if provider_match:
                follow_up["with_who"] = provider_match.group(1)
        
        return follow_up
    
    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate a narrative summary of the medical conversation.
        
        Args:
            results: Dictionary with extracted medical information
            
        Returns:
            Narrative summary as a string
        """
        summary_parts = []
        
        # Patient info
        patient_name = results["patient_info"]["patient_name"]
        care_manager_name = results["patient_info"]["care_manager_name"]
        summary_parts.append(f"{patient_name} had a routine check-in call with {care_manager_name}.")
        
        # Health status
        health_status = results["health_status"]
        if health_status["has_symptoms"]:
            summary_parts.append(f"Patient reported symptoms: {health_status['symptom_text']}.")
        else:
            summary_parts.append("Patient reported no unusual symptoms.")
        
        # Conditions
        if health_status.get("conditions"):
            condition_texts = []
            for condition in health_status["conditions"]:
                condition_text = condition["name"]
                if condition.get("severity"):
                    condition_text += f" ({condition['severity']})"
                condition_texts.append(condition_text)
            summary_parts.append(f"Patient has the following conditions: {', '.join(condition_texts)}.")
        
        # Vital signs
        vitals = health_status["vital_signs"]
        vital_parts = []
        
        if vitals["blood_pressure"] and len(vitals["blood_pressure"]) > 0:
            bp = vitals["blood_pressure"][0]["full"]
            vital_parts.append(f"blood pressure of {bp}")
        
        if vitals["glucose"] and len(vitals["glucose"]) > 0:
            glucose = vitals["glucose"][0]["value"]
            vital_parts.append(f"glucose reading of {glucose}")
        
        if vitals.get("weight_change") and isinstance(vitals["weight_change"], dict):
            weight_change = vitals["weight_change"]
            direction = weight_change.get("direction", "")
            value = weight_change.get("value", "")
            if direction and value:
                vital_parts.append(f"weight {direction} of {value} pounds")
        
        if vital_parts:
            summary_parts.append(f"Patient reports {', '.join(vital_parts)}.")
        
        # Medications
        medications = results["medications"].get("medications", [])
        if medications:
            med_names = [m["name"] for m in medications]
            summary_parts.append(f"Current medications include: {', '.join(med_names)}.")
            
            adherence = results["medications"].get("adherence", "")
            if adherence:
                summary_parts.append(f"Medication adherence: {adherence}.")
            
            side_effects = results["medications"].get("side_effects", "")
            if side_effects and side_effects != "No side effects reported":
                summary_parts.append(f"Side effects: {side_effects}.")
        
        # Follow-up plan
        follow_up = results["plan"]["follow_up"]
        if follow_up:
            if isinstance(follow_up, dict) and (follow_up.get("timeframe") or follow_up.get("complete_text")):
                if follow_up.get("timeframe") and follow_up.get("type"):
                    follow_up_text = f"{follow_up.get('type', 'Follow-up')} in {follow_up['timeframe']}"
                    if follow_up.get("with_who"):
                        follow_up_text += f" with {follow_up['with_who']}"
                    summary_parts.append(f"Follow-up plan: {follow_up_text}.")
                elif follow_up.get("complete_text"):
                    summary_parts.append(f"Follow-up plan: {follow_up['complete_text']}.")
            else:
                summary_parts.append(f"Follow-up plan: {follow_up}.")
        
        return " ".join(summary_parts)
