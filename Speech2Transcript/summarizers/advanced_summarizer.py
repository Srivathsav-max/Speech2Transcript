import os
import json
import re
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

# Import base summarizer
from .base_summarizer import BaseSummarizer

# Import extractors
from .extractors import (
    SpeakerExtractor,
    PatientInfoExtractor,
    MedicalConditionExtractor,
    MedicationExtractor,
    VitalSignsExtractor,
    LifestyleExtractor,
    FollowUpExtractor,
    PreventiveCareExtractor
)

# Optional imports for transformer models
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        AutoModelForSequenceClassification,
        BertForQuestionAnswering,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class AdvancedMedicalSummarizer(BaseSummarizer):
    """
    Enterprise-grade advanced medical transcript summarizer.
    
    Features:
    - Modular architecture with specialized extractors
    - Optional transformer-based NLP capabilities
    - Configurable pipeline with dependency injection
    - Performance optimization with caching
    """
    
    def __init__(
        self,
        model_config: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize advanced medical summarizer.
        
        Args:
            model_config: Dictionary with model identifiers for different NLP tasks
            device: Device to run models on ('cuda', 'cpu', or None for auto-detect)
            cache_dir: Directory for caching models and results
            logger: Optional logger for messages
        """
        super().__init__(logger)
        
        # Initialize configuration
        self.cache_dir = cache_dir
        
        # Setup extractors
        self._setup_extractors()
        
        # Setup transformer models - try to use them by default when available
        try:
            self._setup_transformer_models(model_config, device)
            self.use_transformers = True
        except Exception as e:
            self._log(f"Transformer models not available: {e}", level="warning")
            self.use_transformers = False
            self._log("Using rule-based extractors only")
        
        self._log("Medical Summarizer initialized")
    
    def _setup_extractors(self):
        """Initialize extractors for different information types."""
        self.speaker_extractor = SpeakerExtractor(self.logger)
        self.patient_info_extractor = PatientInfoExtractor(self.logger)
        self.condition_extractor = MedicalConditionExtractor(self.logger)
        self.medication_extractor = MedicationExtractor(self.logger)
        self.vitals_extractor = VitalSignsExtractor(self.logger)
        self.lifestyle_extractor = LifestyleExtractor(self.logger)
        self.followup_extractor = FollowUpExtractor(self.logger)
        self.preventive_care_extractor = PreventiveCareExtractor(self.logger)
    
    def _setup_transformer_models(self, model_config: Optional[Dict[str, str]], device: Optional[str]):
        """
        Initialize transformer models for enhanced NLP.
        
        Args:
            model_config: Dictionary with model identifiers for different tasks
            device: Device to run models on ('cuda', 'cpu', or None for auto-detect)
        """
        if not TRANSFORMERS_AVAILABLE:
            self._log("Transformer models not available, using fallback extractors", level="warning")
            return
        
        try:
            # Default model configuration
            self.model_config = model_config or {
                "ner_model": "emilyalsentzer/Bio_ClinicalBERT",
                "qa_model": "distilbert-base-cased-distilled-squad",
                "sentiment_model": "bhadresh-savani/distilbert-base-uncased-emotion",
                "summarization_model": "philschmid/bart-large-cnn-samsum"
            }
            
            # Determine device
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            
            self._log(f"Using device: {self.device}")
            
            # Initialize NER model for medical entity extraction
            self._log(f"Loading NER model: {self.model_config['ner_model']}")
            self.ner_pipeline = pipeline(
                "token-classification",
                model=self.model_config["ner_model"],
                aggregation_strategy="simple",
                device=0 if self.device == "cuda" else -1
            )
            
            # Initialize QA model for information extraction
            self._log(f"Loading QA model: {self.model_config['qa_model']}")
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model_config["qa_model"],
                device=0 if self.device == "cuda" else -1
            )
            
            # Initialize sentiment model for patient attitude analysis
            self._log(f"Loading sentiment model: {self.model_config['sentiment_model']}")
            self.sentiment_pipeline = pipeline(
                "text-classification",
                model=self.model_config["sentiment_model"],
                return_all_scores=True,
                device=0 if self.device == "cuda" else -1
            )
            
            # Initialize summarization model for dialogue summarization
            self._log(f"Loading summarization model: {self.model_config['summarization_model']}")
            self.summarization_pipeline = pipeline(
                "summarization",
                model=self.model_config["summarization_model"],
                device=0 if self.device == "cuda" else -1
            )
            
            self._log("All transformer models loaded successfully")
            
        except Exception as e:
            self._log(f"Error loading transformer models: {e}", level="error")
            self.use_transformers = False
            self._log("Falling back to rule-based extractors")
    
    def extract_text_from_segments(self, segments: List[Dict[str, Any]], 
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
                          speaker_column: str = "speaker") -> Dict[str, Any]:
        """
        Process a transcript and generate a detailed medical summary.
        
        Args:
            transcript_path: Path to transcript file (JSON)
            transcript_data: Alternatively, provide transcript data directly
            output_path: Optional path to save results
            text_column: Column name containing the transcription text
            speaker_column: Column name containing the speaker ID
            
        Returns:
            Dictionary with the extracted data and generated summary
        """
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
        
        # Extract information using specialized extractors
        speaker_data = self.speaker_extractor.extract(segments)
        
        # Combine data for extraction context
        extraction_context = {**text_data, **speaker_data}
        
        patient_info = self.patient_info_extractor.extract(extraction_context)
        conditions = self.condition_extractor.extract(extraction_context)
        medications = self.medication_extractor.extract(extraction_context)
        vitals = self.vitals_extractor.extract(extraction_context)
        lifestyle = self.lifestyle_extractor.extract(extraction_context)
        followup = self.followup_extractor.extract(extraction_context)
        preventive_care = self.preventive_care_extractor.extract(extraction_context)
        
        # Enhanced extraction with transformer models if enabled
        enhanced_data = {}
        if self.use_transformers:
            try:
                enhanced_data = self._enhance_with_transformers(extraction_context)
                self._log("Enhanced extraction with transformer models")
            except Exception as e:
                self._log(f"Error in transformer-based enhancement: {e}", level="error")
        
        # Compile extracted data
        extracted_data = {
            "speakers": speaker_data["speakers"],
            "patient_info": patient_info,
            "conditions": conditions,
            "medications": medications,
            "vital_signs": vitals,
            "lifestyle": lifestyle,
            "follow_up": followup,
            "preventive_care": preventive_care
        }
        
        # Merge with enhanced data if available
        if enhanced_data:
            # Intelligently merge data, prioritizing higher confidence information
            for key, value in enhanced_data.items():
                if key in extracted_data and isinstance(value, dict) and isinstance(extracted_data[key], dict):
                    # Merge dictionaries, keeping original keys that aren't in enhanced data
                    extracted_data[key] = {**extracted_data[key], **value}
                elif key == "medical_entities" or key == "sentiment" or key == "dialogue_summary":
                    # Add new categories
                    extracted_data[key] = value
        
        # Generate health assessment
        health_assessment = self._assess_health_status(extracted_data)
        extracted_data["health_assessment"] = health_assessment
        
        # Generate detailed summary
        detailed_summary = self._generate_detailed_summary(extracted_data)
        
        # Create result object
        result = {
            "extracted_data": extracted_data,
            "detailed_summary": detailed_summary
        }
        
        # Save results if output path provided
        if output_path:
            self.save_results(result, output_path)
        
        return result
    
    def _enhance_with_transformers(self, extraction_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance extraction with transformer models.
        
        Args:
            extraction_context: Extracted conversation data
            
        Returns:
            Dictionary with enhanced extracted information
        """
        if not self.use_transformers:
            return {}
        
        enhanced_data = {}
        
        # Get conversation and text
        conversation = extraction_context["conversation"]
        full_text = extraction_context["full_text"]
        speakers = extraction_context["speakers"]
        
        # Extract medical entities with NER
        enhanced_data["medical_entities"] = self._extract_medical_entities(full_text)
        
        # Extract medication adherence with sentiment analysis
        enhanced_data["medication_adherence"] = self._extract_medication_adherence(conversation, speakers)
        
        # Generate dialogue summary
        enhanced_data["dialogue_summary"] = self._generate_dialogue_summary(conversation, speakers)
        
        return enhanced_data
    
    def _extract_medical_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract medical entities using NER model.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with categorized medical entities
        """
        medical_entities = {
            "conditions": [],
            "medications": [],
            "symptoms": [],
            "procedures": [],
            "lab_tests": []
        }
        
        # Split text into chunks to avoid context length issues
        chunk_size = 500
        overlap = 50
        text_chunks = []
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            text_chunks.append(chunk)
        
        # Process each chunk with NER
        try:
            all_entities = []
            for chunk in text_chunks:
                entities = self.ner_pipeline(chunk)
                all_entities.extend(entities)
            
            # Categorize entities
            for entity in all_entities:
                # Skip low confidence detections
                if entity.get('score', 0) < 0.7:
                    continue
                    
                entity_text = entity['word']
                entity_type = entity.get('entity_group', '')
                
                # Map entity types to our categories
                if entity_type in ['DISEASE', 'PROBLEM', 'DIS']:
                    # Check if entity already exists
                    if not any(c['name'].lower() == entity_text.lower() for c in medical_entities['conditions']):
                        medical_entities['conditions'].append({
                            'name': entity_text,
                            'confidence': entity.get('score', 0)
                        })
                
                elif entity_type in ['CHEMICAL', 'DRUG', 'MEDICATION']:
                    if not any(m['name'].lower() == entity_text.lower() for m in medical_entities['medications']):
                        medical_entities['medications'].append({
                            'name': entity_text,
                            'confidence': entity.get('score', 0)
                        })
                
                elif entity_type in ['SYMPTOM', 'SIGN']:
                    if not any(s['name'].lower() == entity_text.lower() for s in medical_entities['symptoms']):
                        medical_entities['symptoms'].append({
                            'name': entity_text,
                            'confidence': entity.get('score', 0)
                        })
                
                elif entity_type in ['PROCEDURE', 'TREATMENT']:
                    if not any(p['name'].lower() == entity_text.lower() for p in medical_entities['procedures']):
                        medical_entities['procedures'].append({
                            'name': entity_text,
                            'confidence': entity.get('score', 0)
                        })
                
                elif entity_type in ['TEST', 'LAB']:
                    if not any(l['name'].lower() == entity_text.lower() for l in medical_entities['lab_tests']):
                        medical_entities['lab_tests'].append({
                            'name': entity_text,
                            'confidence': entity.get('score', 0)
                        })
        
        except Exception as e:
            self._log(f"Error in NER for medical entities: {e}", level="warning")
        
        return medical_entities
    
    def _extract_medication_adherence(self, conversation: List[Tuple[str, str]], 
                                     speakers: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract medication adherence with sentiment analysis.
        
        Args:
            conversation: List of (speaker, text) tuples
            speakers: Speaker role mapping
            
        Returns:
            Dictionary with adherence information
        """
        care_manager = speakers["care_manager"]
        patient = speakers["patient"]
        
        adherence_info = {
            "status": "Unknown",
            "confidence": 0.0,
            "details": [],
            "sentiment": None
        }
        
        # Combine all text about medications and adherence
        adherence_texts = []
        
        # Look for medication adherence questions and answers
        for i, (speaker, text) in enumerate(conversation):
            if speaker == care_manager and re.search(r"(tak(ing|e)|following).+(medication|medicine|prescription)", text.lower()):
                # Look for the patient's response
                for j in range(i+1, min(i+4, len(conversation))):
                    if j < len(conversation) and conversation[j][0] == patient:
                        response = conversation[j][1]
                        adherence_texts.append(response)
                        break
        
        # If no adherence texts found, return default
        if not adherence_texts:
            return adherence_info
        
        # Analyze adherence text with sentiment analysis
        try:
            combined_text = " ".join(adherence_texts)
            
            # Use QA to get a direct answer
            qa_result = self.qa_pipeline(
                question="Is the patient taking medications as prescribed?",
                context=combined_text
            )
            
            # Use sentiment analysis as a supplement
            sentiment_result = self.sentiment_pipeline(combined_text)
            sentiment_scores = {item['label']: item['score'] for item in sentiment_result[0]}
            
            # Determine adherence status
            if re.search(r"yes|yeah|taking|regularly", combined_text.lower()):
                adherence_info["status"] = "Good"
                adherence_info["confidence"] = max(0.8, qa_result['score'])
            elif re.search(r"not|miss|forgot|sometimes|try", combined_text.lower()):
                adherence_info["status"] = "Variable"
                adherence_info["confidence"] = max(0.7, qa_result['score'])
            elif re.search(r"no|stopped|don't", combined_text.lower()):
                adherence_info["status"] = "Poor"
                adherence_info["confidence"] = max(0.8, qa_result['score'])
            else:
                # Use sentiment to determine if unclear
                if sentiment_scores.get('joy', 0) > 0.5 or sentiment_scores.get('surprise', 0) > 0.7:
                    adherence_info["status"] = "Good"
                    adherence_info["confidence"] = 0.6
                elif sentiment_scores.get('sadness', 0) > 0.5 or sentiment_scores.get('anger', 0) > 0.3:
                    adherence_info["status"] = "Poor"
                    adherence_info["confidence"] = 0.6
                else:
                    adherence_info["status"] = "Unclear"
                    adherence_info["confidence"] = 0.5
            
            # Store sentiment info
            top_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])
            adherence_info["sentiment"] = {
                "primary": top_sentiment[0],
                "score": top_sentiment[1],
                "details": sentiment_scores
            }
            
            # Store adherence details
            adherence_info["details"] = [{
                "text": text,
                "sentiment": self.sentiment_pipeline(text)[0][0]['label'],
                "confidence": qa_result['score'] if i == 0 else 0.5
            } for i, text in enumerate(adherence_texts)]
            
        except Exception as e:
            self._log(f"Error analyzing medication adherence: {e}", level="warning")
        
        return adherence_info
    
    def _generate_dialogue_summary(self, conversation: List[Tuple[str, str]], 
                                  speakers: Dict[str, str]) -> str:
        """
        Generate a conversational summary using summarization model.
        
        Args:
            conversation: List of (speaker, text) tuples
            speakers: Speaker role mapping
            
        Returns:
            Summary text
        """
        care_manager_id = speakers["care_manager"]
        patient_id = speakers["patient"]
        
        # Format the conversation for the summarization model
        formatted_conversation = []
        
        for speaker, text in conversation:
            speaker_label = "Care Manager" if speaker == care_manager_id else "Patient"
            formatted_conversation.append(f"{speaker_label}: {text}")
        
        # Join the formatted conversation
        conversation_text = "\n".join(formatted_conversation)
        
        # Generate the summary
        try:
            # Split into chunks if needed
            max_length = 1024
            summary = ""
            
            if len(conversation_text) > max_length:
                # Process in chunks with overlap
                chunks = []
                chunk_size = max_length - 100
                overlap = 200
                
                for i in range(0, len(conversation_text), chunk_size - overlap):
                    chunk = conversation_text[i:i + chunk_size]
                    chunks.append(chunk)
                
                # Summarize each chunk
                chunk_summaries = []
                
                for i, chunk in enumerate(chunks):
                    result = self.summarization_pipeline(
                        chunk,
                        max_length=150 if i < len(chunks) - 1 else 200,
                        min_length=50,
                        do_sample=False
                    )
                    chunk_summaries.append(result[0]['summary_text'])
                
                # Combine chunk summaries
                summary = " ".join(chunk_summaries)
            else:
                # Summarize the entire conversation at once
                result = self.summarization_pipeline(
                    conversation_text,
                    max_length=250,
                    min_length=100,
                    do_sample=False
                )
                summary = result[0]['summary_text']
            
            # Post-process the summary into a proper paragraph format
            # First normalize whitespace
            summary = re.sub(r'\s+', ' ', summary).strip()
            
            # Add proper paragraph structure
            # Split by sentence endings and rebuild with proper spacing
            summary_paragraphs = []
            sentences = re.split(r'(?<=[.!?])\s+', summary)
            
            current_paragraph = []
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                current_paragraph.append(sentence)
                
                # Create a new paragraph after 2-3 sentences or if it's a topic shift
                if len(current_paragraph) >= 3 or re.search(r'(however|moreover|furthermore|in addition|additionally)', sentence, re.IGNORECASE):
                    summary_paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = []
            
            # Add any remaining sentences as a paragraph
            if current_paragraph:
                summary_paragraphs.append(" ".join(current_paragraph))
                
            # Join paragraphs with double newlines
            if summary_paragraphs:
                summary = "\n\n".join(summary_paragraphs)
            
            return summary
        
        except Exception as e:
            self._log(f"Error generating dialogue summary: {e}", level="warning")
            
            # Fall back to template-based summary if model fails
            return self._generate_template_summary(conversation, speakers)
    
    def _generate_template_summary(self, conversation: List[Tuple[str, str]], 
                                  speakers: Dict[str, str]) -> str:
        """Generate a template-based summary if the model-based approach fails."""
        care_manager_id = speakers["care_manager"]
        
        # Extract potential topics
        topics = []
        for speaker, text in conversation:
            if speaker == care_manager_id:
                if re.search(r"(health|feeling|symptoms)", text.lower()):
                    topics.append("health status")
                if re.search(r"(medication|medicine|pill)", text.lower()):
                    topics.append("medications")
                if re.search(r"(diet|eating|food)", text.lower()):
                    topics.append("diet")
                if re.search(r"(exercise|activity|walk)", text.lower()):
                    topics.append("physical activity")
                if re.search(r"(follow|appointment|visit)", text.lower()):
                    topics.append("follow-up plan")
                if re.search(r"(vaccin|shot|immuniz)", text.lower()):
                    topics.append("vaccinations")
                if re.search(r"(screen|test|colonoscopy)", text.lower()):
                    topics.append("screenings")
        
        # Remove duplicates while preserving order
        topics = list(dict.fromkeys(topics))
        
        # Generate first paragraph about the call purpose
        paragraphs = ["The care manager called to check on the patient's health status."]
        
        # Generate second paragraph about discussed topics
        if topics:
            topic_text = ", ".join(topics[:-1])
            if topic_text:
                topic_text += f" and {topics[-1]}"
            else:
                topic_text = topics[-1]
            
            paragraphs.append(f"During the conversation, they discussed {topic_text} in detail. The care manager asked several questions to assess the patient's current condition and medication adherence.")
        
        # Generate third paragraph about follow-up and next steps
        follow_up_paragraph = "The patient provided updates on their current health situation and daily routine."
        
        # Look for follow-up
        for speaker, text in conversation[-5:]:
            if speaker == care_manager_id and re.search(r"(call|follow|contact|appointment)", text.lower()):
                follow_up_paragraph += " A follow-up was scheduled to continue monitoring the patient's progress."
                break
        
        paragraphs.append(follow_up_paragraph)
        
        # Join paragraphs with double newlines for proper paragraph formatting
        return "\n\n".join(paragraphs)
    
    def _assess_health_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess overall health status based on extracted data.
        
        Args:
            data: Dictionary with extracted medical information
            
        Returns:
            Health assessment
        """
        assessment = {
            "status": "stable",
            "confidence": 0.7,
            "concerns": [],
            "positives": []
        }
        
        # Check for concerning vital signs
        vitals = data.get("vital_signs", {})
        if vitals.get("blood_pressure") and len(vitals["blood_pressure"]) > 0:
            bp = vitals["blood_pressure"][0]
            if bp["systolic"] > 160 or bp["diastolic"] > 100:
                assessment["concerns"].append({
                    "type": "vital",
                    "detail": f"Elevated blood pressure: {bp['full']}",
                    "severity": "moderate"
                })
                assessment["status"] = "concerning"
        
        if vitals.get("glucose") and len(vitals["glucose"]) > 0:
            glucose = vitals["glucose"][0]["value"]
            if glucose > 200:
                assessment["concerns"].append({
                    "type": "vital",
                    "detail": f"Elevated blood glucose: {glucose} mg/dL",
                    "severity": "moderate"
                })
                assessment["status"] = "concerning"
        
        # Check medication adherence
        if data.get("medications", {}).get("adherence") == "Poor":
            assessment["concerns"].append({
                "type": "adherence",
                "detail": "Poor medication adherence",
                "severity": "high"
            })
            assessment["status"] = "concerning"
        
        # Check for positive health factors
        lifestyle = data.get("lifestyle", {})
        if lifestyle.get("exercise", {}).get("type") != "Unknown":
            assessment["positives"].append({
                "type": "lifestyle",
                "detail": "Regular physical activity"
            })
        
        if lifestyle.get("diet", {}).get("habits") == "Healthy":
            assessment["positives"].append({
                "type": "lifestyle",
                "detail": "Healthy diet habits"
            })
        
        if lifestyle.get("smoking") == "Non-smoker":
            assessment["positives"].append({
                "type": "lifestyle",
                "detail": "Non-smoker"
            })
        
        if vitals.get("weight_change") and vitals["weight_change"].get("direction") == "lost":
            assessment["positives"].append({
                "type": "progress",
                "detail": f"Weight loss of {vitals['weight_change']['value']} {vitals['weight_change']['unit']}"
            })
        
        # Set confidence based on available data
        data_completeness = sum([
            1 if vitals.get("blood_pressure") else 0,
            1 if vitals.get("glucose") else 0,
            1 if data.get("medications", {}).get("adherence") != "Unknown" else 0,
            1 if lifestyle.get("exercise", {}).get("type") != "Unknown" else 0,
            1 if lifestyle.get("diet", {}).get("habits") != "Unknown" else 0
        ]) / 5.0
        
        assessment["confidence"] = 0.5 + (data_completeness * 0.4)
        
        return assessment
    
    def _generate_detailed_summary(self, data: Dict[str, Any]) -> str:
        """
        Generate a comprehensive summary from the extracted data.
        
        Args:
            data: Dictionary with all extracted information
            
        Returns:
            Formatted summary text
        """
        # Start with patient information
        patient_name = data['patient_info']['name']
        provider_name = data['patient_info']['provider']
        
        # Build a comprehensive narrative paragraph first
        narrative_paragraphs = []
        
        # First paragraph: overview of the call
        first_para = f"This is a healthcare follow-up conversation between {patient_name} and a care manager"
        if provider_name != "Provider":
            first_para += f" from {provider_name}'s office"
        first_para += "."
        
        # Add age if available
        if data['patient_info']['age']:
            first_para += f" The patient is {data['patient_info']['age']} years old."
        
        # Add dialogue summary if available (from transformer models)
        if "dialogue_summary" in data:
            # The dialogue summary should already be formatted with paragraphs
            narrative_paragraphs.append(first_para)
            narrative_paragraphs.append(data["dialogue_summary"])
        else:
            # Create our own detailed paragraph about the conversation
            conditions_text = ""
            if data['conditions']:
                conditions_text = f", who has been diagnosed with {', '.join(data['conditions'])}"
            
            health_status = "stable"
            if data['health_assessment']['status'] == "concerning":
                health_status = "concerning"
            
            # Expand the first paragraph with more context
            first_para += f" The patient{conditions_text} appears to be in {health_status} condition."
            
            # Add medication information
            if data['medications']['medications']:
                med_names = list(set([m['name'] for m in data['medications']['medications']]))
                first_para += f" Medications discussed include {', '.join(med_names[:3])}"
                if len(med_names) > 3:
                    first_para += f" and {len(med_names) - 3} others"
                first_para += "."
                
                # Add adherence info
                adherence = data['medications']['adherence']
                if adherence != "Unknown":
                    first_para += f" The patient's medication adherence is {adherence.lower()}."
            
            narrative_paragraphs.append(first_para)
            
            # Second paragraph: Additional health details
            second_para = ""
            
            # Add vital signs
            vitals_text = []
            vitals = data['vital_signs']
            
            if vitals.get('blood_pressure') and len(vitals['blood_pressure']) > 0:
                bp = vitals['blood_pressure'][0]['full']
                vitals_text.append(f"blood pressure of {bp}")
            
            if vitals.get('glucose') and len(vitals['glucose']) > 0:
                glucose = vitals['glucose'][0]['value']
                vitals_text.append(f"blood glucose of {glucose} mg/dL")
            
            if vitals.get('weight_change'):
                direction = vitals['weight_change']['direction']
                value = vitals['weight_change']['value']
                unit = vitals['weight_change']['unit']
                vitals_text.append(f"{direction} {value} {unit}")
            
            if vitals_text:
                second_para += f"The patient has a {', '.join(vitals_text)}. "
            
            # Add lifestyle information in narrative form
            lifestyle = data['lifestyle']
            lifestyle_elements = []
            
            if lifestyle['exercise']['type'] != "Unknown":
                exercise_info = f"engages in {lifestyle['exercise']['type'].lower()}"
                if lifestyle['exercise']['frequency'] != "Unknown":
                    exercise_info += f" {lifestyle['exercise']['frequency']}"
                if lifestyle['exercise']['duration'] != "Unknown":
                    exercise_info += f" for {lifestyle['exercise']['duration']}"
                lifestyle_elements.append(exercise_info)
            
            if lifestyle['diet']['habits'] != "Unknown":
                diet_info = f"maintains a {lifestyle['diet']['habits'].lower()} diet"
                if lifestyle['diet']['restrictions']:
                    diet_info += f" while restricting {', '.join(lifestyle['diet']['restrictions'])}"
                lifestyle_elements.append(diet_info)
            
            if lifestyle['smoking'] != "Unknown":
                lifestyle_elements.append(f"is a {lifestyle['smoking'].lower()}")
            
            if lifestyle['alcohol'] != "Unknown":
                lifestyle_elements.append(f"is an {lifestyle['alcohol'].lower()} alcohol consumer")
            
            if lifestyle_elements:
                second_para += "Regarding lifestyle, the patient "
                if len(lifestyle_elements) == 1:
                    second_para += lifestyle_elements[0] + "."
                else:
                    second_para += ", ".join(lifestyle_elements[:-1]) + f" and {lifestyle_elements[-1]}."
            
            # Add preventive care
            if data['preventive_care']:
                care_elements = []
                for care, status in data['preventive_care'].items():
                    care_name = care.replace('_', ' ').lower()
                    care_elements.append(f"{care_name} was {status.lower()}")
                
                if care_elements:
                    second_para += " Preventive care discussed included "
                    if len(care_elements) == 1:
                        second_para += care_elements[0] + "."
                    else:
                        second_para += ", ".join(care_elements[:-1]) + f" and {care_elements[-1]}."
            
            # Add follow-up information
            if data['follow_up']['timeframe']:
                action = data['follow_up'].get('action', 'follow-up').lower()
                timeframe = data['follow_up']['timeframe']
                second_para += f" A {action} is scheduled in {timeframe}."
            
            if second_para:
                narrative_paragraphs.append(second_para)
        
        # Additional data in more structured format (after the narrative paragraphs)
        summary_parts = []
        
        # Add patient header
        summary_parts.append(f"PATIENT: {patient_name}")
        summary_parts.append(f"PROVIDER: {provider_name}")
        if data['patient_info']['age']:
            summary_parts.append(f"AGE: {data['patient_info']['age']}")
        summary_parts.append("")
        
        # Add the narrative paragraphs (comprehensive summary)
        summary_parts.append("SUMMARY:")
        for paragraph in narrative_paragraphs:
            summary_parts.append(paragraph)
            summary_parts.append("")  # Add blank line after each paragraph
        
        # Add key details in structured format
        # Only include sections if they have meaningful content
        
        # Conditions
        if data['conditions']:
            summary_parts.append("CONDITIONS:")
            summary_parts.append(", ".join(data['conditions']))
            summary_parts.append("")
        
        # Medications with adherence
        if data['medications']['medications']:
            summary_parts.append("MEDICATIONS:")
            med_names = list(set([m['name'] for m in data['medications']['medications']]))
            summary_parts.append(", ".join(med_names))
            summary_parts.append(f"Adherence: {data['medications']['adherence']}")
            summary_parts.append("")
        
        # Health assessment
        assessment = data['health_assessment']
        status = assessment['status']
        summary_parts.append("ASSESSMENT:")
        
        if status == "stable":
            summary_parts.append("Patient appears to be in stable condition. No urgent concerns identified.")
        else:
            summary_parts.append("Some health parameters require attention:")
            for concern in assessment['concerns']:
                summary_parts.append(f"- {concern['detail']}")
        
        # Positive health factors
        if assessment['positives']:
            summary_parts.append("\nPositive factors:")
            for positive in assessment['positives']:
                summary_parts.append(f"- {positive['detail']}")
        
        # Return the complete summary
        return "\n".join(summary_parts)