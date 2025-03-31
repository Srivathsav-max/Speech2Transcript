import os
import json
import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import re

# Import base summarizer
from .base_summarizer import BaseSummarizer

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class GeminiSummarizer(BaseSummarizer):
    """
    Gemini-powered medical transcript summarizer.
    
    Uses Google's Gemini API to generate natural, cohesive summaries
    from medical conversation transcripts.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Gemini summarizer.
        
        Args:
            api_key: Google API key for Gemini access (default: uses GOOGLE_API_KEY env var)
            model_name: Gemini model to use
            logger: Optional logger for messages
        """
        super().__init__(logger)
        
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        
        # Check if Gemini is available and configure it
        if GEMINI_AVAILABLE and self.api_key:
            try:
                # Initialize with new client-based approach
                self._log(f"Initializing Gemini API client with model: {model_name}")
                self.client = genai.Client(api_key=self.api_key)
                
                # Try to list models to confirm connection
                try:
                    models = self.client.models.list_models()
                    model_names = [model.name for model in models]
                    self._log(f"Available Gemini models: {', '.join(model_names)}")
                except Exception as e:
                    self._log(f"Could not retrieve list of available models: {e}", level="warning")
                
                self.gemini_available = True
            except Exception as e:
                self._log(f"Error initializing Gemini API: {e}", level="error")
                self.gemini_available = False
        else:
            self._log("Gemini API not available or no API key provided", level="warning")
            self.gemini_available = False
    
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

    def _identify_speakers(self, conversation: List[tuple]) -> Dict[str, str]:
        """
        Identify the likely roles of speakers (patient vs care manager).
        
        Args:
            conversation: List of (speaker, text) tuples
            
        Returns:
            Dictionary mapping speaker IDs to roles
        """
        # Basic heuristic: Care managers typically ask more questions
        question_count = {}
        total_words = {}
        
        for speaker, text in conversation:
            # Count questions by speaker
            if "?" in text:
                question_count[speaker] = question_count.get(speaker, 0) + 1
            
            # Count total words by speaker
            words = len(text.split())
            total_words[speaker] = total_words.get(speaker, 0) + words
        
        # Determine care manager (person who asks more questions)
        if not question_count:
            # If no questions found, the care manager likely speaks less
            care_manager = min(total_words.items(), key=lambda x: x[1])[0] if total_words else None
        else:
            care_manager = max(question_count.items(), key=lambda x: x[1])[0]
        
        # Determine patient (other speakers)
        speakers = list(set(s for s, _ in conversation))
        patient = next((s for s in speakers if s != care_manager), None)
        
        return {
            "care_manager": care_manager,
            "patient": patient
        }
    
    def _extract_basic_info(self, conversation: List[tuple], speakers: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract basic information from conversation to provide context for summary.
        
        Args:
            conversation: List of (speaker, text) tuples
            speakers: Dictionary mapping speaker IDs to roles
            
        Returns:
            Dictionary with basic extracted information
        """
        info = {
            "patient_name": "Patient",
            "provider_name": "Provider",
            "conditions": [],
            "medications": [],
            "vital_signs": {}
        }
        
        # Extract patient and provider names from early conversation
        early_conversation = conversation[:10]
        patient_speaker = speakers.get("patient")
        care_manager = speakers.get("care_manager")
        
        # Look for patient name in care manager greetings
        for speaker, text in early_conversation:
            if speaker == care_manager:
                # Check for greeting patterns with names
                name_match = re.search(r'(?:hello|hi|good\s+(?:morning|afternoon|evening))\s+(?:mr\.|mrs\.|ms\.|miss)?\s*([A-Z][a-z]+)', text)
                if name_match:
                    info["patient_name"] = name_match.group(1)
                
                # Look for provider reference
                provider_match = re.search(r"(?:from|with)\s+(?:Dr\.|Doctor)\s+([A-Z][a-z]+)", text)
                if provider_match:
                    info["provider_name"] = f"Dr. {provider_match.group(1)}"
        
        # Look for common conditions
        condition_patterns = {
            "diabetes": r'\b(?:diabetes|diabetic|blood\s+sugar|glucose|insulin|a1c)\b',
            "hypertension": r'\b(?:hypertension|high\s+blood\s+pressure|blood\s+pressure)\b',
            "heart disease": r'\b(?:heart\s+(?:disease|failure|problem)|cardiac)\b',
            "obesity": r'\b(?:obesity|weight\s+(?:issue|problem|loss|management))\b'
        }
        
        # Extract conditions
        full_text = " ".join(text for _, text in conversation).lower()
        for condition, pattern in condition_patterns.items():
            if re.search(pattern, full_text):
                info["conditions"].append(condition)
        
        # Extract medications
        med_patterns = [
            r'\b(?:taking|take|use|on)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b',
            r'\b(?:medication|medicine|pill|drug)\s+(?:called|named)?\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b'
        ]
        
        # Common medications to look for
        common_meds = ["insulin", "metformin", "ozempic", "lisinopril", "atorvastatin", "aspirin"]
        
        for med in common_meds:
            if re.search(r'\b' + med + r'\b', full_text, re.IGNORECASE):
                info["medications"].append(med)
        
        # Look for blood pressure values
        bp_matches = re.finditer(r'(\d{2,3})[/\\](\d{2,3})', full_text)
        for match in bp_matches:
            systolic = int(match.group(1))
            diastolic = int(match.group(2))
            if 80 <= systolic <= 200 and 40 <= diastolic <= 120:
                info["vital_signs"]["blood_pressure"] = f"{systolic}/{diastolic}"
                break
        
        # Look for glucose values
        glucose_match = re.search(r'(?:glucose|sugar)\s+(?:is|of|at|around|about)?\s*(\d{2,3})', full_text)
        if glucose_match:
            glucose = int(glucose_match.group(1))
            if 50 <= glucose <= 400:
                info["vital_signs"]["glucose"] = glucose
        
        return info
    
    def _generate_summary_prompt(self, conversation_text: str, basic_info: Dict[str, Any]) -> str:
        """
        Generate a prompt for Gemini to create the summary.
        
        Args:
            conversation_text: Full conversation text
            basic_info: Dictionary with basic extracted information
            
        Returns:
            Formatted prompt for Gemini
        """
        prompt = f"""
You are a medical summarization expert. I need you to create a comprehensive, coherent narrative summary of the following healthcare conversation between a care manager and a patient. 

Patient Name: {basic_info['patient_name']}
Provider: {basic_info['provider_name']}
Key Conditions: {', '.join(basic_info['conditions']) if basic_info['conditions'] else 'Unknown'}
Medications: {', '.join(basic_info['medications']) if basic_info['medications'] else 'Unknown'}

Here's the transcript:
```
{conversation_text}
```

Please create a professional, cohesive narrative summary that:
1. Identifies who called whom and for what purpose
2. Summarizes the patient's reported health status, symptoms, and recent events
3. Covers key vitals, medications and adherence
4. Includes information about diet, exercise, and other lifestyle factors
5. Notes preventive care status and any barriers to care
6. Concludes with follow-up plans

The summary should be 4-5 paragraphs in length, written in third-person narrative style like a medical note. Use a professional tone and focus on medically relevant information. Do not use bullet points or structured headings - write it as a continuous narrative similar to a medical progress note.
"""
        return prompt
    
    def _generate_summary_with_gemini(self, prompt: str) -> str:
        """
        Generate summary using Gemini API.
        
        Args:
            prompt: Formatted prompt for Gemini
            
        Returns:
            Generated summary text
        """
        if not self.gemini_available:
            error_msg = "Gemini API not available. Please check your API key and connection."
            self._log(error_msg, level="error")
            return error_msg
        
        try:
            self._log("Sending prompt to Gemini API...")
            self._log(f"Using model: {self.model_name}")
            
            # Generate content with the new client approach
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            self._log("Successfully generated summary with Gemini")
            
            # Extract the text from the response
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'parts'):
                parts_text = ''.join([part.text for part in response.parts if hasattr(part, 'text')])
                if parts_text:
                    return parts_text
            
            # Last resort - convert to string
            return str(response)
                
        except Exception as e:
            error_msg = f"Error generating summary with Gemini: {str(e)}"
            self._log(error_msg, level="error")
            self._log(f"Error details: {type(e).__name__}: {str(e)}", level="error")
            
            # Return a fallback message
            return f"Error generating summary: {str(e)}"
    
    def process_transcript(self, 
                          transcript_path: Optional[str] = None,
                          transcript_data: Optional[Dict[str, Any]] = None,
                          output_path: Optional[str] = None,
                          text_column: str = "transcription",
                          speaker_column: str = "speaker") -> Dict[str, Any]:
        """
        Process a transcript and generate a comprehensive summary.
        
        Args:
            transcript_path: Path to transcript file (JSON)
            transcript_data: Alternatively, provide transcript data directly
            output_path: Optional path to save results
            text_column: Column name containing the transcription text
            speaker_column: Column name containing the speaker ID
            
        Returns:
            Dictionary with the summary and extracted information
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
        
        # Identify speakers
        speakers = self._identify_speakers(text_data["conversation"])
        
        # Extract basic information for context
        basic_info = self._extract_basic_info(text_data["conversation"], speakers)
        
        # Generate prompt for Gemini
        prompt = self._generate_summary_prompt(text_data["full_text"], basic_info)
        
        # Generate summary with Gemini
        summary = self._generate_summary_with_gemini(prompt)
        
        # Create result object
        result = {
            "summary": summary,
            "extracted_info": basic_info
        }
        
        # Save results if output path provided
        if output_path:
            self.save_results(result, output_path)
        
        return result