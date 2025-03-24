"""
Speaker identification module that identifies care manager and patient from conversation.
"""
import re
from typing import Dict, List, Tuple, Any
from .base_extractor import BaseExtractor

class SpeakerIdentifier(BaseExtractor):
    """
    Specialized extractor that identifies which speaker is the care manager and
    which speaker is the patient in a medical conversation.
    """
    
    def __init__(self, logger=None):
        """Initialize the speaker identifier extractor."""
        super().__init__(logger)
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize patterns for speaker identification."""
        # Define care manager patterns (phrases typically used by care managers)
        self.care_mgr_indicators = [
            # Standard greetings/introductions
            "how are you feeling", "have you been taking", "do you have any",
            "let me know", "i'll call you", "call me if", "check your",
            "recommended", "prescrib", "how about your", "have you checked",
            
            # Medical terminology used by care managers
            "blood pressure", "glucose levels", "readings", "medication adherence",
            "monitor", "follow up", "diet", "exercise", "lifestyle", 
            
            # Professional phrases
            "dr", "doctor", "office", "clinic", "appointment", "schedule",
            "calling from", "this is", "speaking", "my name is", "care manager",
            
            # Follow-up related phrases
            "i will call you", "next appointment", "weeks", "follow-up",
            "supervising", "provider", "would recommend", "should check"
        ]
        
        # Define patient patterns (phrases typically used by patients)
        self.patient_indicators = [
            # First-person statements
            "i feel", "i've been", "i have", "i'm taking", "my medication",
            "my blood", "my pressure", "my sugar", "my diet", "i walk", "i lost",
            
            # Patient-specific phrases
            "taking my", "my doctor", "my health", "my symptoms", "my readings", 
            "i check", "i measure", "i stopped", "i started", "changed my",
            
            # Lifestyle indicators
            "my routine", "daily", "morning", "evening", "meals", "eat", 
            "family", "work", "tired", "pain", "uncomfortable",
            
            # Responses to care manager
            "yes i am", "no i haven't", "will do", "thank you", "i understand"
        ]
        
        # Symptom reporting terms (strong patient indicator)
        self.symptom_terms = [
            "pain", "hurt", "feel", "felt", "tired", "dizzy", "sick",
            "discomfort", "symptom", "problem with my"
        ]
        
        # Acknowledgment phrases (often used by care managers)
        self.acknowledgment_phrases = [
            "good", "okay", "that's great", "wonderful", "perfect", 
            "i understand", "i see", "thank you for"
        ]
    
    def extract(self, segments: List[Dict], text_column: str = "transcription", 
                speaker_column: str = "speaker") -> Dict[str, Any]:
        """
        Identify which speaker is the care manager and which is the patient.
        
        Args:
            segments: List of conversation segments
            text_column: Column name containing the transcript text
            speaker_column: Column name containing the speaker ID
            
        Returns:
            Dictionary with speaker identification results
        """
        speakers = self._identify_speakers(segments, text_column, speaker_column)
        names = self._extract_names(segments, speakers["care_manager"], 
                                   speakers["patient"], text_column, speaker_column)
        
        return {
            "speakers": speakers,
            "names": names
        }
    
    def _identify_speakers(self, segments: List[Dict], text_column: str, 
                          speaker_column: str) -> Dict[str, str]:
        """
        Identify which speaker is the care manager and which is the patient using
        conversational patterns and domain-specific heuristics.
        
        Args:
            segments: List of conversation segments
            text_column: Column name for the transcript text
            speaker_column: Column name for the speaker ID
            
        Returns:
            Dictionary with care_manager and patient speaker IDs
        """
        if not segments:
            return {"care_manager": "UNKNOWN", "patient": "UNKNOWN"}

        # Extract all unique speaker IDs
        speaker_ids = set(segment.get(speaker_column, "") for segment in segments 
                         if speaker_column in segment and segment.get(speaker_column))
        
        if len(speaker_ids) < 2:
            return {"care_manager": list(speaker_ids)[0] if speaker_ids else "UNKNOWN", 
                    "patient": "UNKNOWN"}

        # Count care-manager-specific patterns for each speaker
        care_mgr_score = {speaker: 0 for speaker in speaker_ids}
        patient_score = {speaker: 0 for speaker in speaker_ids}

        # Analyze all conversation segments
        conversation_flow = []
        for i, segment in enumerate(segments):
            speaker = segment.get(speaker_column, "")
            text = segment.get(text_column, "").lower() if text_column in segment else ""
            
            if not text:
                continue
                
            # Keep track of conversation flow (for turn-taking analysis)
            conversation_flow.append((speaker, text))
            
            # Check for care manager introduction patterns (stronger indicators)
            if i < 5:  # First few segments often contain introductions
                if re.search(r"(this is|calling from|speaking|care manager|office of|dr\.?\s+\w+)", text):
                    care_mgr_score[speaker] += 10
                
                # Care manager typically introduces themselves and states purpose
                if "calling" in text and ("check" in text or "follow" in text):
                    care_mgr_score[speaker] += 8
            
            # Check for questioning patterns (care manager typically asks questions)
            if text.endswith("?"):
                care_mgr_score[speaker] += 2
            
            # Multiple questions in a single turn is strong indicator of care manager
            question_count = text.count("?")
            if question_count > 1:
                care_mgr_score[speaker] += question_count * 2
                
            # Check for title usage (Mr/Mrs/Ms often used by care manager)
            if re.search(r"(mr\.?|mrs\.?|ms\.?)\s+\w+", text):
                care_mgr_score[speaker] += 5
                
            # Medical professionals often validate/acknowledge responses
            if i > 0 and any(phrase in text for phrase in self.acknowledgment_phrases):
                care_mgr_score[speaker] += 1

            # Check for all care manager indicator phrases
            for indicator in self.care_mgr_indicators:
                if indicator in text:
                    care_mgr_score[speaker] += 1

            # Check for all patient indicator phrases
            for indicator in self.patient_indicators:
                if indicator in text:
                    patient_score[speaker] += 1
                    
            # Check for symptom reporting (strong patient indicator)
            if any(symptom in text for symptom in self.symptom_terms):
                patient_score[speaker] += 2
                
            # Check for medication taking patterns (patient)
            if re.search(r"i('m| am)?\s+taking", text) or "my medication" in text:
                patient_score[speaker] += 3
                
            # Check for follow-up scheduling (care manager)
            if "call you" in text or "follow up" in text or "follow-up" in text:
                care_mgr_score[speaker] += 3

        # Analyze turn-taking patterns for additional clues
        if len(conversation_flow) >= 6:
            # Care managers typically initiate conversations
            first_speaker = conversation_flow[0][0]
            care_mgr_score[first_speaker] += 3
            
            # Analyze turn-taking patterns - care managers typically guide conversation
            speaker_counts = {}
            for speaker, _ in conversation_flow:
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
                
            # Care managers typically speak more in medical conversations
            if len(speaker_counts) >= 2:
                most_frequent_speaker = max(speaker_counts, key=speaker_counts.get)
                care_mgr_score[most_frequent_speaker] += 2

        # Determine the most likely care manager
        care_manager = max(care_mgr_score, key=care_mgr_score.get)

        # Find the most likely patient (highest patient score or not the care manager)
        other_speakers = [s for s in speaker_ids if s != care_manager]
        if not other_speakers:
            patient = "UNKNOWN"
        else:
            # If we have patient indicator scores, use them; otherwise pick the first non-care-manager
            patient_candidates = {s: patient_score[s] for s in other_speakers}
            patient = max(patient_candidates, key=patient_candidates.get) if patient_candidates else other_speakers[0]
            
        self._log(f"Speaker role analysis - Care Manager: {care_manager} (score: {care_mgr_score[care_manager]}), " +
                 f"Patient: {patient} (score: {patient_score.get(patient, 0)})")

        return {"care_manager": care_manager, "patient": patient}
    
    def _extract_names(self, segments: List[Dict], care_manager_id: str, patient_id: str, 
                       text_column: str, speaker_column: str) -> Dict[str, str]:
        """
        Extract names of care manager and patient from the entire conversation context.
        Enhanced to analyze the complete transcript for more accurate name detection.
        
        Args:
            segments: Conversation segments
            care_manager_id: ID of the care manager speaker
            patient_id: ID of the patient speaker
            text_column: Column name for the transcript text
            speaker_column: Column name for the speaker ID
            
        Returns:
            Dictionary with patient_name and care_manager_name
        """
        patient_name = "Unknown Patient"
        care_manager_name = "Unknown Care Manager"
        
        # Extract all text by speaker for comprehensive analysis
        all_text_by_speaker = {}
        for segment in segments:
            speaker = segment.get(speaker_column, "")
            text = segment.get(text_column, "") if text_column in segment else ""
            if not text:
                continue
                
            if speaker not in all_text_by_speaker:
                all_text_by_speaker[speaker] = ""
            all_text_by_speaker[speaker] += " " + text
        
        # Combine all text to build a name reference database
        all_text = " ".join(all_text_by_speaker.values())
        
        # STEP 1: Look for care manager introduction patterns in their complete text
        if care_manager_id in all_text_by_speaker:
            care_mgr_text = all_text_by_speaker[care_manager_id]
            
            # Check common introduction patterns
            intro_patterns = [
                r"(?:this is|I'm|I am|my name is|speaking|name's)\s+(\w+)",
                r"(?:this is|from)\s+(?:the office of|dr\.?\s+\w+[,.]?\s+)?(\w+)",
                r"(?:calling|this is)\s+(\w+)(?:\s+from|with)"
            ]
            
            for pattern in intro_patterns:
                matches = re.finditer(pattern, care_mgr_text, re.IGNORECASE)
                for match in matches:
                    potential_name = match.group(1).strip()
                    # Filter out common false positives
                    if (len(potential_name) >= 3 and 
                        potential_name.lower() not in ["the", "a", "an", "just", "from", "this", "that", "your", "office"]):
                        care_manager_name = potential_name.capitalize()
                        self._log(f"Found care manager name '{care_manager_name}' from introduction pattern")
                        break
                if care_manager_name != "Unknown Care Manager":
                    break
        
        # STEP 2: Look for formal title usage to identify the patient's name
        patient_name_candidates = []
        
        title_patterns = [
            # Full patterns with titles
            r"(?:mr|mrs|ms|miss)\.?\s*(\w+)",
            r"(?:mr|mrs|ms|miss)\.?\s*(\w+)'s",
            r"(?:to|with|for|hello|hi|hey)\s+(?:mr|mrs|ms|miss)\.?\s*(\w+)",
            # Direct address patterns
            r"(?:hello|hi|hey|thank|thanks)\s+(\w+)",
        ]
        
        # Check for titles in the whole transcript
        for pattern in title_patterns:
            matches = re.finditer(pattern, all_text, re.IGNORECASE)
            for match in matches:
                potential_name = match.group(1).strip()
                # Filter out common false positives and very short names
                if (len(potential_name) >= 3 and 
                    potential_name.lower() not in ["the", "a", "an", "sir", "mam", "madam", "there", "here", "you", "for", "again"]):
                    
                    # Extract surrounding context to validate
                    text_pos = all_text.lower().find(match.group(0).lower())
                    context_start = max(0, text_pos - 30)
                    context_end = min(len(all_text), text_pos + 50)
                    context = all_text[context_start:context_end]
                    
                    # Match title used
                    title_match = re.search(r"(mr|mrs|ms|miss)\.?", match.group(0), re.IGNORECASE)
                    title = "Mr." if title_match and title_match.group(1).lower() == "mr" else \
                            "Ms." if title_match and title_match.group(1).lower() == "ms" else \
                            "Mrs." if title_match and title_match.group(1).lower() == "mrs" else \
                            "Miss" if title_match and title_match.group(1).lower() == "miss" else ""
                    
                    if title:
                        candidate = {
                            "name": potential_name.capitalize(),
                            "title": title,
                            "full_name": f"{title} {potential_name.capitalize()}",
                            "count": 1,
                            "context": context
                        }
                        
                        # Check if we already have this candidate
                        found = False
                        for existing in patient_name_candidates:
                            if existing["name"].lower() == potential_name.lower():
                                existing["count"] += 1
                                found = True
                                break
                                
                        if not found:
                            patient_name_candidates.append(candidate)
        
        # Choose the most frequent patient name candidate
        if patient_name_candidates:
            best_candidate = max(patient_name_candidates, key=lambda x: x["count"])
            patient_name = best_candidate["full_name"]
            self._log(f"Found patient name '{patient_name}' with {best_candidate['count']} occurrences")
        
        # STEP 3: If patient name is still unknown, look for direct address patterns
        if patient_name == "Unknown Patient":
            # Check for name mentions by the care manager
            if care_manager_id in all_text_by_speaker:
                care_mgr_text = all_text_by_speaker[care_manager_id]
                
                # Look for direct addresses of the patient
                for pattern in [r"(?:hello|hi|hey|thank|thanks)\s+(\w+)", r"how are you\s+(?:doing|feeling)?\s*(?:today|now)?,?\s*(\w+)"]:
                    matches = re.finditer(pattern, care_mgr_text, re.IGNORECASE)
                    for match in matches:
                        potential_name = match.group(1).strip()
                        if (len(potential_name) >= 3 and 
                            potential_name.lower() not in ["the", "a", "an", "sir", "mam", "madam", "there", "here", "you", "today", "doing"]):
                            patient_name = potential_name.capitalize()
                            self._log(f"Found patient first name '{patient_name}' from direct address")
                            break
                    if patient_name != "Unknown Patient":
                        break
        
        # STEP 4: Look for more care manager name mentions if still unknown
        if care_manager_name == "Unknown Care Manager":
            # Check if the patient refers to the care manager by name
            if patient_id in all_text_by_speaker:
                patient_text = all_text_by_speaker[patient_id]
                
                # Look for thank you patterns
                for pattern in [r"(?:thank|thanks),?\s+(\w+)", r"thank you,?\s+(\w+)"]:
                    matches = re.finditer(pattern, patient_text, re.IGNORECASE)
                    for match in matches:
                        potential_name = match.group(1).strip()
                        if (len(potential_name) >= 3 and 
                            potential_name.lower() not in ["so", "much", "very", "doctor", "dr", "for", "you"]):
                            care_manager_name = potential_name.capitalize()
                            self._log(f"Found care manager name '{care_manager_name}' from thank you pattern")
                            break
                    if care_manager_name != "Unknown Care Manager":
                        break
        
        # Process name information (prefix with title if needed for patient)
        if patient_name != "Unknown Patient" and not any(title in patient_name for title in ["Mr.", "Mrs.", "Ms.", "Miss"]):
            # Check if we should add a title based on context
            for segment in segments:
                text = segment.get(text_column, "").lower() if text_column in segment else ""
                if patient_name.lower() in text.lower():
                    if "mr" in text.lower():
                        patient_name = f"Mr. {patient_name}"
                        break
                    elif "mrs" in text.lower():
                        patient_name = f"Mrs. {patient_name}"
                        break
                    elif "ms" in text.lower():
                        patient_name = f"Ms. {patient_name}"
                        break
        
        self._log(f"Final name extraction - Patient: {patient_name}, Care Manager: {care_manager_name}")
        return {"patient_name": patient_name, "care_manager_name": care_manager_name}
