from typing import Dict, List, Any, Optional, Union, Tuple
import re
import logging
from abc import ABC, abstractmethod


class BaseExtractor(ABC):
    """Base class for all extractors."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize with optional logger."""
        self.logger = logger
    
    def _log(self, message: str, level: str = "info") -> None:
        """Log message with appropriate level."""
        if self.logger:
            if level == "info":
                self.logger.info(message)
            elif level == "error":
                self.logger.error(message)
            elif level == "warning":
                self.logger.warning(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    @abstractmethod
    def extract(self, data: Any) -> Dict[str, Any]:
        """
        Extract information from provided data.
        
        Args:
            data: Data to extract information from
            
        Returns:
            Dictionary with extracted information
        """
        pass


class SpeakerExtractor(BaseExtractor):
    """Extract and identify speakers from transcript segments."""
    
    def extract(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract speaker information from segments.
        
        Args:
            segments: List of transcript segments
            
        Returns:
            Dictionary with speaker information
        """
        # Build conversation and speaker data
        conversation = []
        speaker_texts = {}
        speaker_times = {}
        
        for segment in segments:
            if "transcription" not in segment or "speaker" not in segment:
                continue
                
            speaker = segment["speaker"]
            text = segment.get("transcription", "")
            
            if not text or not speaker:
                continue
                
            # Add to conversation timeline
            conversation.append((speaker, text))
            
            # Add to speaker-specific text
            if speaker not in speaker_texts:
                speaker_texts[speaker] = []
            speaker_texts[speaker].append(text)
            
            # Track speaking time
            if "start" in segment and "end" in segment:
                duration = segment["end"] - segment["start"]
                if speaker not in speaker_times:
                    speaker_times[speaker] = 0
                speaker_times[speaker] += duration
        
        # Convert speaker text lists to strings
        speaker_texts = {speaker: " ".join(texts) for speaker, texts in speaker_texts.items()}
        
        # Identify speakers (care manager vs patient)
        speakers = list(speaker_texts.keys())
        speaker_roles = self._identify_speaker_roles(conversation, speaker_texts, speaker_times)
        
        return {
            "conversation": conversation,
            "speaker_texts": speaker_texts,
            "speaker_times": speaker_times,
            "speakers": speaker_roles
        }
    
    def _identify_speaker_roles(self, conversation, speaker_texts, speaker_times):
        """Identify roles of speakers."""
        speakers = list(speaker_texts.keys())
        
        if len(speakers) < 2:
            return {
                "care_manager": speakers[0] if speakers else "Unknown",
                "patient": "Unknown"
            }
        
        # Multiple features to identify care manager vs patient
        features = {}
        
        # 1. Look for introductions in the first segments
        for i, (speaker, text) in enumerate(conversation[:5]):
            if re.search(r"(care manager|calling from|office|clinic|dr\.)", text.lower()):
                features["introduction"] = {
                    "care_manager": speaker,
                    "weight": 5.0,
                    "position": i
                }
                break
        
        # 2. Question analysis (care managers ask more questions)
        question_counts = {}
        for speaker in speakers:
            question_counts[speaker] = 0
        
        for speaker, text in conversation:
            if re.search(r"\?|how are you|have you been|do you|are you|could you|would you", text.lower()):
                question_counts[speaker] += 1
        
        max_questions = max(question_counts.values()) if question_counts else 0
        if max_questions > 0:
            # Find speaker with most questions
            for speaker, count in question_counts.items():
                if count == max_questions:
                    features["questions"] = {
                        "care_manager": speaker,
                        "weight": 3.0,
                        "ratio": count / sum(question_counts.values()) if sum(question_counts.values()) > 0 else 0
                    }
        
        # 3. Speaking time analysis (care managers often speak more)
        if speaker_times:
            most_speaking = max(speaker_times.items(), key=lambda x: x[1])
            features["speaking_time"] = {
                "care_manager": most_speaking[0],
                "weight": 1.0,
                "ratio": most_speaking[1] / sum(speaker_times.values()) if sum(speaker_times.values()) > 0 else 0
            }
        
        # Weighted voting to determine roles
        scores = {speaker: 0.0 for speaker in speakers}
        total_weight = 0.0
        
        for feature_name, feature in features.items():
            care_manager = feature["care_manager"]
            weight = feature["weight"]
            scores[care_manager] += weight
            total_weight += weight
        
        # Default to using the first speakers if no features found
        if total_weight == 0.0:
            return {
                "care_manager": speakers[0],
                "patient": speakers[1] if len(speakers) > 1 else "Unknown"
            }
        
        # Normalize scores and identify care manager
        scores = {s: (score / total_weight) for s, score in scores.items()}
        sorted_speakers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        care_manager = sorted_speakers[0][0]
        
        # Determine patient (highest speaking time among non-care managers)
        non_care_managers = [s for s in speakers if s != care_manager]
        if not non_care_managers:
            patient = "Unknown"
        else:
            if speaker_times:
                patient_scores = {s: speaker_times.get(s, 0) for s in non_care_managers}
                patient = max(patient_scores.items(), key=lambda x: x[1])[0]
            else:
                patient = non_care_managers[0]
        
        return {
            "care_manager": care_manager,
            "patient": patient
        }


class PatientInfoExtractor(BaseExtractor):
    """Extract patient information from conversation."""
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract patient information.
        
        Args:
            data: Dictionary with conversation and speaker information
            
        Returns:
            Dictionary with patient information
        """
        conversation = data["conversation"]
        speakers = data["speakers"]
        
        care_manager = speakers["care_manager"]
        patient = speakers["patient"]
        
        # Extract patient and provider identifiers in a context-based way
        # instead of relying on specific title patterns
        
        patient_info = {
            "name": "Patient",  # Default to generic "Patient" instead of "Unknown"
            "provider": "Provider",  # Default to generic "Provider" instead of "Unknown"
            "age": None
        }
        
        # Find potential names in the conversation
        person_names = self._extract_person_names(conversation)
        
        # Try to determine which names belong to patient vs provider
        if len(person_names) >= 2:
            # Look for greeting patterns in the first 5 utterances
            for i, (speaker, text) in enumerate(conversation[:5]):
                if speaker == care_manager:
                    for name in person_names:
                        # Check if the care manager appears to be greeting the patient
                        if re.search(rf"\b(hi|hello|good|morning|afternoon|evening).*\b{re.escape(name)}\b", text.lower()):
                            patient_info["name"] = name
                            # Remove this name from candidates for provider
                            person_names.remove(name)
                            break
            
            # Look for provider references (e.g., "calling from Dr. X's office")
            for i, (speaker, text) in enumerate(conversation[:5]):
                if speaker == care_manager:
                    for name in person_names:
                        if re.search(rf"\b(from|with|at).*\b{re.escape(name)}('s)?\b.*(office|clinic|practice|center)", text.lower()):
                            patient_info["provider"] = name
                            break
        
        # If we couldn't identify specific names, use fallback method
        # Extract from greeting patterns as a backup
        if patient_info["name"] == "Patient":
            for speaker, text in conversation[:5]:
                if speaker == care_manager:
                    name_match = re.search(r"(mr\.|mrs\.|ms\.|miss)\s+([a-z]+)", text.lower())
                    if name_match:
                        patient_info["name"] = f"{name_match.group(1).capitalize()} {name_match.group(2).capitalize()}"
                        break
        
        if patient_info["provider"] == "Provider":
            for speaker, text in conversation[:5]:
                if speaker == care_manager:
                    provider_match = re.search(r"dr\.\s+([a-z]+)", text.lower())
                    if provider_match:
                        patient_info["provider"] = f"Dr. {provider_match.group(1).capitalize()}"
                        break
        
        # Extract age if mentioned
        for speaker, text in conversation:
            age_match = re.search(r"(\d+)\s+years?\s+old", text.lower())
            if age_match:
                candidate_age = int(age_match.group(1))
                if 1 <= candidate_age <= 120:  # Reasonable age range
                    patient_info["age"] = candidate_age
                    break
        
        return patient_info
    
    def _extract_person_names(self, conversation: List[Tuple[str, str]]) -> List[str]:
        """Extract potential person names from the conversation."""
        potential_names = []
        
        # First look for formal titles followed by words
        for speaker, text in conversation[:10]:  # Check first 10 utterances
            # Look for formal titles
            formal_matches = re.finditer(r"\b(mr\.|mrs\.|ms\.|miss|dr\.)\s+([A-Za-z]+)", text, re.IGNORECASE)
            for match in formal_matches:
                title = match.group(1).capitalize()
                name = match.group(2).capitalize()
                full_name = f"{title} {name}"
                if full_name not in potential_names:
                    potential_names.append(full_name)
        
        # Look for capitalized words that might be names (excluding sentence starts)
        for speaker, text in conversation[:10]:
            # Split into sentences
            sentences = re.split(r'[.!?]\s+', text)
            for sentence in sentences:
                words = sentence.split()
                for i, word in enumerate(words):
                    # Skip first word of sentence as it's naturally capitalized
                    if i > 0 and re.match(r'^[A-Z][a-z]{2,}$', word):
                        # Check if it's not a common word
                        if word.lower() not in ["the", "and", "but", "for", "with", "about"]:
                            if word not in potential_names:
                                potential_names.append(word)
        
        return potential_names


class MedicalConditionExtractor(BaseExtractor):
    """Extract medical conditions from conversation."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize with condition patterns."""
        super().__init__(logger)
        
        # Common chronic conditions to look for
        self.condition_patterns = {
            "diabetes": r"diabet(es|ic)|blood (sugar|glucose)|a1c|insulin|ozempic|metformin",
            "hypertension": r"(high blood pressure|hypertension)|blood pressure",
            "heart disease": r"heart (disease|condition|failure|problem)|cardiac|cardiovascular|cholesterol|statins",
            "copd": r"copd|chronic obstructive|emphysema|pulmonary",
            "asthma": r"asthma|inhaler|breathing problem",
            "arthritis": r"arthritis|joint pain",
            "obesity": r"weight|obesity|bmi|body mass"
        }
    
    def extract(self, data: Dict[str, Any]) -> List[str]:
        """
        Extract medical conditions.
        
        Args:
            data: Dictionary with conversation and speaker information
            
        Returns:
            List of identified conditions
        """
        conversation = data["conversation"]
        speakers = data["speakers"]
        care_manager = speakers["care_manager"]
        
        conditions = []
        full_text = " ".join([text for _, text in conversation]).lower()
        
        for condition, pattern in self.condition_patterns.items():
            if re.search(pattern, full_text):
                # Check if condition is actually discussed meaningfully
                for speaker, text in conversation:
                    if speaker == care_manager and re.search(pattern, text.lower()):
                        conditions.append(condition)
                        break
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(conditions))


class MedicationExtractor(BaseExtractor):
    """Extract medications and adherence information."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize with medication patterns."""
        super().__init__(logger)
        
        # Common medication names and classes
        self.med_patterns = [
            r"insulin",
            r"metformin",
            r"ozempic|semaglutide",
            r"zempek|zempic",  # Common misspellings of Ozempic
            r"lisinopril",
            r"atorvastatin|lipitor",
            r"statin",
            r"aspirin",
            r"blood (pressure|thinner)",
            r"ace inhibitor",
            r"beta blocker",
            r"calcium channel",
            r"diuretic",
        ]
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract medications and adherence.
        
        Args:
            data: Dictionary with conversation and speaker information
            
        Returns:
            Dictionary with medication information
        """
        conversation = data["conversation"]
        speakers = data["speakers"]
        
        # Extract medications
        medications = []
        full_text = " ".join([text for _, text in conversation]).lower()
        
        for pattern in self.med_patterns:
            matches = re.finditer(pattern, full_text)
            for match in matches:
                # Get some context around the match
                start = max(0, match.start() - 30)
                end = min(len(full_text), match.end() + 30)
                context = full_text[start:end]
                
                # Extract medication name
                med_name = match.group(0)
                medications.append({
                    "name": med_name,
                    "context": context
                })
        
        # Look for medication adherence information
        adherence = self._extract_adherence_info(conversation, speakers)
        
        return {
            "medications": medications,
            "adherence": adherence
        }
    
    def _extract_adherence_info(self, conversation, speakers):
        """Extract medication adherence information."""
        care_manager = speakers["care_manager"]
        patient = speakers["patient"]
        
        adherence_status = "Unknown"
        
        # Look for medication adherence questions
        for i, (speaker, text) in enumerate(conversation):
            if speaker == care_manager and re.search(r"(tak(ing|e)|following).+(medication|medicine|prescription)", text.lower()):
                # Look for the patient's response
                for j in range(i+1, min(i+4, len(conversation))):
                    if conversation[j][0] == patient:
                        response = conversation[j][1].lower()
                        if re.search(r"yes|yeah|taking|regularly", response):
                            adherence_status = "Good"
                        elif re.search(r"not|miss|forgot|sometimes|try", response):
                            adherence_status = "Variable"
                        elif re.search(r"no|stopped|don't", response):
                            adherence_status = "Poor"
                        break
        
        return adherence_status


class VitalSignsExtractor(BaseExtractor):
    """Extract vital signs from conversation."""
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract vital signs.
        
        Args:
            data: Dictionary with conversation and speaker information
            
        Returns:
            Dictionary with vital sign information
        """
        conversation = data["conversation"]
        speakers = data["speakers"]
        patient = speakers["patient"]
        
        # Initialize vital signs
        vitals = {
            "blood_pressure": [],
            "glucose": [],
            "weight_change": None
        }
        
        # Extract blood pressure readings
        bp_pattern = r"(\d{2,3})[/\\](\d{2,3})"
        for speaker, text in conversation:
            bp_matches = re.finditer(bp_pattern, text)
            for match in bp_matches:
                systolic = int(match.group(1))
                diastolic = int(match.group(2))
                
                # Validate if values are in reasonable range
                if 80 <= systolic <= 220 and 40 <= diastolic <= 130:
                    vitals["blood_pressure"].append({
                        "systolic": systolic,
                        "diastolic": diastolic,
                        "full": f"{systolic}/{diastolic}"
                    })
        
        # Extract blood glucose readings
        glucose_pattern = r"(\d{2,3})(?:\s*(?:mg|mmol|blood sugar|glucose))"
        for speaker, text in conversation:
            glucose_matches = re.finditer(glucose_pattern, text.lower())
            for match in glucose_matches:
                glucose = int(match.group(1))
                
                # Validate if values are in reasonable range for mg/dL
                if 50 <= glucose <= 500:
                    vitals["glucose"].append({
                        "value": glucose,
                        "unit": "mg/dL"
                    })
        
        # Look for mentions of specific glucose values
        for speaker, text in conversation:
            if re.search(r"(\d{2,3}),?\s*(\d{2,3})", text):
                # This could be a blood glucose range or multiple readings
                numbers = re.findall(r"(\d{2,3})", text)
                for num in numbers:
                    glucose = int(num)
                    if 70 <= glucose <= 300:  # Typical blood glucose range
                        vitals["glucose"].append({
                            "value": glucose,
                            "unit": "mg/dL"
                        })
        
        # Extract weight changes
        weight_pattern = r"(lost|gained)\s+(\d+)\s+(pound|lb|kg)"
        for speaker, text in conversation:
            if speaker == patient:
                weight_match = re.search(weight_pattern, text.lower())
                if weight_match:
                    direction = weight_match.group(1)
                    amount = int(weight_match.group(2))
                    unit = weight_match.group(3)
                    
                    vitals["weight_change"] = {
                        "direction": direction,
                        "value": amount,
                        "unit": "pounds" if unit in ["pound", "lb"] else "kg"
                    }
        
        return vitals


class LifestyleExtractor(BaseExtractor):
    """Extract lifestyle information from conversation."""
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract lifestyle information.
        
        Args:
            data: Dictionary with conversation and speaker information
            
        Returns:
            Dictionary with lifestyle information
        """
        conversation = data["conversation"]
        speakers = data["speakers"]
        patient = speakers["patient"]
        
        # Initialize lifestyle info
        lifestyle = {
            "exercise": {
                "frequency": "Unknown",
                "duration": "Unknown",
                "type": "Unknown"
            },
            "diet": {
                "habits": "Unknown",
                "restrictions": []
            },
            "smoking": "Unknown",
            "alcohol": "Unknown"
        }
        
        # Extract exercise information
        for speaker, text in conversation:
            if speaker == patient:
                # Look for walking frequency
                walk_match = re.search(r"walk\w*\s+([\d\-]+)\s*(time|minute|hour)s?\s*(day|week|daily|a day)", text.lower())
                if walk_match:
                    quantity = walk_match.group(1)
                    unit = walk_match.group(2)
                    timeframe = walk_match.group(3)
                    
                    if unit == "time":
                        lifestyle["exercise"]["frequency"] = f"{quantity} times per {timeframe}"
                    else:
                        lifestyle["exercise"]["duration"] = f"{quantity} {unit}s"
                    
                    lifestyle["exercise"]["type"] = "Walking"
        
        # Extract diet information
        for speaker, text in conversation:
            if speaker == patient:
                if re.search(r"(diet|eating|food)", text.lower()):
                    if re.search(r"(cutting|cut down|reduc\w+)\s+(\w+)", text.lower()):
                        restriction_match = re.search(r"(cutting|cut down|reduc\w+)\s+(\w+)", text.lower())
                        if restriction_match:
                            lifestyle["diet"]["restrictions"].append(restriction_match.group(2))
                    
                    if re.search(r"(good|healthy|balanced|careful)", text.lower()):
                        lifestyle["diet"]["habits"] = "Healthy"
                    elif re.search(r"(bad|poor|cheat|not good)", text.lower()):
                        lifestyle["diet"]["habits"] = "Needs improvement"
        
        # Extract smoking information
        for speaker, text in conversation:
            if speaker == patient:
                if re.search(r"(smok\w+|cigarette)", text.lower()):
                    if re.search(r"(no|don't|never)\s+smok\w+", text.lower()):
                        lifestyle["smoking"] = "Non-smoker"
                    elif re.search(r"(quit|stopped)\s+smok\w+", text.lower()):
                        lifestyle["smoking"] = "Former smoker"
                    elif re.search(r"(yes|do)\s+smok\w+", text.lower()):
                        lifestyle["smoking"] = "Smoker"
        
        # Extract alcohol consumption
        for speaker, text in conversation:
            if speaker == patient:
                if re.search(r"(alcohol|drink|wine|beer)", text.lower()):
                    if re.search(r"(no|don't|never)\s+(drink|alcohol)", text.lower()):
                        lifestyle["alcohol"] = "Non-drinker"
                    elif re.search(r"(occasionally|sometimes|social|rare|once in a while)", text.lower()):
                        lifestyle["alcohol"] = "Occasional"
                    elif re.search(r"(regular|daily|often)", text.lower()):
                        lifestyle["alcohol"] = "Regular"
        
        return lifestyle


class FollowUpExtractor(BaseExtractor):
    """Extract follow-up information from the end of the conversation."""
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract follow-up information.
        
        Args:
            data: Dictionary with conversation and speaker information
            
        Returns:
            Dictionary with follow-up information
        """
        conversation = data["conversation"]
        speakers = data["speakers"]
        care_manager = speakers["care_manager"]
        
        follow_up = {
            "timeframe": None,
            "action": None
        }
        
        # Look at the end of the conversation for follow-up plans
        end_segments = conversation[-5:] if len(conversation) > 5 else conversation
        
        for speaker, text in end_segments:
            if speaker == care_manager:
                # Look for timeframe
                time_match = re.search(r"(call|appointment|visit|follow\s*up).+(in|next)\s+(\d+)\s+(day|week|month)", text.lower())
                if time_match:
                    action = time_match.group(1)
                    quantity = time_match.group(3)
                    unit = time_match.group(4)
                    
                    follow_up["action"] = action.strip()
                    follow_up["timeframe"] = f"{quantity} {unit}{'s' if int(quantity) > 1 else ''}"
                    break
        
        return follow_up


class PreventiveCareExtractor(BaseExtractor):
    """Extract preventive care information from conversation."""
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract preventive care information.
        
        Args:
            data: Dictionary with conversation and speaker information
            
        Returns:
            Dictionary with preventive care information
        """
        conversation = data["conversation"]
        
        preventive_care = {}
        
        # Look for preventive care topics
        preventive_patterns = {
            "colonoscopy": r"colonoscopy",
            "vaccines": r"vaccine|vaccination|flu shot|pneumonia",
            "mammogram": r"mammogram",
            "pap_smear": r"pap|cervical",
            "prostate": r"prostate|psa"
        }
        
        for topic, pattern in preventive_patterns.items():
            for speaker, text in conversation:
                if re.search(pattern, text.lower()):
                    # Extract timeframe if available
                    time_match = re.search(r"(last|next)\s+(\w+)", text.lower())
                    if time_match:
                        timeframe = f"{time_match.group(1)} {time_match.group(2)}"
                        preventive_care[topic] = timeframe
                    else:
                        preventive_care[topic] = "Discussed"
        
        return preventive_care
