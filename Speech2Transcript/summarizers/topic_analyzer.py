import re
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
import math

# Optional transformer models
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    import numpy as np
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class TopicDiscovery:
    """Dynamically discover topics and key themes in conversations."""
    
    def __init__(self, use_transformers: bool = True):
        """
        Initialize topic discovery system.
        
        Args:
            use_transformers: Whether to use transformer models
        """
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE
        
        # Initialize topic analysis components
        if self.use_transformers:
            try:
                # Load sentence-transformers model for embedding
                self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
                self.model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(self.device)
            except Exception as e:
                print(f"Error loading transformer models: {e}")
                self.use_transformers = False
        
        # Define common medical topics for clustering
        self.topic_keywords = {
            "medications": ["medication", "medicine", "drug", "pill", "prescription", "dose", "pharmacy", 
                           "refill", "take", "taking", "tablet", "capsule", "inject", "inhaler"],
            "symptoms": ["symptom", "pain", "discomfort", "feeling", "dizzy", "tired", "fatigue", "nausea", 
                        "headache", "cough", "fever", "sore", "ache", "swelling"],
            "vital_signs": ["blood pressure", "heart rate", "temperature", "pulse", "oxygen", "breathing", 
                           "rate", "systolic", "diastolic", "glucose", "sugar", "reading", "level"],
            "diabetes": ["diabetes", "sugar", "glucose", "insulin", "A1C", "diabetic", "metformin", 
                        "hyperglycemia", "hypoglycemia", "pump", "ozempic", "jardiance"],
            "heart": ["heart", "cardiac", "cardiovascular", "chest pain", "palpitation", "arrhythmia", 
                     "blood pressure", "hypertension", "cholesterol", "statin", "lipitor"],
            "lifestyle": ["diet", "exercise", "activity", "walking", "eating", "food", "nutrition", 
                         "weight", "sleep", "stress", "smoking", "alcohol", "drinking"],
            "preventive": ["screening", "check-up", "vaccination", "vaccine", "immunization", "colonoscopy", 
                          "mammogram", "pap smear", "prostate", "prevention"],
            "follow_up": ["appointment", "follow up", "schedule", "visit", "call", "check in", 
                         "return", "next time", "see you", "contact"],
            "concerns": ["worry", "concerned", "afraid", "scared", "anxious", "anxiety", "stress", 
                        "depression", "mental", "emotional", "psychological"]
        }
        
        # Common words to ignore
        self.stopwords = {"the", "a", "an", "and", "but", "or", "because", "as", "if", "when", "than", 
                         "but", "for", "with", "about", "against", "between", "into", "through", 
                         "during", "before", "after", "above", "below", "to", "from", "up", "down", 
                         "in", "out", "on", "off", "over", "under", "again", "further", "then", 
                         "once", "here", "there", "all", "any", "both", "each", "few", "more", 
                         "most", "other", "some", "such", "no", "nor", "not", "only", "own", 
                         "same", "so", "than", "too", "very", "can", "will", "just", "should", 
                         "now", "i", "you", "he", "she", "we", "they", "it", "me", "him", "her", 
                         "us", "them", "this", "that", "these", "those", "am", "is", "are", "was", 
                         "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
                         "did", "doing", "would", "could", "should", "might", "must", "okay", "ok", 
                         "yes", "no", "yeah", "well", "um", "uh", "like", "know", "think", "see", "look",
                         "hi", "hello", "thank", "thanks", "thank you", "good", "going", "really",
                         "mister", "doctor", "dr", "mr", "mrs", "ms", "miss"}
    
    def extract_topics(self, conversation: List[Tuple[str, str]], 
                       speaker_roles: Dict[str, str]) -> Dict[str, Any]:
        """
        Dynamically discover topics in the conversation.
        
        Args:
            conversation: List of (speaker, text) tuples
            speaker_roles: Dictionary mapping speakers to roles
            
        Returns:
            Dictionary with discovered topics and insights
        """
        # Extract all utterances
        all_utterances = [text for _, text in conversation]
        care_manager_utterances = [text for speaker, text in conversation 
                                  if speaker == speaker_roles.get("care_manager")]
        patient_utterances = [text for speaker, text in conversation 
                             if speaker == speaker_roles.get("patient")]
        
        # Find the most significant topics based on keyword density
        topic_scores = self._score_topics_by_keywords(all_utterances)
        
        # Sort topics by importance
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        significant_topics = [topic for topic, score in sorted_topics if score > 0.05]
        
        # Extract key sentences for each significant topic
        topic_key_sentences = {}
        for topic in significant_topics:
            topic_key_sentences[topic] = self._extract_key_sentences(all_utterances, topic)
        
        # Identify main topics discussed by the care manager (questions asked)
        cm_topics = self._score_topics_by_keywords(care_manager_utterances)
        cm_topics = sorted(cm_topics.items(), key=lambda x: x[1], reverse=True)
        cm_significant_topics = [topic for topic, score in cm_topics if score > 0.05]
        
        # Identify main topics mentioned by the patient (concerns, issues)
        patient_topics = self._score_topics_by_keywords(patient_utterances)
        patient_topics = sorted(patient_topics.items(), key=lambda x: x[1], reverse=True)
        patient_significant_topics = [topic for topic, score in patient_topics if score > 0.05]
        
        # Identify questions and answers
        questions = self._extract_questions(conversation, speaker_roles)
        
        # Extract numerical values mentioned (could be important measurements)
        numerical_values = self._extract_numerical_values(all_utterances)
        
        # Identify potential concerns or issues
        concerns = self._identify_concerns(conversation, speaker_roles)
        
        # Detect sentiment around key topics
        topic_sentiment = {}
        for topic in significant_topics:
            topic_sentiment[topic] = self._analyze_topic_sentiment(conversation, topic)
        
        # Look for temporal markers (indicators of time-based information)
        temporal_markers = self._extract_temporal_markers(conversation)
        
        # Attempt to identify cause-effect relationships
        causal_relationships = self._extract_causal_relationships(conversation)
        
        # Build the topic analysis result
        result = {
            "primary_topics": significant_topics[:3] if len(significant_topics) >= 3 else significant_topics,
            "all_topics": sorted_topics,
            "topic_sentences": topic_key_sentences,
            "care_manager_focus": cm_significant_topics[:2] if len(cm_significant_topics) >= 2 else cm_significant_topics,
            "patient_focus": patient_significant_topics[:2] if len(patient_significant_topics) >= 2 else patient_significant_topics,
            "questions": questions,
            "numerical_values": numerical_values,
            "concerns": concerns,
            "topic_sentiment": topic_sentiment,
            "temporal_markers": temporal_markers,
            "causal_relationships": causal_relationships
        }
        
        # Add vector-based topic clustering if transformers are available
        if self.use_transformers:
            result["semantic_clusters"] = self._cluster_utterances_by_embedding(all_utterances)
        
        return result
    
    def _score_topics_by_keywords(self, utterances: List[str]) -> Dict[str, float]:
        """
        Score topics by keyword density in utterances.
        
        Args:
            utterances: List of utterance texts
            
        Returns:
            Dictionary mapping topics to importance scores
        """
        # Combine all text
        combined_text = " ".join(utterances).lower()
        
        # Count keywords for each topic
        topic_matches = {}
        for topic, keywords in self.topic_keywords.items():
            matches = 0
            for keyword in keywords:
                matches += len(re.findall(r'\b' + re.escape(keyword) + r'\b', combined_text))
            topic_matches[topic] = matches
        
        # Calculate importance score (normalized by total matches and keywords)
        total_matches = sum(topic_matches.values())
        if total_matches == 0:
            return {topic: 0.0 for topic in self.topic_keywords}
        
        return {topic: count / total_matches for topic, count in topic_matches.items()}
    
    def _extract_key_sentences(self, utterances: List[str], topic: str) -> List[str]:
        """
        Extract key sentences that best represent a topic.
        
        Args:
            utterances: List of utterance texts
            topic: Topic to extract sentences for
            
        Returns:
            List of key sentences for the topic
        """
        # Get relevant keywords for the topic
        keywords = self.topic_keywords.get(topic, [])
        if not keywords:
            return []
        
        # Split all text into sentences
        sentences = []
        for utterance in utterances:
            sentences.extend(re.split(r'(?<=[.!?])\s+', utterance))
        
        # Score sentences by keyword presence
        sentence_scores = []
        for sentence in sentences:
            score = 0
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', sentence.lower()):
                    score += 1
            if score > 0:
                sentence_scores.append((sentence, score))
        
        # Sort by score and return top sentences
        sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
        return [sentence for sentence, _ in sorted_sentences[:3]]
    
    def _extract_questions(self, conversation: List[Tuple[str, str]], 
                          speaker_roles: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Extract questions and their answers from the conversation.
        
        Args:
            conversation: List of (speaker, text) tuples
            speaker_roles: Dictionary mapping speakers to roles
            
        Returns:
            List of question-answer pairs
        """
        care_manager = speaker_roles.get("care_manager")
        patient = speaker_roles.get("patient")
        
        questions = []
        
        for i, (speaker, text) in enumerate(conversation):
            # Identify potential questions (by question mark or common question phrases)
            is_question = False
            if "?" in text:
                is_question = True
            elif speaker == care_manager and re.search(r'\b(how|what|when|where|why|do you|have you|are you|did you|could you|would you)\b', text.lower()):
                # Common question patterns without question marks
                is_question = True
            
            if is_question:
                # Look for an answer in the next 3 utterances
                answer_text = None
                answer_speaker = None
                
                for j in range(i+1, min(i+4, len(conversation))):
                    ans_speaker, ans_text = conversation[j]
                    if ans_speaker != speaker:
                        answer_text = ans_text
                        answer_speaker = ans_speaker
                        break
                
                # Determine the topic of the question
                question_topic = "general"
                for topic, keywords in self.topic_keywords.items():
                    for keyword in keywords:
                        if re.search(r'\b' + re.escape(keyword) + r'\b', text.lower()):
                            question_topic = topic
                            break
                    if question_topic != "general":
                        break
                
                # Add to questions list
                questions.append({
                    "question": text,
                    "answer": answer_text,
                    "question_speaker": speaker,
                    "answer_speaker": answer_speaker,
                    "topic": question_topic
                })
        
        return questions
    
    def _extract_numerical_values(self, utterances: List[str]) -> List[Dict[str, Any]]:
        """
        Extract numerical values that might be important measurements.
        
        Args:
            utterances: List of utterance texts
            
        Returns:
            List of extracted numerical values with context
        """
        numerical_values = []
        
        for utterance in utterances:
            # Look for numbers followed by units or in specific contexts
            number_matches = re.finditer(r'(\d+(?:\.\d+)?)\s*(mg|kg|mmHg|mm|cm|pounds|lbs|minutes|hours|days|weeks|months|years|%)', utterance, re.IGNORECASE)
            
            for match in number_matches:
                value = float(match.group(1))
                unit = match.group(2).lower()
                
                # Get some context around the number
                start = max(0, match.start() - 30)
                end = min(len(utterance), match.end() + 30)
                context = utterance[start:end]
                
                # Try to determine what is being measured
                measurement_type = "unknown"
                if unit in ["mg/dL", "mmol/L"] or "glucose" in context.lower() or "sugar" in context.lower():
                    measurement_type = "blood_glucose"
                elif unit in ["mmHg"] or "blood pressure" in context.lower() or "systolic" in context.lower() or "diastolic" in context.lower():
                    measurement_type = "blood_pressure"
                elif unit in ["pounds", "lbs", "kg"] or "weight" in context.lower():
                    measurement_type = "weight"
                elif unit in ["years"] and value < 120:
                    measurement_type = "age"
                
                numerical_values.append({
                    "value": value,
                    "unit": unit,
                    "context": context,
                    "measurement_type": measurement_type
                })
        
        return numerical_values
    
    def _identify_concerns(self, conversation: List[Tuple[str, str]], 
                          speaker_roles: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Identify potential concerns or issues mentioned in the conversation.
        
        Args:
            conversation: List of (speaker, text) tuples
            speaker_roles: Dictionary mapping speakers to roles
            
        Returns:
            List of identified concerns
        """
        patient = speaker_roles.get("patient")
        
        concern_indicators = [
            r'\b(worry|worried|concern|concerned|afraid|scared|anxious|anxiety|stress|stressed)\b',
            r'\b(problem|issue|trouble|difficult|hard|pain|hurts|hurting|uncomfortable)\b',
            r'\bnot\s+(feel|feeling|good|well|great|okay|ok|right)\b',
            r'\b(side\s+effect|reaction)\b'
        ]
        
        concerns = []
        
        for speaker, text in conversation:
            # Focus on patient utterances
            if speaker == patient:
                for indicator_pattern in concern_indicators:
                    if re.search(indicator_pattern, text.lower()):
                        # Try to identify what the concern is about
                        concern_topic = "general"
                        for topic, keywords in self.topic_keywords.items():
                            for keyword in keywords:
                                if re.search(r'\b' + re.escape(keyword) + r'\b', text.lower()):
                                    concern_topic = topic
                                    break
                            if concern_topic != "general":
                                break
                        
                        concerns.append({
                            "text": text,
                            "topic": concern_topic,
                            "indicator": re.search(indicator_pattern, text.lower()).group(0)
                        })
                        break  # Only count each utterance once
        
        return concerns
    
    def _analyze_topic_sentiment(self, conversation: List[Tuple[str, str]], topic: str) -> str:
        """
        Analyze the sentiment around a specific topic.
        
        Args:
            conversation: List of (speaker, text) tuples
            topic: Topic to analyze sentiment for
            
        Returns:
            Sentiment analysis ("positive", "negative", "neutral", or "mixed")
        """
        # Get relevant keywords for the topic
        keywords = self.topic_keywords.get(topic, [])
        if not keywords:
            return "neutral"
        
        # Find sentences mentioning the topic
        topic_sentences = []
        
        for _, text in conversation:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for sentence in sentences:
                for keyword in keywords:
                    if re.search(r'\b' + re.escape(keyword) + r'\b', sentence.lower()):
                        topic_sentences.append(sentence)
                        break
        
        if not topic_sentences:
            return "neutral"
        
        # Simple rule-based sentiment analysis
        positive_terms = ["good", "great", "excellent", "better", "improved", "normal", "stable", 
                          "fine", "well", "happy", "glad", "comfortable", "improvement", "progress"]
        negative_terms = ["bad", "worse", "difficult", "problem", "issue", "concern", "pain", 
                          "uncomfortable", "not good", "poor", "trouble", "worried", "worrisome"]
        
        positive_count = 0
        negative_count = 0
        
        for sentence in topic_sentences:
            sentence_lower = sentence.lower()
            for term in positive_terms:
                if re.search(r'\b' + re.escape(term) + r'\b', sentence_lower):
                    positive_count += 1
            
            for term in negative_terms:
                if re.search(r'\b' + re.escape(term) + r'\b', sentence_lower):
                    negative_count += 1
        
        # Determine overall sentiment
        if positive_count > negative_count * 2:
            return "positive"
        elif negative_count > positive_count * 2:
            return "negative"
        elif positive_count > 0 and negative_count > 0:
            return "mixed"
        else:
            return "neutral"
    
    def _extract_temporal_markers(self, conversation: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Extract temporal markers indicating when events happened or will happen.
        
        Args:
            conversation: List of (speaker, text) tuples
            
        Returns:
            List of temporal markers with context
        """
        temporal_markers = []
        
        time_patterns = [
            r'\b(yesterday|today|tomorrow)\b',
            r'\b(last|next)\s+(week|month|year|time|visit|appointment)\b',
            r'\b(\d+)\s+(day|week|month|year)s?\s+(ago|later|from now)\b',
            r'\bin\s+(\d+)\s+(day|week|month|year)s?\b',
            r'\b(morning|afternoon|evening|night)\b',
            r'\b(since|after|before|during|while)\b'
        ]
        
        for speaker, text in conversation:
            for pattern in time_patterns:
                matches = re.finditer(pattern, text.lower())
                
                for match in matches:
                    # Get some context
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end]
                    
                    temporal_markers.append({
                        "marker": match.group(0),
                        "context": context,
                        "speaker": speaker
                    })
        
        return temporal_markers
    
    def _extract_causal_relationships(self, conversation: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Extract potential cause-effect relationships from the conversation.
        
        Args:
            conversation: List of (speaker, text) tuples
            
        Returns:
            List of identified causal relationships
        """
        causal_patterns = [
            r'(.*?)\s+because\s+(.*?)(\.|\?|!|$)',
            r'(.*?)\s+caused by\s+(.*?)(\.|\?|!|$)',
            r'(.*?)\s+resulted in\s+(.*?)(\.|\?|!|$)',
            r'(.*?)\s+led to\s+(.*?)(\.|\?|!|$)',
            r'(.*?)\s+due to\s+(.*?)(\.|\?|!|$)',
            r'if\s+(.*?),\s+then\s+(.*?)(\.|\?|!|$)',
            r'when\s+(.*?),\s+(.*?)(\.|\?|!|$)'
        ]
        
        causal_relationships = []
        
        for speaker, text in conversation:
            for pattern in causal_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    if len(match.groups()) >= 2:
                        cause = match.group(2).strip()
                        effect = match.group(1).strip()
                        
                        if cause and effect:
                            causal_relationships.append({
                                "cause": cause,
                                "effect": effect,
                                "speaker": speaker,
                                "full_text": text
                            })
        
        return causal_relationships
    
    def _cluster_utterances_by_embedding(self, utterances: List[str]) -> List[Dict[str, Any]]:
        """
        Cluster utterances by semantic similarity using embeddings.
        
        Args:
            utterances: List of utterance texts
            
        Returns:
            List of semantic clusters
        """
        if not self.use_transformers:
            return []
        
        # Create sentence embeddings
        embeddings = []
        kept_utterances = []
        
        for utterance in utterances:
            # Skip very short utterances
            if len(utterance.split()) < 3:
                continue
                
            # Get embedding using sentence-transformers
            try:
                inputs = self.tokenizer(utterance, return_tensors="pt", 
                                       padding=True, truncation=True, max_length=128)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Mean pooling to get sentence embedding
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = (sum_embeddings / sum_mask).squeeze().cpu().numpy()
                
                # Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)
                
                embeddings.append(embedding)
                kept_utterances.append(utterance)
            except Exception as e:
                print(f"Error creating embedding: {e}")
                continue
        
        if not embeddings:
            return []
        
        # Convert to numpy array
        embeddings = np.array(embeddings)
        
        # Simple clustering based on pairwise similarity
        clusters = []
        used_indices = set()
        
        for i in range(len(embeddings)):
            if i in used_indices:
                continue
                
            # Find similar utterances
            similarities = np.dot(embeddings, embeddings[i])
            similar_indices = [j for j in range(len(similarities)) 
                              if similarities[j] > 0.7 and j not in used_indices]
            
            # Create a cluster
            if similar_indices:
                cluster_utterances = [kept_utterances[j] for j in similar_indices]
                
                # Try to identify a theme for this cluster
                combined_text = " ".join(cluster_utterances).lower()
                cluster_theme = "misc"
                
                for topic, keywords in self.topic_keywords.items():
                    topic_score = 0
                    for keyword in keywords:
                        topic_score += combined_text.count(keyword.lower())
                    
                    if topic_score > 0:
                        cluster_theme = topic
                        break
                
                clusters.append({
                    "theme": cluster_theme,
                    "utterances": cluster_utterances,
                    "count": len(cluster_utterances),
                    "primary_utterance": kept_utterances[i]
                })
                
                # Mark indices as used
                used_indices.update(similar_indices)
                used_indices.add(i)
        
        # Sort clusters by size
        clusters = sorted(clusters, key=lambda x: x["count"], reverse=True)
        
        return clusters


class InsightExtractor:
    """Extract dynamic, context-specific insights from analyzed conversations."""
    
    def __init__(self):
        """Initialize insight extractor."""
        # Define insight generation patterns
        self.insight_patterns = {
            "adherence": {
                "keywords": ["taking", "medication", "adherence", "regularly", "forget", "missed"],
                "template": "Patient appears to have {adherence_level} medication adherence."
            },
            "lifestyle_change": {
                "keywords": ["exercise", "diet", "weight", "activity", "eating", "changed", "started"],
                "template": "Patient has made changes to {lifestyle_aspect}."
            },
            "health_improvement": {
                "keywords": ["better", "improved", "improvement", "progress", "good", "normal", "stable"],
                "template": "Patient reports improvement in {health_aspect}."
            },
            "health_decline": {
                "keywords": ["worse", "declined", "not good", "problem", "issue", "concern", "pain"],
                "template": "Patient reports concerns about {health_aspect}."
            },
            "follow_up_needed": {
                "keywords": ["check", "monitor", "follow", "appointment", "schedule", "next", "return"],
                "template": "Follow-up is scheduled to monitor {follow_up_reason}."
            }
        }
    
    def generate_insights(self, topic_analysis: Dict[str, Any], 
                         extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate dynamic insights from topic analysis and extracted data.
        
        Args:
            topic_analysis: Results from topic discovery
            extracted_data: Structured data extracted from the conversation
            
        Returns:
            List of generated insights
        """
        insights = []
        
        # Analyze primary topics to generate topic-specific insights
        if "primary_topics" in topic_analysis:
            insights.extend(self._generate_topic_insights(topic_analysis))
        
        # Generate insights about patient concerns
        if "concerns" in topic_analysis and topic_analysis["concerns"]:
            insights.extend(self._generate_concern_insights(topic_analysis["concerns"]))
        
        # Generate insights about medication adherence
        if "medications" in extracted_data and "adherence" in extracted_data["medications"]:
            adherence = extracted_data["medications"]["adherence"]
            if adherence != "Unknown":
                insights.append({
                    "type": "adherence",
                    "insight": f"Patient demonstrates {adherence.lower()} medication adherence.",
                    "confidence": 0.8,
                    "supporting_evidence": "Medication adherence assessment"
                })
        
        # Generate insights about vital signs
        if "vital_signs" in extracted_data:
            insights.extend(self._generate_vital_sign_insights(extracted_data["vital_signs"]))
        
        # Generate lifestyle insights
        if "lifestyle" in extracted_data:
            insights.extend(self._generate_lifestyle_insights(extracted_data["lifestyle"]))
        
        # Generate insights from causal relationships
        if "causal_relationships" in topic_analysis and topic_analysis["causal_relationships"]:
            insights.extend(self._generate_causal_insights(topic_analysis["causal_relationships"]))
        
        # Generate temporal insights
        if "temporal_markers" in topic_analysis and topic_analysis["temporal_markers"]:
            insights.extend(self._generate_temporal_insights(topic_analysis["temporal_markers"]))
        
        # Generate semantic cluster insights
        if "semantic_clusters" in topic_analysis and topic_analysis["semantic_clusters"]:
            insights.extend(self._generate_cluster_insights(topic_analysis["semantic_clusters"]))
        
        # Generate question-answer insights
        if "questions" in topic_analysis and topic_analysis["questions"]:
            insights.extend(self._generate_qa_insights(topic_analysis["questions"]))
        
        # Generate health status assessment
        if "health_assessment" in extracted_data:
            insights.extend(self._generate_health_status_insights(extracted_data["health_assessment"]))
        
        # Deduplicate insights
        unique_insights = []
        seen_insights = set()
        
        for insight in insights:
            insight_text = insight["insight"]
            if insight_text not in seen_insights:
                seen_insights.add(insight_text)
                unique_insights.append(insight)
        
        # Sort by confidence
        sorted_insights = sorted(unique_insights, key=lambda x: x.get("confidence", 0), reverse=True)
        
        return sorted_insights
    
    def _generate_topic_insights(self, topic_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights based on topic analysis."""
        insights = []
        
        # Get primary topics and their key sentences
        primary_topics = topic_analysis.get("primary_topics", [])
        topic_sentences = topic_analysis.get("topic_sentences", {})
        topic_sentiment = topic_analysis.get("topic_sentiment", {})
        
        for topic in primary_topics:
            # Skip if no key sentences
            if topic not in topic_sentences or not topic_sentences[topic]:
                continue
            
            # Get sentiment
            sentiment = topic_sentiment.get(topic, "neutral")
            
            # Generate insight based on topic and sentiment
            if sentiment == "positive":
                insight_text = f"Patient reported positive experiences with {topic.replace('_', ' ')}."
            elif sentiment == "negative":
                insight_text = f"Patient expressed concerns about {topic.replace('_', ' ')}."
            elif sentiment == "mixed":
                insight_text = f"Patient showed mixed feelings about {topic.replace('_', ' ')}."
            else:
                insight_text = f"Discussion focused on {topic.replace('_', ' ')}."
            
            # Add supporting evidence
            supporting_evidence = topic_sentences[topic][0] if topic_sentences[topic] else ""
            
            insights.append({
                "type": "topic",
                "topic": topic,
                "insight": insight_text,
                "confidence": 0.7,
                "supporting_evidence": supporting_evidence,
                "sentiment": sentiment
            })
        
        return insights
    
    def _generate_concern_insights(self, concerns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate insights based on patient concerns."""
        insights = []
        
        # Group concerns by topic
        concerns_by_topic = {}
        for concern in concerns:
            topic = concern.get("topic", "general")
            if topic not in concerns_by_topic:
                concerns_by_topic[topic] = []
            concerns_by_topic[topic].append(concern)
        
        # Generate an insight for each topic with concerns
        for topic, topic_concerns in concerns_by_topic.items():
            if len(topic_concerns) == 1:
                insight_text = f"Patient expressed concern about {topic.replace('_', ' ')}."
            else:
                insight_text = f"Patient expressed multiple concerns about {topic.replace('_', ' ')}."
            
            # Get a representative concern as supporting evidence
            supporting_evidence = topic_concerns[0]["text"]
            
            insights.append({
                "type": "concern",
                "topic": topic,
                "insight": insight_text,
                "confidence": 0.75,
                "supporting_evidence": supporting_evidence,
                "concern_count": len(topic_concerns)
            })
        
        return insights
    
    def _generate_vital_sign_insights(self, vital_signs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights based on vital signs."""
        insights = []
        
        # Check blood pressure
        if "blood_pressure" in vital_signs and vital_signs["blood_pressure"]:
            bp = vital_signs["blood_pressure"][0]
            systolic = bp.get("systolic", 0)
            diastolic = bp.get("diastolic", 0)
            
            if systolic > 0 and diastolic > 0:
                # Evaluate blood pressure
                if systolic >= 140 or diastolic >= 90:
                    insight_text = f"Patient's blood pressure is elevated at {bp.get('full', '')}."
                    insights.append({
                        "type": "vital_sign",
                        "vital": "blood_pressure",
                        "insight": insight_text,
                        "confidence": 0.8,
                        "status": "elevated"
                    })
                elif systolic <= 90 or diastolic <= 60:
                    insight_text = f"Patient's blood pressure is low at {bp.get('full', '')}."
                    insights.append({
                        "type": "vital_sign",
                        "vital": "blood_pressure",
                        "insight": insight_text,
                        "confidence": 0.8,
                        "status": "low"
                    })
                else:
                    insight_text = f"Patient's blood pressure is within normal range at {bp.get('full', '')}."
                    insights.append({
                        "type": "vital_sign",
                        "vital": "blood_pressure",
                        "insight": insight_text,
                        "confidence": 0.8,
                        "status": "normal"
                    })
        
        # Check glucose
        if "glucose" in vital_signs and vital_signs["glucose"]:
            glucose = vital_signs["glucose"][0]
            value = glucose.get("value", 0)
            
            if value > 0:
                # Evaluate glucose
                if value >= 200:
                    insight_text = f"Patient's blood glucose is high at {value} mg/dL."
                    insights.append({
                        "type": "vital_sign",
                        "vital": "glucose",
                        "insight": insight_text,
                        "confidence": 0.8,
                        "status": "high"
                    })
                elif value <= 70:
                    insight_text = f"Patient's blood glucose is low at {value} mg/dL."
                    insights.append({
                        "type": "vital_sign",
                        "vital": "glucose",
                        "insight": insight_text,
                        "confidence": 0.8,
                        "status": "low"
                    })
                else:
                    insight_text = f"Patient's blood glucose is within target range at {value} mg/dL."
                    insights.append({
                        "type": "vital_sign",
                        "vital": "glucose",
                        "insight": insight_text,
                        "confidence": 0.8,
                        "status": "normal"
                    })
        
        # Check weight change
        if "weight_change" in vital_signs and vital_signs["weight_change"]:
            weight_change = vital_signs["weight_change"]
            direction = weight_change.get("direction", "")
            value = weight_change.get("value", 0)
            unit = weight_change.get("unit", "")
            
            if direction and value > 0:
                insight_text = f"Patient has {direction} {value} {unit} in weight."
                insights.append({
                    "type": "vital_sign",
                    "vital": "weight",
                    "insight": insight_text,
                    "confidence": 0.8,
                    "status": direction
                })
        
        return insights
    
    def _generate_lifestyle_insights(self, lifestyle: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights based on lifestyle information."""
        insights = []
        
        # Generate exercise insight
        if lifestyle.get("exercise", {}).get("type") != "Unknown":
            exercise_info = lifestyle["exercise"]
            exercise_type = exercise_info.get("type", "")
            frequency = exercise_info.get("frequency", "Unknown")
            duration = exercise_info.get("duration", "Unknown")
            
            insight_text = f"Patient engages in {exercise_type.lower()}"
            if frequency != "Unknown":
                insight_text += f" {frequency}"
            if duration != "Unknown":
                insight_text += f" for {duration}"
            insight_text += "."
            
            insights.append({
                "type": "lifestyle",
                "aspect": "exercise",
                "insight": insight_text,
                "confidence": 0.75
            })
        
        # Generate diet insight
        if lifestyle.get("diet", {}).get("habits") != "Unknown":
            diet_info = lifestyle["diet"]
            habits = diet_info.get("habits", "")
            restrictions = diet_info.get("restrictions", [])
            
            insight_text = f"Patient maintains a {habits.lower()} diet"
            if restrictions:
                insight_text += f" while restricting {', '.join(restrictions)}"
            insight_text += "."
            
            insights.append({
                "type": "lifestyle",
                "aspect": "diet",
                "insight": insight_text,
                "confidence": 0.75
            })
        
        # Generate smoking insight
        if lifestyle.get("smoking") != "Unknown":
            smoking_status = lifestyle.get("smoking", "")
            insight_text = f"Patient is a {smoking_status.lower()}."
            
            insights.append({
                "type": "lifestyle",
                "aspect": "smoking",
                "insight": insight_text,
                "confidence": 0.8
            })
        
        # Generate alcohol insight
        if lifestyle.get("alcohol") != "Unknown":
            alcohol_status = lifestyle.get("alcohol", "")
            insight_text = f"Patient consumes alcohol {alcohol_status.lower()}."
            
            insights.append({
                "type": "lifestyle",
                "aspect": "alcohol",
                "insight": insight_text,
                "confidence": 0.8
            })
        
        return insights
    
    def _generate_causal_insights(self, causal_relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate insights based on causal relationships."""
        insights = []
        
        for relation in causal_relationships:
            cause = relation.get("cause", "")
            effect = relation.get("effect", "")
            
            if cause and effect:
                insight_text = f"{effect} because {cause}."
                
                insights.append({
                    "type": "causal",
                    "insight": insight_text,
                    "confidence": 0.6,
                    "supporting_evidence": relation.get("full_text", "")
                })
        
        return insights
    
    def _generate_temporal_insights(self, temporal_markers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate insights based on temporal markers."""
        insights = []
        
        # Group by marker type
        future_markers = []
        past_markers = []
        
        for marker in temporal_markers:
            marker_text = marker.get("marker", "").lower()
            
            if any(word in marker_text for word in ["next", "tomorrow", "later", "from now", "in"]):
                future_markers.append(marker)
            elif any(word in marker_text for word in ["last", "yesterday", "ago", "since", "before"]):
                past_markers.append(marker)
        
        # Generate insights for future events
        if future_markers:
            # Pick most relevant future marker
            future_marker = future_markers[0]
            context = future_marker.get("context", "")
            
            insight_text = f"Follow-up actions are planned for {future_marker.get('marker')}."
            
            insights.append({
                "type": "temporal",
                "time_reference": "future",
                "insight": insight_text,
                "confidence": 0.7,
                "supporting_evidence": context
            })
        
        # Generate insights for past events
        if past_markers:
            # Pick most relevant past marker
            past_marker = past_markers[0]
            context = past_marker.get("context", "")
            
            insight_text = f"Patient referenced past events from {past_marker.get('marker')}."
            
            insights.append({
                "type": "temporal",
                "time_reference": "past",
                "insight": insight_text,
                "confidence": 0.7,
                "supporting_evidence": context
            })
        
        return insights
    
    def _generate_cluster_insights(self, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate insights based on semantic clusters."""
        insights = []
        
        for cluster in clusters[:3]:  # Focus on top 3 clusters
            theme = cluster.get("theme", "misc")
            count = cluster.get("count", 0)
            
            if count >= 3:  # Only consider substantial clusters
                insight_text = f"Conversation focused significantly on {theme.replace('_', ' ')}."
                
                insights.append({
                    "type": "topic_cluster",
                    "theme": theme,
                    "insight": insight_text,
                    "confidence": 0.65,
                    "supporting_evidence": cluster.get("primary_utterance", "")
                })
        
        return insights
    
    def _generate_qa_insights(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate insights based on question-answer pairs."""
        insights = []
        
        # Group questions by topic
        questions_by_topic = {}
        for question in questions:
            topic = question.get("topic", "general")
            if topic not in questions_by_topic:
                questions_by_topic[topic] = []
            questions_by_topic[topic].append(question)
        
        # Generate insights for topics with multiple questions
        for topic, topic_questions in questions_by_topic.items():
            if len(topic_questions) >= 2:
                insight_text = f"Care manager asked multiple questions about {topic.replace('_', ' ')}."
                
                # Get a representative question-answer pair
                q_a_pair = f"Q: {topic_questions[0].get('question', '')}"
                if topic_questions[0].get('answer'):
                    q_a_pair += f" A: {topic_questions[0].get('answer', '')}"
                
                insights.append({
                    "type": "question_focus",
                    "topic": topic,
                    "insight": insight_text,
                    "confidence": 0.7,
                    "supporting_evidence": q_a_pair,
                    "question_count": len(topic_questions)
                })
        
        return insights
    
    def _generate_health_status_insights(self, health_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights based on health status assessment."""
        insights = []
        
        status = health_assessment.get("status", "stable")
        confidence = health_assessment.get("confidence", 0.5)
        concerns = health_assessment.get("concerns", [])
        positives = health_assessment.get("positives", [])
        
        # Generate overall health status insight
        if status == "stable":
            insight_text = "Patient's overall health appears stable with no urgent concerns."
        else:
            insight_text = "Patient's health shows areas requiring attention."
        
        insights.append({
            "type": "health_status",
            "insight": insight_text,
            "confidence": confidence,
            "status": status
        })
        
        # Generate insights for each concern
        for concern in concerns:
            detail = concern.get("detail", "")
            severity = concern.get("severity", "moderate")
            
            insight_text = f"{detail} ({severity} concern)."
            
            insights.append({
                "type": "health_concern",
                "insight": insight_text,
                "confidence": 0.75,
                "severity": severity
            })
        
        # Generate insights for positive factors
        if positives:
            positive_factors = [p.get("detail", "") for p in positives]
            positive_text = ", ".join(positive_factors[:-1])
            if positive_text:
                positive_text += f" and {positive_factors[-1]}"
            else:
                positive_text = positive_factors[-1]
            
            insight_text = f"Positive health factors include {positive_text}."
            
            insights.append({
                "type": "health_positive",
                "insight": insight_text,
                "confidence": 0.75,
                "positive_count": len(positives)
            })
        
        return insights
