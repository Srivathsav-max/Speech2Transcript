import re
import torch
from sentence_transformers import SentenceTransformer

class VectorizedContradictionDetector:
    """Detects contradictions between entities using vectorized approach for scalability"""
    
    def __init__(self, sentence_model=None, device="cpu"):
        self.device = device
        self.sentence_model = None
        
        # Load sentence embedding model if provided
        if sentence_model:
            self._load_sentence_model(sentence_model)
    
    def _load_sentence_model(self, model_name):
        """Load sentence embedding model"""
        try:
            self.sentence_model = SentenceTransformer(model_name).to(self.device)
            print(f"Loaded sentence embedding model: {model_name}")
        except Exception as e:
            print(f"Error loading sentence embedding model: {e}")
            self.sentence_model = None
    
    def detect_contradictions(self, entities, text=None):
        """Detect contradictions between entities using vectorized approach"""
        # If we don't have a sentence model, fall back to pairwise comparison
        if self.sentence_model is None:
            return self._pairwise_contradiction_detection(entities)
        
        # Vectorized approach
        contradictions = []
        
        # Group entities by type
        for entity_type, entity_list in entities.items():
            # Skip if fewer than 2 entities
            if len(entity_list) < 2:
                continue
            
            # Create statement representations for each entity
            statements = []
            for entity in entity_list:
                # Skip if hypothetical or uncertain
                if entity.is_hypothetical or (entity.is_uncertain and entity.confidence < 0.6):
                    continue
                
                # Create a statement representation
                statement = entity.value
                if entity.normalized_value:
                    statement = entity.normalized_value
                
                # Add negation marker
                if entity.is_negated:
                    statement = f"not {statement}"
                
                # Add temporal context if available
                if entity.temporal_context and entity.temporal_context != "unknown":
                    statement = f"{statement} ({entity.temporal_context})"
                
                # For numeric values, extract numbers
                numeric_values = []
                if entity.normalized_value:
                    numeric_values = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', entity.normalized_value) if n]
                
                statements.append({
                    "entity": entity,
                    "statement": statement,
                    "is_negated": entity.is_negated,
                    "temporal_context": entity.temporal_context,
                    "numeric_values": numeric_values
                })
            
            # Get statement texts for embedding
            statement_texts = [s["statement"] for s in statements]
            
            # Skip if no statements
            if not statement_texts:
                continue
            
            try:
                # Encode statements as vectors
                embeddings = self.sentence_model.encode(statement_texts, convert_to_tensor=True)
                
                # Calculate pairwise similarities
                similarity_matrix = self._calculate_pairwise_similarities(embeddings)
                
                # Check for potential contradictions
                for i in range(len(statements)):
                    for j in range(i+1, len(statements)):
                        similarity = similarity_matrix[i][j].item()
                        statement1 = statements[i]
                        statement2 = statements[j]
                        
                        # Skip comparison if temporal contexts are explicitly different
                        if (statement1["temporal_context"] and statement2["temporal_context"] and
                            statement1["temporal_context"] != statement2["temporal_context"]):
                            continue
                        
                        # Check for logical contradiction (similar statements but opposite negation)
                        if similarity > 0.8 and statement1["is_negated"] != statement2["is_negated"]:
                            contradictions.append({
                                "type": "logical_contradiction",
                                "entity_type": entity_type,
                                "entity1": statement1["entity"],
                                "entity2": statement2["entity"],
                                "description": f"Logical contradiction: '{statement1['statement']}' vs '{statement2['statement']}'"
                            })
                        
                        # Check for value contradiction (similar entities with different values)
                        elif similarity > 0.7 and statement1["numeric_values"] and statement2["numeric_values"]:
                            # Compare numeric values
                            if self._numeric_values_contradicting(
                                    statement1["numeric_values"], statement2["numeric_values"]):
                                contradictions.append({
                                    "type": "value_contradiction",
                                    "entity_type": entity_type,
                                    "entity1": statement1["entity"],
                                    "entity2": statement2["entity"],
                                    "description": f"Value contradiction: '{statement1['statement']}' vs '{statement2['statement']}'"
                                })
                
            except Exception as e:
                print(f"Error in vectorized contradiction detection: {e}")
                # Fall back to regular detection
                return self._pairwise_contradiction_detection(entities)
        
        return contradictions
    
    def _calculate_pairwise_similarities(self, embeddings):
        """Calculate pairwise cosine similarities between embeddings"""
        import torch.nn.functional as F
        
        # Normalize embeddings to unit length
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.transpose(0, 1))
        
        return similarity_matrix
    
    def _numeric_values_contradicting(self, values1, values2):
        """Check if numeric values are contradicting each other"""
        for v1 in values1:
            for v2 in values2:
                # Skip if very small values (could be noise)
                if v1 < 0.01 and v2 < 0.01:
                    continue
                
                # Calculate relative difference
                relative_diff = abs(v1 - v2) / max(abs(v1), abs(v2))
                
                # Different thresholds based on magnitude
                if v1 > 100:  # Large values like blood glucose
                    threshold = 0.2  # 20% difference
                elif v1 > 10:  # Medium values like blood pressure
                    threshold = 0.15  # 15% difference
                else:  # Small values like lab results
                    threshold = 0.1  # 10% difference
                
                if relative_diff > threshold:
                    return True
        
        return False
    
    def _pairwise_contradiction_detection(self, entities):
        """Fallback to pairwise contradiction detection"""
        contradictions = []
        
        # Define contradiction types and their detection functions
        contradiction_types = {
            "value_contradiction": self._detect_value_contradiction,
            "temporal_contradiction": self._detect_temporal_contradiction,
            "logical_contradiction": self._detect_logical_contradiction,
            "source_contradiction": self._detect_source_contradiction
        }
        
        # Group entities by type for comparison
        for entity_type, entity_list in entities.items():
            # Skip if fewer than 2 entities (no contradiction possible)
            if len(entity_list) < 2:
                continue
            
            # Check each pair of entities
            for i, entity1 in enumerate(entity_list):
                for j in range(i+1, len(entity_list)):
                    entity2 = entity_list[j]
                    
                    # Apply each contradiction detection method
                    for contradiction_type, detector in contradiction_types.items():
                        contradiction = detector(entity1, entity2)
                        if contradiction:
                            contradictions.append({
                                "type": contradiction_type,
                                "entity_type": entity_type,
                                "entity1": entity1,
                                "entity2": entity2,
                                "description": contradiction
                            })
        
        return contradictions
    
    def _detect_value_contradiction(self, entity1, entity2):
        """Detect contradiction in values"""
        # Skip if entities are negated differently
        if entity1.is_negated != entity2.is_negated:
            return None
        
        # Skip if hypothetical or uncertain
        if entity1.is_hypothetical or entity2.is_hypothetical or entity1.is_uncertain or entity2.is_uncertain:
            return None
        
        # Check for numeric value contradictions
        if entity1.normalized_value and entity2.normalized_value:
            # Extract numbers from both values
            nums1 = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', entity1.normalized_value)]
            nums2 = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', entity2.normalized_value)]
            
            if nums1 and nums2:
                # Allow small differences in values (e.g., rounding differences)
                threshold = 0.1  # 10% difference threshold
                
                for n1 in nums1:
                    for n2 in nums2:
                        # Skip if values are close enough
                        if abs(n1 - n2) <= max(n1, n2) * threshold:
                            continue
                        
                        # Check temporal context - not contradictory if from different times
                        if entity1.temporal_context != entity2.temporal_context:
                            if entity1.temporal_context and entity2.temporal_context:
                                return None
                        
                        return f"Value contradiction: {entity1.value} vs {entity2.value}"
        
        return None
    
    def _detect_temporal_contradiction(self, entity1, entity2):
        """Detect contradiction in temporal context"""
        # Skip if one doesn't have temporal context
        if not entity1.temporal_context or not entity2.temporal_context:
            return None
        
        # Check if values are similar but temporal contexts conflict
        if entity1.normalized_value and entity2.normalized_value:
            # Extract numbers from both values
            nums1 = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', entity1.normalized_value)]
            nums2 = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', entity2.normalized_value)]
            
            # Check if values are similar
            if nums1 and nums2 and abs(nums1[0] - nums2[0]) <= 5:
                # Check if temporal contexts are different
                if entity1.temporal_context != entity2.temporal_context:
                    return f"Temporal contradiction: {entity1.value} at {entity1.temporal_context} vs {entity2.value} at {entity2.temporal_context}"
        
        return None
    
    def _detect_logical_contradiction(self, entity1, entity2):
        """Detect logical contradiction (e.g., having and not having a condition)"""
        # Check if one is negated and the other isn't
        if entity1.is_negated != entity2.is_negated:
            # Check if values are similar
            if entity1.value.lower() == entity2.value.lower():
                return f"Logical contradiction: {'' if entity1.is_negated else 'not '}{entity1.value} vs {'' if entity2.is_negated else 'not '}{entity2.value}"
        
        return None
    
    def _detect_source_contradiction(self, entity1, entity2):
        """Detect contradiction between patient and provider statements"""
        # Check if from different sources
        if entity1.source != entity2.source and entity1.source in ["patient", "provider"] and entity2.source in ["patient", "provider"]:
            # Check if values conflict
            if entity1.normalized_value and entity2.normalized_value:
                # Extract numbers from both values
                nums1 = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', entity1.normalized_value)]
                nums2 = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', entity2.normalized_value)]
                
                if nums1 and nums2:
                    # Significant difference threshold
                    threshold = 0.2  # 20% difference threshold
                    
                    for n1 in nums1:
                        for n2 in nums2:
                            if abs(n1 - n2) > max(n1, n2) * threshold:
                                return f"Source contradiction: {entity1.source} says {entity1.value} but {entity2.source} says {entity2.value}"
        
        return None