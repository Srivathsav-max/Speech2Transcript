import os 
import re
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification


class LearnedTemporalModel:
    """Temporal sequence model that learns from data rather than using static probabilities"""
    
    def __init__(self, model_path=None, device="cpu"):
        self.device = device
        self.model = None
        self.tokenizer = None
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            # Fallback to rule-based approach
            print("No temporal model found, falling back to rule-based approach")
    
    def _load_model(self, path):
        """Load pre-trained temporal sequence model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForTokenClassification.from_pretrained(path)
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded temporal model from {path}")
        except Exception as e:
            print(f"Error loading temporal model: {e}")
            self.model = None
    
    def process_timeline(self, entities, text):
        """Assign temporal context to entities using the learned model"""
        # If model is not available, use rule-based approach
        if self.model is None:
            return self._rule_based_temporal_assignment(entities, text)
        
        # With model, do sequence labeling
        sequence_labeled_entities = self._model_based_temporal_assignment(entities, text)
        
        # Apply HeidelTime for absolute timestamp extraction
        return self._enhance_with_heideltime(sequence_labeled_entities, text)
    
    def _model_based_temporal_assignment(self, entities, text):
        """Use the sequence model to assign temporal contexts"""
        import torch
        
        # Prepare examples with context
        examples = []
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                if "context" in entity.metadata:
                    context = entity.metadata["context"]
                    examples.append({
                        "context": context,
                        "entity": entity
                    })
        
        # Process in batches
        batch_size = 16
        all_predictions = []
        
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i+batch_size]
            contexts = [ex["context"] for ex in batch]
            
            # Tokenize
            inputs = self.tokenizer(
                contexts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
            
            # Process predictions
            for j, pred in enumerate(predictions):
                # Get the most common predicted label (excluding special tokens)
                valid_preds = [p.item() for p, m in zip(pred, inputs["attention_mask"][j]) if m.item() == 1]
                if not valid_preds:
                    continue
                    
                # Get most common prediction
                from collections import Counter
                pred_counter = Counter(valid_preds)
                most_common_pred = pred_counter.most_common(1)[0][0]
                
                # Map to temporal context
                temporal_contexts = [
                    "past_history", "recent_past", "current", 
                    "immediate_future", "distant_future"
                ]
                temporal_context = temporal_contexts[most_common_pred % len(temporal_contexts)]
                
                # Assign to entity
                examples[i+j]["entity"].temporal_context = temporal_context
                
        # Return updated entities
        return entities
    
    def _rule_based_temporal_assignment(self, entities, text):
        """Fallback rule-based approach for temporal assignment"""
        # Temporal expression patterns
        temporal_patterns = {
            "past_history": [
                r'(?:history|histories|previously|prior|before|earlier|in\s+the\s+past|past\s+medical\s+history|pmh|used\s+to|formerly|once)',
                r'(?:as\s+a\s+child|when\s+younger|years\s+ago|long\s+ago|chronic|longstanding)'
            ],
            "recent_past": [
                r'(?:recently|lately|last\s+(?:few|couple|several)\s+(?:days|weeks|months)|past\s+(?:day|week|month)|earlier\s+(?:today|this\s+week|this\s+month))',
                r'(?:yesterday|earlier\s+today|last\s+(?:night|evening|morning|week|month))'
            ],
            "current": [
                r'(?:now|currently|presently|at\s+present|at\s+this\s+time|right\s+now|today)',
                r'(?:ongoing|continuing|still|remains|persists|present|active)'
            ],
            "immediate_future": [
                r'(?:tomorrow|soon|shortly|in\s+the\s+next\s+(?:few|couple|several)\s+(?:days|weeks)|next\s+(?:week|month))',
                r'(?:upcoming|approaching|impending|scheduled|planned|arranged)'
            ],
            "distant_future": [
                r'(?:in\s+(?:the\s+)?(?:future|long\s+term|long\s+run)|eventually|someday|later|down\s+the\s+road)',
                r'(?:next\s+(?:year|summer|winter|spring|fall|season)|months\s+from\s+now|years\s+from\s+now)'
            ]
        }
        
        # Assign temporal context based on surrounding text
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                if not entity.position or not text:
                    entity.temporal_context = "unknown"
                    continue
                
                start, end = entity.position
                
                # Extract context (sentence containing the entity)
                context_start = max(0, text.rfind(".", 0, start))
                if context_start == -1:
                    context_start = max(0, start - 200)
                
                context_end = min(len(text), text.find(".", end))
                if context_end == -1:
                    context_end = min(len(text), end + 200)
                
                context = text[context_start:context_end]
                
                # Check for temporal patterns in context
                best_match = "current"  # Default
                best_score = 0
                
                for temporal_state, patterns in temporal_patterns.items():
                    score = 0
                    for pattern in patterns:
                        matches = re.finditer(r'\b' + pattern + r'\b', context, re.IGNORECASE)
                        for match in matches:
                            # Calculate score based on proximity to the entity
                            match_start, match_end = match.span()
                            distance = min(abs(match_start - start), abs(match_end - end))
                            proximity_score = 1.0 / (1.0 + 0.01 * distance)
                            score += proximity_score
                    
                    if score > best_score:
                        best_score = score
                        best_match = temporal_state
                
                entity.temporal_context = best_match
        
        return entities
    
    def _enhance_with_heideltime(self, entities, text):
        """Enhance temporal information with HeidelTime or similar parser"""
        try:
            # Extract temporal expressions
            temporal_expressions = self._extract_temporal_expressions(text)
            
            # For each entity, check if there's a nearby temporal expression
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    if not entity.position:
                        continue
                    
                    start, end = entity.position
                    
                    # Find nearest temporal expression
                    nearest_expr = None
                    min_distance = float('inf')
                    
                    for expr in temporal_expressions:
                        expr_start, expr_end = expr["position"]
                        distance = min(abs(expr_start - end), abs(expr_end - start))
                        
                        if distance < min_distance:
                            min_distance = distance
                            nearest_expr = expr
                    
                    # If a nearby temporal expression exists, enhance the entity with it
                    if nearest_expr and min_distance < 100:  # 100 chars threshold
                        entity.metadata["explicit_temporal"] = nearest_expr["expression"]
                        entity.metadata["temporal_type"] = nearest_expr["type"]
                        
                        if nearest_expr.get("normalized_value"):
                            entity.metadata["temporal_value"] = nearest_expr["normalized_value"]
            
            return entities
            
        except Exception as e:
            print(f"Error enhancing with HeidelTime: {e}")
            return entities
    
    def _extract_temporal_expressions(self, text):
        """Extract temporal expressions from text (simplified)"""
        temporal_expressions = []
        
        patterns = {
            "absolute_date": [
                r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}',
                r'\d{1,2}/\d{1,2}/\d{2,4}',
                r'\d{4}-\d{2}-\d{2}'
            ],
            "relative_date": [
                r'(?:yesterday|today|tomorrow|last\s+(?:week|month|year)|next\s+(?:week|month|year))',
                r'(?:in|within|after|before|during|over|since|for)\s+(?:the\s+)?(?:last|next|past|previous|coming|future|upcoming)\s+(\d+)\s+(?:day|days|week|weeks|month|months|year|years)'
            ],
            "duration": [
                r'(?:for|over)\s+(?:the\s+)?(?:last|past|previous)\s+(\d+)\s+(?:day|days|week|weeks|month|months|year|years)',
                r'(?:since|from)\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)(?:\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})?'
            ],
            "frequency": [
                r'(?:once|twice|three\s+times|four\s+times|five\s+times|six\s+times|seven\s+times)\s+(?:a|per|every)\s+(?:day|week|month|year)',
                r'(?:daily|weekly|monthly|yearly|annually|every\s+(?:day|other\s+day|week|other\s+week|month|other\s+month|year|other\s+year))',
                r'(?:every|each)\s+(?:morning|evening|night|afternoon)'
            ]
        }
        
        for expr_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    temporal_expressions.append({
                        "expression": match.group(0),
                        "position": match.span(),
                        "type": expr_type,
                        "normalized_value": self._normalize_temporal_expression(match.group(0), expr_type)
                    })
        
        return temporal_expressions
    
    def _normalize_temporal_expression(self, expression, expr_type):
        """Normalize temporal expressions to standard format (simplified)"""
        import datetime
        now = datetime.datetime.now()
        
        # Handle some common cases
        if expr_type == "absolute_date":
            try:
                # Try to parse with dateutil
                return expression
            except:
                return None
        
        elif expr_type == "relative_date":
            if "yesterday" in expression.lower():
                yesterday = now - datetime.timedelta(days=1)
                return yesterday.strftime("%Y-%m-%d")
            elif "today" in expression.lower():
                return now.strftime("%Y-%m-%d")
            elif "tomorrow" in expression.lower():
                tomorrow = now + datetime.timedelta(days=1)
                return tomorrow.strftime("%Y-%m-%d")
            elif "last week" in expression.lower():
                last_week = now - datetime.timedelta(days=7)
                return last_week.strftime("%Y-%m-%d")
        
        return None