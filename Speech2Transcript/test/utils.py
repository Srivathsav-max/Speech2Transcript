def _handle_speaker_overlaps(self, df, text_column, speaker_column):
    """Handle overlapping speaker segments"""
    overlaps = []
    sorted_df = df.sort_values(by=['start'])
    
    for i in range(len(sorted_df) - 1):
        current = sorted_df.iloc[i]
        next_seg = sorted_df.iloc[i+1]
        
        if current['end'] > next_seg['start']:
            # Overlap detected
            overlap_size = current['end'] - next_seg['start']
            overlap_percentage = overlap_size / (current['end'] - current['start'])
            
            overlaps.append({
                'index1': current.name,
                'index2': next_seg.name,
                'overlap_size': overlap_size,
                'overlap_percentage': overlap_percentage
            })
    
    # For significant overlaps, merge the text
    if overlaps:
        for overlap in overlaps:
            if overlap['overlap_percentage'] > 0.5:  # Significant overlap
                idx1 = overlap['index1']
                idx2 = overlap['index2']
                
                # Merge text from both segments
                merged_text = f"{df.loc[idx1, text_column]} [OVERLAP] {df.loc[idx2, text_column]}"
                df.loc[idx1, text_column] = merged_text
                
                # Mark the second segment for removal
                df.loc[idx2, text_column] = ""
        
        # Remove empty segments
        df = df[df[text_column].str.strip() != ""]
    
    return df

def _analyze_patient_provider_disagreements(self, entities, patient_text, provider_text):
    """Analyze disagreements between patient and provider statements"""
    disagreements = []
    
    for entity_type, entity_list in entities.items():
        # Skip irrelevant entity types
        if entity_type not in ["blood_pressure", "blood_glucose", "medications", "symptoms"]:
            continue
        
        for entity in entity_list:
            if not entity.value:
                continue
                
            in_patient = entity.value.lower() in patient_text.lower()
            in_provider = entity.value.lower() in provider_text.lower()
            
            # If entity appears in both patient and provider text
            if in_patient and in_provider:
                # Get context for both mentions
                patient_context = self._extract_entity_context(entity.value, patient_text)
                provider_context = self._extract_entity_context(entity.value, provider_text)
                
                # Check for negation in either context
                patient_negated = self._is_negated(patient_context, entity.value)
                provider_negated = self._is_negated(provider_context, entity.value)
                
                # Check for disagreement
                if patient_negated != provider_negated:
                    disagreements.append({
                        "entity_type": entity_type,
                        "value": entity.value,
                        "patient_context": patient_context,
                        "provider_context": provider_context,
                        "patient_negated": patient_negated,
                        "provider_negated": provider_negated
                    })
    
    return disagreements

def _identify_missing_topics(self, entities, chronic_conditions):
    """Identify standard-of-care topics that should have been discussed but weren't"""
    missing_topics = []
    
    # Define standard topics by condition
    standard_topics = {
        "diabetes": ["blood_glucose", "hba1c", "foot_exam", "eye_exam", "kidney_function"],
        "hypertension": ["blood_pressure", "salt_intake", "exercise", "medication_adherence"],
        "heart_failure": ["weight", "edema", "dyspnea", "orthopnea", "medication_adherence"],
        "asthma": ["inhaler_use", "triggers", "peak_flow", "symptoms"],
        "copd": ["inhaler_use", "oxygen", "dyspnea", "activity_level"]
    }
    
    # Check if key topics for each condition were discussed
    for condition in chronic_conditions:
        condition_name = condition["condition"].lower()
        
        # Find matching standard condition
        for std_condition, topics in standard_topics.items():
            if std_condition in condition_name:
                # Check each standard topic
                for topic in topics:
                    topic_discussed = False
                    
                    # Check if topic was discussed
                    for entity_type, entity_list in entities.items():
                        for entity in entity_list:
                            if topic.lower() in entity.value.lower():
                                topic_discussed = True
                                break
                        if topic_discussed:
                            break
                    
                    if not topic_discussed:
                        missing_topics.append({
                            "condition": std_condition,
                            "missing_topic": topic,
                            "importance": "high" if topic in ["blood_glucose", "blood_pressure", "medication_adherence"] else "medium"
                        })
    
    return missing_topics

def _resolve_temporal_ambiguities(self, timeline):
    """Resolve ambiguous temporal references in the timeline"""
    resolved_timeline = {}
    temporal_ambiguities = []
    
    for temporal_context, entities in timeline.items():
        resolved_timeline[temporal_context] = []
        
        for entity in entities:
            if entity["type"] in ["blood_glucose", "blood_pressure", "weight_change"]:
                # Extract date/time information if present
                time_info = self._extract_time_info(entity["value"])
                
                if time_info:
                    entity["specific_time"] = time_info
                else:
                    # Check if this conflicts with other entries
                    conflicts = self._check_temporal_conflicts(entity, timeline)
                    if conflicts:
                        temporal_ambiguities.append({
                            "entity": entity,
                            "conflicts": conflicts,
                            "resolution": "keep most recent"  # Default resolution
                        })
            
            resolved_timeline[temporal_context].append(entity)
    
    return resolved_timeline, temporal_ambiguities

def _viterbi_sequence_labeling(self, text, spans):
    """Apply Viterbi algorithm for more accurate entity labeling"""
    # Define states (entity types)
    states = ["O", "B-MEDICATION", "I-MEDICATION", "B-SYMPTOM", "I-SYMPTOM", 
             "B-CONDITION", "I-CONDITION", "B-VITAL", "I-VITAL"]
    
    # Initial probabilities
    start_p = {s: 0.1 for s in states}
    start_p["O"] = 0.3  # Higher probability for non-entity tokens
    
    # Transition probabilities (simplified)
    trans_p = {
        "O": {"O": 0.7, "B-MEDICATION": 0.1, "B-SYMPTOM": 0.1, "B-CONDITION": 0.05, "B-VITAL": 0.05},
        "B-MEDICATION": {"I-MEDICATION": 0.8, "O": 0.2},
        "I-MEDICATION": {"I-MEDICATION": 0.8, "O": 0.2},
        "B-SYMPTOM": {"I-SYMPTOM": 0.7, "O": 0.3},
        "I-SYMPTOM": {"I-SYMPTOM": 0.7, "O": 0.3},
        "B-CONDITION": {"I-CONDITION": 0.8, "O": 0.2},
        "I-CONDITION": {"I-CONDITION": 0.7, "O": 0.3},
        "B-VITAL": {"I-VITAL": 0.8, "O": 0.2},
        "I-VITAL": {"I-VITAL": 0.7, "O": 0.3}
    }
    
    # Tokenize text
    tokens = text.split()
    
    # Emission probabilities based on spans
    emit_p = {}
    for state in states:
        emit_p[state] = {}
        for i, token in enumerate(tokens):
            if state == "O":
                # Default emission probability for "outside" state
                emit_p[state][i] = 0.8
            else:
                # Check if token is in any span
                in_span = False
                entity_type = state.split("-")[1].lower() if "-" in state else ""
                label_prefix = state.split("-")[0] if "-" in state else ""
                
                for span in spans:
                    if span["start"] <= i and i < span["end"] and span["type"].lower() == entity_type:
                        in_span = True
                        # Beginning of entity
                        if i == span["start"] and label_prefix == "B":
                            emit_p[state][i] = 0.9
                        # Inside entity
                        elif i > span["start"] and label_prefix == "I":
                            emit_p[state][i] = 0.9
                        else:
                            emit_p[state][i] = 0.1
                        break
                
                if not in_span:
                    emit_p[state][i] = 0.1
    
    # Run Viterbi algorithm
    viterbi_path = []
    viterbi_prob = []
    
    # Initialize
    curr_path = {}
    curr_prob = {}
    for state in states:
        curr_path[state] = [state]
        curr_prob[state] = start_p[state] * emit_p[state].get(0, 0.001)
    
    # Forward algorithm
    for t in range(1, len(tokens)):
        next_path = {}
        next_prob = {}
        for state in states:
            max_prob = 0
            max_state = None
            
            for prev_state in states:
                prob = curr_prob[prev_state] * trans_p.get(prev_state, {}).get(state, 0.001) * emit_p[state].get(t, 0.001)
                
                if prob > max_prob:
                    max_prob = prob
                    max_state = prev_state
            
            next_path[state] = curr_path[max_state] + [state]
            next_prob[state] = max_prob
        
        curr_path = next_path
        curr_prob = next_prob
    
    # Find optimal path
    max_final_prob = 0
    max_final_state = None
    
    for state in states:
        if curr_prob[state] > max_final_prob:
            max_final_prob = curr_prob[state]
            max_final_state = state
    
    optimal_path = curr_path[max_final_state]
    
    # Convert path to spans
    refined_spans = []
    current_span = None
    
    for i, state in enumerate(optimal_path):
        if state.startswith("B-"):
            # Close any open span
            if current_span:
                current_span["end"] = i
                refined_spans.append(current_span)
            
            # Start new span
            entity_type = state.split("-")[1].lower()
            current_span = {"type": entity_type, "start": i, "end": None}
        
        elif state.startswith("I-"):
            # Continue current span
            pass
        
        elif state == "O" and current_span:
            # Close current span
            current_span["end"] = i
            refined_spans.append(current_span)
            current_span = None
    
    # Close any open span at the end
    if current_span:
        current_span["end"] = len(tokens)
        refined_spans.append(current_span)
    
    return refined_spans

