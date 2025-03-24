# medical_summarizer_advanced.py
import os
import json
import torch
import numpy as np
import pandas as pd
import re
import ctranslate2
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field

from huggingface_hub import snapshot_download
import concurrent.futures

@dataclass
class MedicalEntity:
    """Structured representation of a medical entity with confidence"""
    type: str
    value: str
    normalized_value: Optional[str] = None
    confidence: float = 1.0
    temporal_context: str = "current"  # current, past, target, etc.
    source: str = "unknown"  # patient, provider, etc.
    position: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_negated: bool = False
    is_uncertain: bool = False
    is_hypothetical: bool = False
    is_about_family: bool = False
    measurement_unit: Optional[str] = None


class AdvancedMedicalSummarizer:
    """Advanced medical conversation summarizer with hybrid entity recognition and enhanced temporal processing"""
    
    def __init__(
            self,
            base_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            ner_model: str = "emilyalsentzer/Bio_ClinicalBERT",
            qa_model: str = "dmis-lab/biobert-base-cased-v1.1-squad",
            sentence_model: Optional[str] = None,
            device: Optional[str] = None,
            compute_type: str = "float16",
            cache_dir: Optional[str] = None,
            confidence_threshold: float = 0.65,
            use_enhanced_temporal: bool = True
        ):
        # Device setup
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        
        # CTranslate2 device mapping
        if self.device == "cuda":
            self.ct2_device = "cuda"
        elif self.device == "mps":
            self.ct2_device = "cpu"
            compute_type = "float32"
        else:
            self.ct2_device = "cpu"
        
        self.compute_type = compute_type
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "medical_summarizer")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.confidence_threshold = confidence_threshold
        self.use_enhanced_temporal = use_enhanced_temporal
        
        # Load base model with CTranslate2 acceleration
        print(f"Loading base medical language model: {base_model}")
        self._load_model_with_ctranslate2(base_model, "base_model")
        
        # Load NER model with CTranslate2 acceleration
        print(f"Loading medical NER model: {ner_model}")
        self._load_model_with_ctranslate2(ner_model, "ner_model")
        
        # Load QA model with CTranslate2 acceleration
        print(f"Loading medical QA model: {qa_model}")
        self._load_model_with_ctranslate2(qa_model, "qa_model")
        
        # Load sentence embedding model if provided
        self.sentence_model = None
        if sentence_model:
            print(f"Loading sentence embedding model: {sentence_model}")
            try:
                from sentence_transformers import SentenceTransformer
                self.sentence_model = SentenceTransformer(sentence_model).to(self.device)
            except Exception as e:
                print(f"Error loading sentence embedding model: {e}")
        
        # Initialize the NER ID to label mapping
        self.ner_id2label = {
            0: "O",  # Outside of a named entity
            1: "B-Disease",
            2: "I-Disease",
            3: "B-Chemical",
            4: "I-Chemical",
            5: "B-Symptom",
            6: "I-Symptom",
            7: "B-Medication",
            8: "I-Medication",
            9: "B-Procedure",
            10: "I-Procedure",
            11: "B-Test",
            12: "I-Test"
        }
        
        # Initialize enhanced components
        if use_enhanced_temporal:
            self.temporal_model = LearnedTemporalModel(device=self.device)
            self.temporal_graph = TemporalGraph()
        else:
            # Initialize the classic Viterbi decoder for temporal sequence modeling
            self.viterbi_decoder = self._initialize_viterbi_decoder()
        
        # Initialize the terminology mapper
        self.terminology_mapper = TerminologyMapper(cache_dir=self.cache_dir)
        
        # Initialize the vectorized contradiction detector
        self.contradiction_detector = VectorizedContradictionDetector(sentence_model=sentence_model, device=self.device)
        
        # Comprehensive medical knowledge bases
        self.medical_patterns = self._load_comprehensive_medical_patterns()
        self.medical_entity_normalizers = self._load_entity_normalizers()
        self.medical_terminologies = self._load_medical_terminologies()
        self.negation_patterns = self._load_negation_patterns()
        self.uncertainty_patterns = self._load_uncertainty_patterns()
        self.family_history_patterns = self._load_family_history_patterns()
        self.hypothetical_patterns = self._load_hypothetical_patterns()
        
        # Edge case handlers (legacy components)
        self.uncertainty_analyzer = self._initialize_uncertainty_analyzer()
        
        # Thread pool for parallelism
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    
    
    def _initialize_viterbi_decoder(self):
        """Initialize Viterbi algorithm for temporal sequence modeling"""
        class ViterbiDecoder:
            def __init__(self):
                # Define states for medical events (simplified)
                self.states = [
                    "symptom_onset", "diagnosis", "treatment_start", 
                    "treatment_ongoing", "improvement", "deterioration", 
                    "resolution", "relapse", "follow_up"
                ]
                
                # Initial probabilities for states
                self.start_p = {
                    "symptom_onset": 0.5,
                    "diagnosis": 0.3,
                    "treatment_start": 0.1,
                    "treatment_ongoing": 0.05,
                    "improvement": 0.01,
                    "deterioration": 0.01,
                    "resolution": 0.01,
                    "relapse": 0.01,
                    "follow_up": 0.01
                }
                
                # Transition probabilities between states
                self.trans_p = {
                    "symptom_onset": {"symptom_onset": 0.1, "diagnosis": 0.7, "treatment_start": 0.1, "treatment_ongoing": 0.02, "improvement": 0.02, "deterioration": 0.02, "resolution": 0.01, "relapse": 0.01, "follow_up": 0.02},
                    "diagnosis": {"symptom_onset": 0.05, "diagnosis": 0.05, "treatment_start": 0.8, "treatment_ongoing": 0.03, "improvement": 0.01, "deterioration": 0.01, "resolution": 0.01, "relapse": 0.01, "follow_up": 0.03},
                    # ... (define other transitions)
                }
                
                # Emission probabilities for observations
                self.emit_p = {}  # Will be calculated dynamically based on text
            
            def decode(self, observations, emission_probabilities):
                """Find most likely sequence of states using Viterbi algorithm"""
                self.emit_p = emission_probabilities
                
                # Initialize Viterbi algorithm
                V = [{}]  # Viterbi matrix
                path = {}  # Path tracking
                
                # Initialize base cases
                for state in self.states:
                    V[0][state] = self.start_p[state] * self.emit_p[state].get(observations[0], 0.001)
                    path[state] = [state]
                
                # Run Viterbi algorithm for observations
                for t in range(1, len(observations)):
                    V.append({})
                    new_path = {}
                    
                    for curr_state in self.states:
                        max_prob = 0
                        max_state = None
                        
                        for prev_state in self.states:
                            # Calculate probability of transition
                            trans_prob = self.trans_p.get(prev_state, {}).get(curr_state, 0.001)
                            # Calculate probability of emission
                            emit_prob = self.emit_p[curr_state].get(observations[t], 0.001)
                            # Calculate total probability
                            prob = V[t-1][prev_state] * trans_prob * emit_prob
                            
                            if prob > max_prob:
                                max_prob = prob
                                max_state = prev_state
                        
                        V[t][curr_state] = max_prob
                        new_path[curr_state] = path[max_state] + [curr_state]
                    
                    path = new_path
                
                # Find final state with highest probability
                max_prob = 0
                max_state = None
                
                for state in self.states:
                    if V[len(observations)-1][state] > max_prob:
                        max_prob = V[len(observations)-1][state]
                        max_state = state
                
                return path[max_state], max_prob
        
        return ViterbiDecoder()
    
    def _load_comprehensive_medical_patterns(self):
        """Load comprehensive patterns for medical entity recognition"""
        # Enhanced patterns with more specificity and coverage
        return {
            'blood_pressure': [
                r'\b(?:blood pressure|bp)[:\s]+(?:of|is|was|at)?[:\s]*(\d{2,3})[/](\d{2,3})\b',
                r'\b(\d{2,3})[/](\d{2,3})\s*(?:mm\s*Hg|mmHg)?\b',
                r'\bsystolic\s+(?:of|is|was|at)?[:\s]*(\d{2,3})(?:\s+diastolic\s+(?:of|is|was|at)?[:\s]*(\d{2,3}))?\b',
                r'\b(?:blood pressure|bp)[:\s]+(?:of|is|was|at)?[:\s]*(\d{2,3})\s+(?:over|by)\s+(\d{2,3})\b'
            ],
            'blood_glucose': [
                r'\b(?:blood\s+(?:sugar|glucose)|sugar|glucose)\s+(?:levels?|readings?)?(?:\s+(?:of|at|about|around|was|is))?\s+(\d{2,3})(?:\s*(?:-|to)\s*(\d{2,3}))?\s*(?:mg/dl|mmol/l)?\b',
                r'\bhemoglobin a1c|hba1c|a1c\s+(?:of|is|was|at)?[:\s]*(\d{1,2}(?:\.\d{1,2})?)\s*(?:\%|percent)?\b',
                r'\b(?:sugars?|glucose)\s+(?:of|at|was|is|were|are)\s+(\d{2,3})(?:\s*(?:-|to)\s*(\d{2,3}))?\s*(?:mg/dl|mmol/l)?\b'
            ],
            'weight_change': [
                r'(?:lost|gained|drop(?:ped)?|down|up|increase[d]?|decrease[d]?|chang(?:e|ed))\s+(?:about|around|approximately)?\s*(\d+)\s*(?:pounds|pound|lbs|lb|kilograms|kg|kgs)(?:\s+(?:in|over|during|the\s+past|the\s+last|last|past)\s+(\d+)\s+(?:weeks?|months?|days?|years?))?\b',
                r'(?:weight|mass)\s+(?:change|loss|gain)\s+(?:of)\s+(\d+)\s*(?:pounds|pound|lbs|lb|kilograms|kg|kgs)\b',
                r'(?:weight|mass)\s+(?:is|was|now|at)\s+(\d+)\s*(?:pounds|pound|lbs|lb|kilograms|kg|kgs)\b'
            ],
            # Expanded medication patterns
            'medications': [
                # Generic medication patterns
                r'(?:taking|using|on|prescribed|started|stopped|discontinued)\s+([A-Za-z]+(?:amide|ampicillin|azepam|caine|cillin|cycline|dipine|dronate|fentanil|floxacin|formin|gliptin|glitazone|ic|in|il|ide|ine|ium|oid|ol|one|opril|oxacin|parin|phil|pril|profen|sartan|semide|statin|thiazide|triptan|vir|zole|zosin|zumab)(?:\s+\d+\s*(?:mg|mcg|mL|tablet|pill|injection|patch))?)(?:\s+(?:for|to treat|to control|to manage)\s+([a-z\s]+))?\b',
                r'(?:medication|drug|treatment)\s+(?:called|named)?\s+([A-Za-z]+(?:amide|ampicillin|azepam|caine|cillin|cycline|dipine|dronate|fentanil|floxacin|formin|gliptin|glitazone|ic|in|il|ide|ine|ium|oid|ol|one|opril|oxacin|parin|phil|pril|profen|sartan|semide|statin|thiazide|triptan|vir|zole|zosin|zumab))',
                
                # Specific diabetes medications
                r'\b(?:insulin|metformin|glipizide|glyburide|glimepiride|sitagliptin|linagliptin|empagliflozin|dapagliflozin|canagliflozin|liraglutide|semaglutide|dulaglutide|exenatide|Ozempic|Trulicity|Victoza|Januvia|Jardiance|Farxiga|Invokana|Glucophage|Amaryl|DiaBeta|Glucotrol)\b(?:\s+\d+\s*(?:mg|mcg|units|u|IU))?\b',
                
                # Specific hypertension medications
                r'\b(?:lisinopril|enalapril|ramipril|captopril|benazepril|losartan|valsartan|irbesartan|candesartan|olmesartan|amlodipine|nifedipine|diltiazem|verapamil|metoprolol|atenolol|propranolol|carvedilol|bisoprolol|hydrochlorothiazide|chlorthalidone|indapamide|furosemide|spironolactone|eplerenone|clonidine|hydralazine|minoxidil|Norvasc|Cardizem|Procardia|Toprol|Tenormin|Inderal|Coreg|Zebeta|HydroDIURIL|Lasix|Aldactone|Inspra|Catapres)\b(?:\s+\d+\s*(?:mg|mcg))?\b',
                
                # Common class names
                r'\b(?:ACE inhibitor|ARB|beta blocker|calcium channel blocker|diuretic|statin|anticoagulant|antiplatelet|NSAID)\b'
            ],
            
            # Add more patterns for symptoms and conditions
            'symptoms': [
                r'\b(?:experienc(?:ed|ing)|hav(?:e|ing)|report(?:s|ed|ing)|complain(?:s|ed|ing) of|feeling|felt|suffering from)\s+([a-z\s]+(?:pain|ache|discomfort|nausea|vomiting|fatigue|tiredness|exhaustion|weakness|dizziness|lightheadedness|headache|cough|fever|chills|sweating|numbness|tingling|swelling|rash|itch|burning|shortness of breath|sob|dyspnea))\b',
                r'\b(?:chest pain|chest discomfort|chest tightness|chest pressure|shortness of breath|difficulty breathing|sob|dyspnea|palpitations|heart racing|irregular heartbeat|edema|swelling|nausea|vomiting|headache|dizziness|lightheadedness|fatigue|weakness|abdominal pain|back pain|joint pain|muscle pain)\b'
            ],
            
            'chronic_conditions': [
                r'\b(?:diagnosed with|history of|has|have|having|suffer(?:s|ing) from|manages|treating|treatment for)\s+([a-z\s]+(?:diabetes|hypertension|high blood pressure|copd|asthma|arthritis|cancer|heart disease|heart failure|coronary artery disease|cad|kidney disease|ckd|liver disease|hepatitis|cirrhosis|depression|anxiety|thyroid|hypothyroidism|hyperthyroidism|gerd|ibs|crohn|colitis))\b',
                r'\b(?:type (?:1|2|one|two|I|II) diabetes|dm1|dm2|t1dm|t2dm|hypothyroidism|hyperthyroidism|grave\'s disease|hashimoto\'s|addison\'s disease|cushing\'s syndrome|rheumatoid arthritis|osteoarthritis|osteoporosis|copd|chronic bronchitis|emphysema|asthma|heart failure|chf|coronary artery disease|cad|atrial fibrillation|afib|hypertension|htn|high blood pressure|elevated bp|stroke|cva|transient ischemic attack|tia|peripheral artery disease|pad|chronic kidney disease|ckd|end-stage renal disease|esrd|dialysis|cirrhosis|hepatitis|fatty liver disease|nafld|nash|gerd|acid reflux|irritable bowel syndrome|ibs|crohn\'s disease|ulcerative colitis|uc|depression|anxiety|bipolar disorder|schizophrenia|alzheimer\'s|dementia|parkinson\'s disease|multiple sclerosis|ms|epilepsy|seizure disorder|fibromyalgia|chronic fatigue syndrome|cfs|sleep apnea|osa|gout|lupus|sle|hiv|aids|cancer)\b'
            ]
        }
    
    def _load_negation_patterns(self):
        """Load patterns to detect negated medical concepts"""
        return [
            r'no\s+(?:signs?|symptoms?|evidence|indication)\s+of\s+(\w+)',
            r'(?:denies|denied|deny)\s+(?:any\s+)?([^,.;:]+)',
            r'not\s+(?:having|experiencing|complaining\s+of)\s+([^,.;:]+)',
            r'(?:hasn\'t|has\s+not|have\s+not|haven\'t)\s+(?:had|experienced|noticed)\s+([^,.;:]+)',
            r'never\s+(?:had|experienced|noticed)\s+([^,.;:]+)',
            r'([^,.;:]+)\s+(?:was|were|is|are)\s+(?:not|never)\s+(?:present|noted|observed|found|documented)',
            r'no\s+([^,.;:]+)',
            r'without\s+([^,.;:]+)',
            r'free\s+of\s+([^,.;:]+)',
            r'negative\s+for\s+([^,.;:]+)'
        ]
    
    def _load_uncertainty_patterns(self):
        """Load patterns to detect uncertainty in medical statements"""
        return [
            r'(?:might|may|could|would|can|possibly|perhaps|maybe|potentially)\s+(?:have|be|indicate|suggest|mean|show|result\s+in|lead\s+to|cause|develop)\s+([^,.;:]+)',
            r'(?:not\s+sure|uncertain|unsure|don\'t\s+know|unclear|possible|probable|likely|unlikely|question\s+of|questionable|suspected|suspicious\s+for)\s+(?:if|whether|about|that)?\s+([^,.;:]+)',
            r'(?:suspicion|possibility|probability|chance)\s+of\s+([^,.;:]+)',
            r'rule\s+out\s+([^,.;:]+)',
            r'(?:consistent|compatible|suggestive)\s+with\s+([^,.;:]+)',
            r'(?:think|believe|suspect|assume|presume|impression)\s+(?:that|is|of)?\s+([^,.;:]+)',
            r'(?:if|when)\s+([^,.;:]+)'
        ]
    
    def _load_family_history_patterns(self):
        """Load patterns to detect family history references"""
        return [
            r'(?:family|family\'s|families)\s+(?:history|histories|member|members|relative|relatives)\s+(?:of|with|has|have|had)\s+([^,.;:]+)',
            r'(?:mother|father|parent|parents|brother|sister|sibling|siblings|son|daughter|child|children|grandmother|grandfather|grandparent|grandparents|aunt|uncle|cousin|cousins)(?:\'s)?\s+(?:has|have|had|diagnosed\s+with|history\s+of)\s+([^,.;:]+)',
            r'(?:history|diagnosed)\s+(?:of|with)\s+([^,.;:]+)\s+(?:in|among)\s+(?:family|family\'s|families|mother|father|parent|parents|brother|sister|sibling|siblings|son|daughter|child|children|grandmother|grandfather|grandparent|grandparents|aunt|uncle|cousin|cousins)',
            r'(?:runs|run)\s+in\s+(?:the|my|his|her|their)\s+family',
            r'(?:genetically|genetic|hereditary|inherited|familial)\s+([^,.;:]+)'
        ]
    
    def _load_hypothetical_patterns(self):
        """Load patterns to detect hypothetical discussions"""
        return [
            r'(?:if|when|should|in\s+case|imagine\s+that|let\'s\s+say|suppose|consider|assuming)\s+([^,.;:]+)',
            r'(?:hypothetically|theoretically|in\s+theory)\s+([^,.;:]+)',
            r'(?:possible|potential|future)\s+(?:scenario|situation|case|complication|side\s+effect|outcome)\s+(?:of|is|includes|involves|would\s+be)?\s+([^,.;:]+)',
            r'(?:may|might|could)\s+(?:lead\s+to|result\s+in|cause|develop|experience|have|get)\s+([^,.;:]+)',
            r'(?:risk|risks|chance|chances|probability|likelihood)\s+(?:of|for)\s+([^,.;:]+)',
            r'(?:prevent|prevention|avoid|preventing)\s+([^,.;:]+)',
            r'(?:recommendation|recommendations|suggest|suggested|advise|advised|recommend|recommended|counsel|counseled|encourage|encouraged)\s+(?:to|that|for|against)?\s+([^,.;:]+)',
            r'(?:plan|planning|plans|aim|aiming|aims|goal|goals)\s+(?:to|for)?\s+([^,.;:]+)'
        ]
    
    def _load_entity_normalizers(self):
        """Load enhanced entity normalizers with unit standardization"""
        return {
            'blood_pressure': lambda value: self._normalize_blood_pressure(value),
            'blood_glucose': lambda value: self._normalize_blood_glucose(value),
            'weight_change': lambda value: self._normalize_weight(value),
            'medications': lambda value: self._normalize_medication(value),
            'symptoms': lambda value: self._normalize_symptom(value),
            'chronic_conditions': lambda value: self._normalize_condition(value)
        }
    
    def _normalize_blood_pressure(self, value):
        """Normalize blood pressure values to standard format"""
        # Extract systolic and diastolic values
        bp_match = re.search(r'(\d{2,3})[/](\d{2,3})', value)
        if bp_match:
            systolic = int(bp_match.group(1))
            diastolic = int(bp_match.group(2))
            return f"{systolic}/{diastolic} mmHg"
        
        # Check for "X over Y" format
        bp_over_match = re.search(r'(\d{2,3})\s+(?:over|by)\s+(\d{2,3})', value, re.IGNORECASE)
        if bp_over_match:
            systolic = int(bp_over_match.group(1))
            diastolic = int(bp_over_match.group(2))
            return f"{systolic}/{diastolic} mmHg"
        
        # Try to extract individual components
        systolic_match = re.search(r'systolic\s+(?:of|is|was|at)?[:\s]*(\d{2,3})', value, re.IGNORECASE)
        diastolic_match = re.search(r'diastolic\s+(?:of|is|was|at)?[:\s]*(\d{2,3})', value, re.IGNORECASE)
        
        if systolic_match and diastolic_match:
            systolic = int(systolic_match.group(1))
            diastolic = int(diastolic_match.group(1))
            return f"{systolic}/{diastolic} mmHg"
        elif systolic_match:
            systolic = int(systolic_match.group(1))
            return f"{systolic}/? mmHg"
        elif diastolic_match:
            diastolic = int(diastolic_match.group(1))
            return f"?/{diastolic} mmHg"
        
        return value
    
    def _normalize_blood_glucose(self, value):
        """Normalize blood glucose values with unit standardization"""
        # Check for A1C values first
        a1c_match = re.search(r'(?:hba1c|a1c).*?(\d{1,2}(?:\.\d{1,2})?)', value, re.IGNORECASE)
        if a1c_match:
            a1c_value = float(a1c_match.group(1))
            return f"HbA1c: {a1c_value}%"
        
        # Extract glucose value
        glucose_match = re.search(r'(\d{2,3})(?:\s*(?:-|to)\s*(\d{2,3}))?', value)
        if glucose_match:
            if glucose_match.group(2):  # Range
                start = int(glucose_match.group(1))
                end = int(glucose_match.group(2))
                return f"{start}-{end} mg/dL"
            else:  # Single value
                glucose = int(glucose_match.group(1))
                return f"{glucose} mg/dL"
        
        return value
    
    def _normalize_weight(self, value):
        """Normalize weight values with unit standardization"""
        # Extract weight change
        change_match = re.search(r'(?:lost|gained|drop(?:ped)?|down|up|increase[d]?|decrease[d]?|chang(?:e|ed))\s+(?:about|around|approximately)?\s*(\d+)\s*(?:pounds|pound|lbs|lb|kilograms|kg|kgs)', value, re.IGNORECASE)
        
        if change_match:
            amount = int(change_match.group(1))
            
            # Determine direction
            direction = ""
            if re.search(r'lost|drop(?:ped)?|down|decrease[d]?', value, re.IGNORECASE):
                direction = "lost"
            elif re.search(r'gained|up|increase[d]?', value, re.IGNORECASE):
                direction = "gained"
            
            # Determine unit
            unit = "lbs"
            if re.search(r'kilograms|kg|kgs', value, re.IGNORECASE):
                unit = "kg"
            
            # Determine time period
            time_period = ""
            time_match = re.search(r'(?:in|over|during|the\s+past|the\s+last|last|past)\s+(\d+)\s+(weeks?|months?|days?|years?)', value, re.IGNORECASE)
            if time_match:
                time_amount = time_match.group(1)
                time_unit = time_match.group(2)
                time_period = f" over {time_amount} {time_unit}"
            
            return f"{direction} {amount} {unit}{time_period}"
        
        # Check for current weight
        current_weight_match = re.search(r'(?:weight|mass)\s+(?:is|was|now|at)\s+(\d+)\s*(?:pounds|pound|lbs|lb|kilograms|kg|kgs)', value, re.IGNORECASE)
        if current_weight_match:
            weight = int(current_weight_match.group(1))
            
            # Determine unit
            unit = "lbs"
            if re.search(r'kilograms|kg|kgs', value, re.IGNORECASE):
                unit = "kg"
            
            return f"weight: {weight} {unit}"
        
        return value
    
    def _normalize_medication(self, value):
        """Normalize medication names with doses"""
        # Extract medication name
        med_match = re.search(r'([A-Za-z]+(?:amide|ampicillin|azepam|caine|cillin|cycline|dipine|dronate|fentanil|floxacin|formin|gliptin|glitazone|ic|in|il|ide|ine|ium|oid|ol|one|opril|oxacin|parin|phil|pril|profen|sartan|semide|statin|thiazide|triptan|vir|zole|zosin|zumab))', value, re.IGNORECASE)
        
        if med_match:
            medication = med_match.group(1)
            # Capitalize first letter only
            medication = medication[0].upper() + medication[1:].lower()
            
            # Extract dosage if present
            dose_match = re.search(r'(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?|tabs?|caps?)', value, re.IGNORECASE)
            if dose_match:
                dose = dose_match.group(1)
                unit = dose_match.group(2).lower()
                return f"{medication} {dose} {unit}"
            
            return medication
        
        # Check for specific diabetes medications
        diabetes_meds = [
            "insulin", "metformin", "glipizide", "glyburide", "glimepiride", 
            "sitagliptin", "linagliptin", "empagliflozin", "dapagliflozin", 
            "canagliflozin", "liraglutide", "semaglutide", "dulaglutide", 
            "exenatide", "Ozempic", "Trulicity", "Victoza", "Januvia", 
            "Jardiance", "Farxiga", "Invokana", "Glucophage", "Amaryl", 
            "DiaBeta", "Glucotrol"
        ]
        
        for med in diabetes_meds:
            if re.search(r'\b' + re.escape(med) + r'\b', value, re.IGNORECASE):
                # Extract dosage if present
                dose_match = re.search(r'(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?|u|IU)', value, re.IGNORECASE)
                if dose_match:
                    dose = dose_match.group(1)
                    unit = dose_match.group(2).lower()
                    return f"{med.capitalize()} {dose} {unit}"
                return med.capitalize()
        
        return value
    
    def _normalize_symptom(self, value):
        """Normalize symptom descriptions"""
        # Map common symptom variations
        symptom_mappings = {
            r"shortness\s+of\s+breath": "shortness of breath",
            r"sob": "shortness of breath",
            r"dyspnea": "shortness of breath",
            r"breathing\s+difficult(y|ies)": "shortness of breath",
            r"trouble\s+breathing": "shortness of breath",
            r"chest\s+pain": "chest pain",
            r"chest\s+discomfort": "chest pain",
            r"chest\s+tightness": "chest pain",
            r"chest\s+pressure": "chest pain",
            r"headache": "headache",
            r"dizz(y|iness)": "dizziness",
            r"lightheaded(ness)?": "dizziness",
            r"nause(a|ated)": "nausea",
            r"vomit(ing)?": "vomiting",
            r"fatigue": "fatigue",
            r"tired(ness)?": "fatigue",
            r"exhaustion": "fatigue",
            r"weak(ness)?": "weakness",
            r"fever": "fever",
            r"elevated\s+temperature": "fever",
            r"high\s+temperature": "fever",
            r"cough(ing)?": "cough",
        }
        
        for pattern, normalized in symptom_mappings.items():
            if re.search(pattern, value, re.IGNORECASE):
                return normalized
        
        return value
    
    def _normalize_condition(self, value):
        """Normalize chronic condition descriptions"""
        # Map common condition variations
        condition_mappings = {
            r"diabetes\s+(?:mellitus|type|t)?\s*(?:2|two|ii)": "type 2 diabetes",
            r"(?:t2|type\s*2)\s*(?:dm|diabetes)": "type 2 diabetes",
            r"diabetes\s+(?:mellitus|type|t)?\s*(?:1|one|i)": "type 1 diabetes",
            r"(?:t1|type\s*1)\s*(?:dm|diabetes)": "type 1 diabetes",
            r"high\s+blood\s+pressure": "hypertension",
            r"htn": "hypertension",
            r"elevated\s+bp": "hypertension",
            r"copd": "COPD",
            r"chronic\s+obstructive\s+pulmonary\s+disease": "COPD",
            r"coronary\s+artery\s+disease": "coronary artery disease",
            r"cad": "coronary artery disease",
            r"heart\s+failure": "heart failure",
            r"congestive\s+heart\s+failure": "heart failure",
            r"chf": "heart failure",
            r"chronic\s+kidney\s+disease": "chronic kidney disease",
            r"ckd": "chronic kidney disease",
            r"kidney\s+failure": "chronic kidney disease",
            r"renal\s+failure": "chronic kidney disease",
            r"atrial\s+fibrillation": "atrial fibrillation",
            r"afib": "atrial fibrillation",
            r"a-fib": "atrial fibrillation",
        }
        
        for pattern, normalized in condition_mappings.items():
            if re.search(pattern, value, re.IGNORECASE):
                return normalized
        
        return value
    
    def _initialize_uncertainty_analyzer(self):
        """Initialize uncertainty analyzer for ambiguous medical statements"""
        class UncertaintyAnalyzer:
            def __init__(self):
                # Define uncertainty markers and their weights
                self.uncertainty_markers = {
                    "high": [
                        "might", "may", "could", "would", "possibly", "perhaps", "maybe", "potentially",
                        "not sure", "uncertain", "unsure", "don't know", "unclear", "possible", "probable",
                        "suspicion", "possibility", "probability", "chance"
                    ],
                    "medium": [
                        "likely", "consistent with", "compatible with", "suggestive of", "suspected",
                        "think", "believe", "impression"
                    ],
                    "low": [
                        "probably", "appears", "seems", "suggests", "indicates", "is indicative of"
                    ]
                }
                
                # Weights for uncertainty levels
                self.uncertainty_weights = {
                    "high": 0.8,
                    "medium": 0.5,
                    "low": 0.3
                }
            
            def analyze_uncertainty(self, text, entity):
                """Analyze uncertainty level for an entity based on context"""
                if not entity.position or not text:
                    return 0.0
                
                start, end = entity.position
                # Extract context (up to 10 words before the entity)
                context_start = max(0, text.rfind(".", 0, start))
                if context_start == -1:
                    context_start = max(0, start - 100)
                
                context_end = min(len(text), text.find(".", end))
                if context_end == -1:
                    context_end = min(len(text), end + 100)
                
                context = text[context_start:context_end]
                
                # Check for uncertainty markers in context
                uncertainty_score = 0.0
                
                for level, markers in self.uncertainty_markers.items():
                    for marker in markers:
                        if re.search(r'\b' + re.escape(marker) + r'\b', context, re.IGNORECASE):
                            uncertainty_score = max(uncertainty_score, self.uncertainty_weights[level])
                
                return uncertainty_score
            
            def classify_uncertain_entities(self, entities, text):
                """Classify entities by uncertainty level based on context"""
                for entity_type, entity_list in entities.items():
                    for entity in entity_list:
                        uncertainty_score = self.analyze_uncertainty(text, entity)
                        entity.is_uncertain = uncertainty_score > 0.3
                        entity.metadata["uncertainty_score"] = uncertainty_score
                
                return entities
        
        return UncertaintyAnalyzer()
    
    def set_terminology_mapping(self, terminology_path):
        """Set custom terminology mapping"""
        self.terminology_mapper = TerminologyMapper(terminology_path=terminology_path, cache_dir=self.cache_dir)
    
    def set_temporal_model(self, model_path):
        """Set custom temporal model"""
        self.temporal_model = LearnedTemporalModel(model_path=model_path, device=self.device)
    
    def process_medical_transcript(
            self,
            transcript_data: Union[str, Dict, List, pd.DataFrame],
            text_column: str = "transcription",
            speaker_column: str = "speaker",
            output_path: Optional[str] = None,
            chunk_size: int = 5000,
            chunk_overlap: int = 500
        ) -> Dict[str, Any]:
        """Process a medical transcript with advanced analysis and summarization"""
        # Start timing
        start_time = time.time()
        processing_stats = {}
        
        # Process different input formats
        df = self._preprocess_transcript(transcript_data, text_column, speaker_column)
        preprocessing_time = time.time() - start_time
        processing_stats["preprocessing_time"] = preprocessing_time
        
        # Create text representations
        full_text, care_manager_text, patient_text, turn_by_turn_text = self._create_text_representations(df, text_column, speaker_column)
        text_extraction_time = time.time() - start_time - preprocessing_time
        processing_stats["text_extraction_time"] = text_extraction_time
        
        # Extract medical entities
        entity_extraction_start = time.time()
        if len(full_text) > chunk_size:
            entities = self._extract_entities_chunked(full_text, chunk_size, chunk_overlap)
        else:
            entities = self._extract_medical_entities(full_text)
        entity_extraction_time = time.time() - entity_extraction_start
        processing_stats["entity_extraction_time"] = entity_extraction_time
        
        # Process entities for negation, uncertainty, family history, etc.
        entity_processing_start = time.time()
        entities = self._process_entity_modifiers(entities, full_text)
        entity_processing_time = time.time() - entity_processing_start
        processing_stats["entity_processing_time"] = entity_processing_time
        
        # Detect contradictions
        contradiction_start = time.time()
        contradictions = self.contradiction_detector.detect_contradictions(entities, full_text)
        contradiction_time = time.time() - contradiction_start
        processing_stats["contradiction_detection_time"] = contradiction_time
        
        # Analyze uncertainty
        uncertainty_start = time.time()
        entities = self.uncertainty_analyzer.classify_uncertain_entities(entities, full_text)
        uncertainty_time = time.time() - uncertainty_start
        processing_stats["uncertainty_analysis_time"] = uncertainty_time
        
        # Construct timeline
        timeline_start = time.time()
        if self.use_enhanced_temporal:
            # Build temporal graph
            temporal_graph = TemporalGraph()
            temporal_graph.build_from_entities(entities, full_text)
            structured_timeline = temporal_graph.get_timeline()
            
            # Convert to expected format
            timeline = {state: [] for state in ["past_history", "recent_past", "current", 
                                              "immediate_future", "distant_future", "unknown"]}
            
            for temporal_state, events in structured_timeline.items():
                if temporal_state not in timeline:
                    timeline[temporal_state] = []
                
                for event in events:
                    timeline[temporal_state].append({
                        "type": event["type"],
                        "value": event["value"],
                        "normalized_value": event["normalized_value"],
                        "confidence": 0.8  # Default confidence 
                    })
        else:
            timeline = self._construct_timeline(entities, full_text)
        
        timeline_time = time.time() - timeline_start
        processing_stats["timeline_construction_time"] = timeline_time
        
        # Extract key sections for structured summary
        sections_start = time.time()
        sections = self._extract_key_sections(entities, timeline, contradictions, full_text, care_manager_text, patient_text)
        sections_time = time.time() - sections_start
        processing_stats["sections_extraction_time"] = sections_time
        
        # Generate narrative summary
        summary_start = time.time()
        narrative_summary = self._generate_narrative_summary(sections, entities, contradictions)
        summary_time = time.time() - summary_start
        processing_stats["summary_generation_time"] = summary_time
        
        # Generate SOAP note
        soap_start = time.time()
        soap_note = self._generate_soap_note(sections, entities, timeline)
        soap_time = time.time() - soap_start
        processing_stats["soap_note_generation_time"] = soap_time
        
        # Total processing time
        total_time = time.time() - start_time
        processing_stats["total_processing_time"] = total_time
        
        # Complete result
        result = {
            "narrative_summary": narrative_summary,
            "structured_summary": sections,
            "soap_note": soap_note,
            "timeline": timeline,
            "entities": {
                entity_type: [self._entity_to_dict(e) for e in entity_list]
                for entity_type, entity_list in entities.items()
            },
            "contradictions": contradictions,
            "processing_stats": processing_stats
        }
        
        # Save to output file if requested
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
        
        return result
    
    def _preprocess_transcript(self, transcript_data, text_column, speaker_column):
        """Process transcript into a standard DataFrame format"""
        if isinstance(transcript_data, str):
            if transcript_data.endswith('.json'):
                with open(transcript_data, 'r') as f:
                    data = json.load(f)
                if 'segments' in data:
                    df = pd.DataFrame(data['segments'])
                else:
                    df = pd.DataFrame(data)
            elif transcript_data.endswith('.csv'):
                df = pd.DataFrame(pd.read_csv(transcript_data))
            else:
                raise ValueError("Invalid input file format. Only JSON and CSV files are supported")
        elif isinstance(transcript_data, list):
            df = pd.DataFrame(transcript_data)
        elif isinstance(transcript_data, pd.DataFrame):
            df = transcript_data
        elif isinstance(transcript_data, dict) and 'segments' in transcript_data:
            df = pd.DataFrame(transcript_data['segments'])
        else:
            raise ValueError("Invalid input data format")
        
        # Filter valid transcriptions
        df = df[df[text_column].notna() & (df[text_column].astype(str).str.strip() != "")]
        
        return df
    
    def _create_text_representations(self, df, text_column, speaker_column):
        """Create different text representations of the conversation"""
        # Full conversation text
        full_text = " ".join(df[text_column].astype(str).tolist())
        
        # Speaker-specific text
        care_manager_df = df[df[speaker_column] == "SPEAKER_00"]
        patient_df = df[df[speaker_column] == "SPEAKER_01"]
        
        care_manager_text = " ".join(care_manager_df[text_column].astype(str).tolist())
        patient_text = " ".join(patient_df[text_column].astype(str).tolist())
        
        # Turn-by-turn conversation with speaker labels
        turn_by_turn = []
        for _, row in df.iterrows():
            speaker_label = "Care Manager: " if row[speaker_column] == "SPEAKER_00" else "Patient: "
            turn_by_turn.append(f"{speaker_label}{row[text_column]}")
        
        turn_by_turn_text = "\n".join(turn_by_turn)
        
        return full_text, care_manager_text, patient_text, turn_by_turn_text
    
    def _extract_entities_chunked(self, text, chunk_size, chunk_overlap):
        """Extract entities from text using chunk-based processing for scalability"""
        if len(text) <= chunk_size:
            # For short texts, process directly
            return self._extract_medical_entities(text)
        
        # Split text into overlapping chunks
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk_start = i
            chunk_end = min(i + chunk_size, len(text))
            chunk = text[chunk_start:chunk_end]
            chunks.append((chunk_start, chunk_end, chunk))
        
        # Process each chunk in parallel
        all_entities = {}
        
        # Use ThreadPoolExecutor for parallelization
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(chunks))) as executor:
            # Submit chunk processing tasks
            future_to_chunk = {
                executor.submit(self._process_chunk, chunk_start, chunk_end, chunk): 
                (chunk_start, chunk_end, chunk) for chunk_start, chunk_end, chunk in chunks
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_start, chunk_end, chunk = future_to_chunk[future]
                try:
                    chunk_entities = future.result()
                    
                    # Merge results from each chunk
                    for entity_type, entity_list in chunk_entities.items():
                        if entity_type not in all_entities:
                            all_entities[entity_type] = []
                        all_entities[entity_type].extend(entity_list)
                except Exception as e:
                    print(f"Error processing chunk {chunk_start}-{chunk_end}: {e}")
        
        # Deduplicate entities from overlapping regions
        all_entities = self._deduplicate_chunked_entities(all_entities)
        
        return all_entities
    
    def _process_chunk(self, chunk_start, chunk_end, chunk):
        """Process a single text chunk and extract entities"""
        # Extract entities from this chunk
        chunk_entities = self._extract_medical_entities(chunk)
        
        # Adjust entity positions to match original text
        for entity_type, entity_list in chunk_entities.items():
            for entity in entity_list:
                if entity.position:
                    start, end = entity.position
                    entity.position = (start + chunk_start, end + chunk_start)
        
        return chunk_entities
    
    def _deduplicate_chunked_entities(self, entities):
        """Deduplicate entities from overlapping chunks"""
        deduplicated = {}
        
        for entity_type, entity_list in entities.items():
            if not entity_list:
                deduplicated[entity_type] = []
                continue
            
            # Sort by position
            sorted_entities = sorted(entity_list, 
                                   key=lambda e: e.position[0] if e.position else float('inf'))
            
            # Deduplicate
            unique_entities = []
            
            for entity in sorted_entities:
                # Skip if no position (can't deduplicate)
                if not entity.position:
                    unique_entities.append(entity)
                    continue
                
                # Check if this entity overlaps with any already added
                duplicate = False
                for i, existing in enumerate(unique_entities):
                    if not existing.position:
                        continue
                    
                    # Check for substantial overlap
                    if self._entities_overlap(entity, existing):
                        duplicate = True
                        
                        # Keep the one with higher confidence
                        if entity.confidence > existing.confidence:
                            unique_entities[i] = entity
                        
                        break
                
                if not duplicate:
                    unique_entities.append(entity)
            
            deduplicated[entity_type] = unique_entities
        
        return deduplicated
    
    def _entities_overlap(self, entity1, entity2):
        """Check if two entities have substantial position overlap"""
        if not entity1.position or not entity2.position:
            return False
        
        start1, end1 = entity1.position
        start2, end2 = entity2.position
        
        # Check if positions overlap
        if start1 <= end2 and start2 <= end1:
            overlap_length = min(end1, end2) - max(start1, start2)
            entity1_length = end1 - start1
            entity2_length = end2 - start2
            
            # Consider substantial overlap if > 50% of either entity
            if overlap_length > 0.5 * min(entity1_length, entity2_length):
                return True
        
        return False
    
    def _extract_medical_entities(self, text):
        """Extract medical entities using hybrid approach with model-based and pattern-based extraction"""
        entities = {}
        
        # Model-based extraction (if using hybrid approach and model is available)
        if hasattr(self, 'ner_model_ct2') and self.ner_model_using_ct2:
            # Process text with NER model
            model_entities = self._extract_entities_with_medical_bert(text)
            
            # Add model-extracted entities to our collection
            for entity_type, entity_list in model_entities.items():
                if entity_type not in entities:
                    entities[entity_type] = []
                entities[entity_type].extend(entity_list)
        
        # Pattern-based extraction
        for entity_type, patterns in self.medical_patterns.items():
            if entity_type not in entities:
                entities[entity_type] = []
            
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    value = match.group(0)
                    position = match.span()
                    
                    # Normalize value if normalizer exists
                    normalized_value = None
                    if entity_type in self.medical_entity_normalizers:
                        normalized_value = self.medical_entity_normalizers[entity_type](value)
                    
                    entity = MedicalEntity(
                        type=entity_type,
                        value=value,
                        normalized_value=normalized_value,
                        confidence=0.7,  # Base confidence for pattern matches
                        position=position,
                        source="pattern_match"
                    )
                    
                    # Extract context for metadata
                    context_start = max(0, position[0] - 100)
                    context_end = min(len(text), position[1] + 100)
                    entity.metadata["context"] = text[context_start:context_end]
                    
                    # Map to standard terminology if available
                    entity = self.terminology_mapper.map_entity(entity)
                    
                    entities[entity_type].append(entity)
        
        # Apply medical terminology mapping for better entity consolidation
        entities = self._apply_medical_terminology_mapping(entities)
        
        # Merge similar entities
        entities = self._merge_similar_entities(entities)
        
        return entities
    
    def _extract_entities_with_medical_bert(self, text, chunk_size=512, overlap=50):
        """Extract medical entities using BioClinicalBERT"""
        entities = {}
        
        # Process text in chunks
        text_chunks = []
        positions = []
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk_start = i
            chunk_end = min(i + chunk_size, len(text))
            chunk = text[chunk_start:chunk_end]
            
            text_chunks.append(chunk)
            positions.append((chunk_start, chunk_end))
        
        # Tokenize all chunks
        inputs = self.ner_model_tokenizer(
            text_chunks,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=chunk_size,
            return_offsets_mapping=True
        )
        
        # Get predictions for all chunks using our CTranslate2 model
        for i, (chunk, (chunk_start, chunk_end)) in enumerate(zip(text_chunks, positions)):
            # Get encoding for this chunk
            chunk_inputs = {k: v[i:i+1] for k, v in inputs.items()}
            
            # Convert to format expected by CTranslate2
            ct2_inputs = {
                "input_ids": chunk_inputs["input_ids"].numpy(),
                "attention_mask": chunk_inputs["attention_mask"].numpy()
            }
            
            # Get model output
            outputs = self.ner_model_ct2.forward(ct2_inputs)
            logits = torch.tensor(outputs[0])
            
            # Get predicted tokens
            predictions = torch.argmax(logits, dim=-1)
            
            # Extract entities
            current_entity = None
            offset_mapping = chunk_inputs["offset_mapping"][0]
            
            for token_idx, pred in enumerate(predictions[0]):
                # Skip padding and special tokens
                if chunk_inputs["attention_mask"][0, token_idx] == 0:
                    continue
                    
                token_start, token_end = offset_mapping[token_idx].tolist()
                if token_start == token_end:
                    continue  # Skip special tokens
                    
                # Adjust token positions to the original text
                token_start += chunk_start
                token_end += chunk_start
                    
                # Get the entity type
                entity_type = self.ner_id2label.get(pred.item(), "O")
                
                if entity_type.startswith("B-"):  # Beginning of entity
                    if current_entity is not None:
                        # Add previous entity to results
                        entity_type = current_entity["type"].replace("B-", "").replace("I-", "")
                        
                        if entity_type not in entities:
                            entities[entity_type] = []
                            
                        value = text[current_entity["start"]:current_entity["end"]]
                        entity = MedicalEntity(
                            type=entity_type,
                            value=value,
                            confidence=current_entity["score"],
                            position=(current_entity["start"], current_entity["end"]),
                            source="model_extraction"
                        )
                        
                        # Extract context
                        context_start = max(0, current_entity["start"] - 100)
                        context_end = min(len(text), current_entity["end"] + 100)
                        entity.metadata["context"] = text[context_start:context_end]
                        
                        entities[entity_type].append(entity)
                    
                    # Start new entity
                    current_entity = {
                        "type": entity_type,
                        "start": token_start,
                        "end": token_end,
                        "score": logits[0, token_idx, pred.item()].item()
                    }
                    
                elif entity_type.startswith("I-") and current_entity is not None:
                    # Continue current entity if types match
                    if entity_type.replace("I-", "B-") == current_entity["type"]:
                        current_entity["end"] = token_end
                        # Update score as average
                        current_entity["score"] = (current_entity["score"] + 
                                                  logits[0, token_idx, pred.item()].item()) / 2
                        
                elif entity_type == "O" and current_entity is not None:
                    # End of entity
                    entity_type = current_entity["type"].replace("B-", "").replace("I-", "")
                    
                    if entity_type not in entities:
                        entities[entity_type] = []
                        
                    value = text[current_entity["start"]:current_entity["end"]]
                    entity = MedicalEntity(
                        type=entity_type,
                        value=value,
                        confidence=current_entity["score"],
                        position=(current_entity["start"], current_entity["end"]),
                        source="model_extraction"
                    )
                    
                    # Extract context
                    context_start = max(0, current_entity["start"] - 100)
                    context_end = min(len(text), current_entity["end"] + 100)
                    entity.metadata["context"] = text[context_start:context_end]
                    
                    entities[entity_type].append(entity)
                    current_entity = None
                    
        # Handle any final entity
        if current_entity is not None:
            entity_type = current_entity["type"].replace("B-", "").replace("I-", "")
            
            if entity_type not in entities:
                entities[entity_type] = []
                
            value = text[current_entity["start"]:current_entity["end"]]
            entity = MedicalEntity(
                type=entity_type,
                value=value,
                confidence=current_entity["score"],
                position=(current_entity["start"], current_entity["end"]),
                source="model_extraction"
            )
            
            # Extract context
            context_start = max(0, current_entity["start"] - 100)
            context_end = min(len(text), current_entity["end"] + 100)
            entity.metadata["context"] = text[context_start:context_end]
            
            entities[entity_type].append(entity)
        
        # Map entities to standard terminologies
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                entity = self.terminology_mapper.map_entity(entity)
        
        return entities
    
    def _process_entity_modifiers(self, entities, text):
        """Process entity modifiers like negation, uncertainty, etc."""
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                # Skip entities without context
                if "context" not in entity.metadata:
                    continue
                
                context = entity.metadata["context"]
                
                # Check for negation
                entity.is_negated = self._is_negated(context, entity.value)
                
                # Check for uncertainty
                entity.is_uncertain = self._is_uncertain(context, entity.value)
                
                # Check for family history references
                entity.is_about_family = self._is_about_family(context, entity.value)
                
                # Check for hypothetical discussions
                entity.is_hypothetical = self._is_hypothetical(context, entity.value)
                
                # Adjust confidence based on modifiers
                if entity.is_negated or entity.is_uncertain or entity.is_hypothetical:
                    entity.confidence *= 0.8
                if entity.is_about_family:
                    entity.confidence *= 0.9
        
        return entities
    
    def _is_negated(self, context, value):
        """Check if an entity is negated in context"""
        # Look for negation patterns before the entity
        value_pos = context.lower().find(value.lower())
        if value_pos == -1:
            return False
        
        pre_context = context[:value_pos].lower()
        
        # Check common negation patterns
        for pattern in self.negation_patterns:
            if re.search(pattern, pre_context, re.IGNORECASE):
                return True
        
        # Check for "no" + entity
        if re.search(r'no\s+' + re.escape(value), context, re.IGNORECASE):
            return True
        
        return False
    
    def _is_uncertain(self, context, value):
        """Check if an entity is mentioned with uncertainty"""
        for pattern in self.uncertainty_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                return True
        return False
    
    def _is_about_family(self, context, value):
        """Check if an entity refers to family history rather than patient"""
        for pattern in self.family_history_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                return True
        return False
    
    def _is_hypothetical(self, context, value):
        """Check if an entity is mentioned in a hypothetical context"""
        for pattern in self.hypothetical_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                return True
        return False
    
    def _apply_medical_terminology_mapping(self, entities):
        """Apply medical terminology mappings to standardize concepts"""
        standardized_entities = {}
        
        for entity_type, entity_list in entities.items():
            standardized_entities[entity_type] = []
            
            for entity in entity_list:
                # Try to standardize based on synonyms
                standardized = False
                
                if entity_type in self.medical_terminologies:
                    for standard_term, synonyms in self.medical_terminologies[entity_type].items():
                        for synonym in synonyms:
                            if synonym.lower() in entity.value.lower():
                                # Create standardized entity
                                standardized_entity = MedicalEntity(
                                    type=entity_type,
                                    value=entity.value,
                                    normalized_value=standard_term,
                                    confidence=entity.confidence,
                                    temporal_context=entity.temporal_context,
                                    source=entity.source,
                                    position=entity.position,
                                    metadata=entity.metadata.copy()
                                )
                                standardized_entities[entity_type].append(standardized_entity)
                                standardized = True
                                break
                        if standardized:
                            break
                
                # If not standardized, keep original
                if not standardized:
                    standardized_entities[entity_type].append(entity)
        
        return standardized_entities
    
    def _merge_similar_entities(self, entities):
        """Merge similar entities to remove duplicates"""
        merged_entities = {}
        
        for entity_type, entity_list in entities.items():
            if not entity_list:
                merged_entities[entity_type] = []
                continue
            
            # Group similar entities
            groups = []
            remaining = entity_list.copy()
            
            while remaining:
                current = remaining.pop(0)
                current_group = [current]
                
                i = 0
                while i < len(remaining):
                    other = remaining[i]
                    
                    # If entities are similar, add to current group and remove from remaining
                    if self._are_entities_similar(current, other):
                        current_group.append(other)
                        remaining.pop(i)
                    else:
                        i += 1
                
                groups.append(current_group)
            
            # For each group, create a merged entity
            merged_entities[entity_type] = []
            
            for group in groups:
                if not group:
                    continue
                
                # Take the entity with highest confidence as base
                base_entity = max(group, key=lambda e: e.confidence)
                
                # Calculate new confidence based on multiple detections
                confidence_boost = min(0.2, 0.05 * len(group))  # Diminishing returns
                new_confidence = min(1.0, base_entity.confidence + confidence_boost)
                
                # Merge metadata
                merged_metadata = base_entity.metadata.copy()
                merged_metadata["merged_from"] = len(group)
                merged_metadata["original_values"] = [e.value for e in group]
                
                # Create merged entity
                merged_entity = MedicalEntity(
                    type=entity_type,
                    value=base_entity.value,
                    normalized_value=base_entity.normalized_value,
                    confidence=new_confidence,
                    temporal_context=base_entity.temporal_context,
                    source=base_entity.source,
                    position=base_entity.position,
                    metadata=merged_metadata,
                    is_negated=base_entity.is_negated,
                    is_uncertain=base_entity.is_uncertain,
                    is_hypothetical=base_entity.is_hypothetical,
                    is_about_family=base_entity.is_about_family,
                    measurement_unit=base_entity.measurement_unit
                )
                
                merged_entities[entity_type].append(merged_entity)
        
        return merged_entities
    
    def _are_entities_similar(self, entity1, entity2):
        """Determine if two entities refer to the same information"""
        # Check if positions are close (likely the same mention)
        if entity1.position and entity2.position:
            pos1_start, pos1_end = entity1.position
            pos2_start, pos2_end = entity2.position
            
            # If positions overlap or are very close, consider them the same mention
            if (pos1_start <= pos2_end and pos2_start <= pos1_end) or abs(pos1_end - pos2_start) < 5 or abs(pos2_end - pos1_start) < 5:
                return True
        
        # Check if values are identical or nearly identical
        if entity1.value.lower() == entity2.value.lower():
            return True
            
        # Check if normalized values are identical
        if entity1.normalized_value and entity2.normalized_value and entity1.normalized_value == entity2.normalized_value:
            return True
        
        # For numeric values, check if they're close
        try:
            # Extract numbers from both values
            nums1 = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', entity1.value)]
            nums2 = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', entity2.value)]
            
            # If both have numbers and they're close
            if nums1 and nums2:
                for n1 in nums1:
                    for n2 in nums2:
                        # Allow different thresholds based on magnitude
                        if n1 > 100:  # For larger values like blood glucose
                            threshold = 10
                        elif n1 > 10:  # For medium values
                            threshold = 5
                        else:  # For small values like A1C
                            threshold = 0.5
                            
                        if abs(n1 - n2) <= threshold:
                            return True
        except:
            pass
        
        # For medications and conditions, check substring match
        if entity1.type in ["medications", "chronic_conditions", "symptoms"]:
            if len(entity1.value) > 3 and len(entity2.value) > 3:
                if entity1.value.lower() in entity2.value.lower() or entity2.value.lower() in entity1.value.lower():
                    return True
        
        return False
    
    def _construct_timeline(self, entities, text):
        """Construct a timeline from medical entities (legacy method)"""
        # Use the enhanced temporal resolver if enabled
        if self.use_enhanced_temporal and hasattr(self, 'temporal_model'):
            # Process entities with the temporal model
            entities = self.temporal_model.process_timeline(entities, text)
            
            # Group entities by temporal context
            timeline = {
                "past_history": [],
                "recent_past": [],
                "current": [],
                "immediate_future": [],
                "distant_future": [],
                "unknown": []
            }
            
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    temporal_context = entity.temporal_context or "unknown"
                    if temporal_context not in timeline:
                        timeline[temporal_context] = []
                    
                    timeline[temporal_context].append({
                        "type": entity_type,
                        "value": entity.value,
                        "normalized_value": entity.normalized_value,
                        "confidence": entity.confidence
                    })
            
            return timeline
        
        # Fall back to traditional method
        return self.temporal_resolver._construct_timeline(entities, text)
    
    def _extract_key_sections(self, entities, timeline, contradictions, full_text, care_manager_text, patient_text):
        """Extract key sections for structured summary"""
        sections = {}
        
        # Health status
        sections["health_status"] = self._extract_health_status(entities, timeline)
        
        # Medications section
        sections["medications"] = self._extract_medications_section(entities, timeline, contradictions)
        
        # Vital signs section
        sections["vital_signs"] = self._extract_vital_signs_section(entities, timeline, contradictions)
        
        # Symptoms section
        sections["symptoms"] = self._extract_symptoms_section(entities, timeline)
        
        # Activity and lifestyle section
        sections["lifestyle"] = self._extract_lifestyle_section(entities, timeline)
        
        # Plan and follow-up section
        sections["plan"] = self._extract_plan_section(care_manager_text)
        
        # Key concerns and questions
        sections["concerns"] = self._extract_concerns_section(patient_text)
        
        return sections
    
    def _extract_health_status(self, entities, timeline):
        """Extract health status information"""
        status = {
            "chronic_conditions": [],
            "current_state": [],
            "changes": []
        }
        
        # Add chronic conditions
        if 'chronic_conditions' in entities:
            for entity in entities['chronic_conditions']:
                if not entity.is_negated and not entity.is_about_family and entity.confidence >= self.confidence_threshold:
                    condition = {
                        "condition": entity.normalized_value or entity.value,
                        "confidence": entity.confidence,
                        "temporal_context": entity.temporal_context
                    }
                    status["chronic_conditions"].append(condition)
        
        # Add current health state
        current_entities = timeline.get("current", [])
        for entity in current_entities:
            if entity.get("type") in ["symptoms", "vital_signs", "blood_pressure", "blood_glucose"]:
                status["current_state"].append({
                    "finding": entity.get("normalized_value") or entity.get("value"),
                    "type": entity.get("type"),
                    "confidence": entity.get("confidence", 0.7)
                })
        
        # Add changes in health
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                if "change" in entity.value.lower() or "improved" in entity.value.lower() or "worse" in entity.value.lower() or "better" in entity.value.lower():
                    if not entity.is_negated and entity.confidence >= self.confidence_threshold:
                        status["changes"].append({
                            "change": entity.normalized_value or entity.value,
                            "type": entity.type,
                            "confidence": entity.confidence
                        })
        
        return status
    
    def _extract_medications_section(self, entities, timeline, contradictions):
        """Extract medications information"""
        medications = {
            "current": [],
            "discontinued": [],
            "changed": [],
            "adherence": []
        }
        
        # Add medications
        if 'medications' in entities:
            for entity in entities['medications']:
                if not entity.is_hypothetical and entity.confidence >= self.confidence_threshold:
                    med_info = {
                        "medication": entity.normalized_value or entity.value,
                        "confidence": entity.confidence,
                        "temporal_context": entity.temporal_context
                    }
                    
                    # Check for discontinuation language
                    if "stop" in entity.value.lower() or "discontinue" in entity.value.lower():
                        medications["discontinued"].append(med_info)
                    # Check for change language
                    elif "change" in entity.value.lower() or "adjust" in entity.value.lower() or "increase" in entity.value.lower() or "decrease" in entity.value.lower():
                        medications["changed"].append(med_info)
                    # Check current medications
                    elif not entity.is_negated:
                        medications["current"].append(med_info)
        
        # Check for adherence information
        adherence_patterns = [
            r'(?:taking|using|following|adhering\s+to|compliant\s+with)\s+(?:all|the|medication|medications|meds|treatment|plan|regimen)',
            r'(?:missed|skipped|forgot|not\s+taking|not\s+using|stopped|discontinue)\s+(?:medication|medications|meds|doses|pills|treatment)',
            r'(?:every\s+day|daily|regularly|as\s+prescribed|as\s+directed)'
        ]
        
        for pattern in adherence_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                medications["adherence"].append({
                    "statement": match.group(0),
                    "position": match.span()
                })
        
        return medications
    
    def _extract_vital_signs_section(self, entities, timeline, contradictions):
        """Extract vital signs information"""
        vitals = {
            "blood_pressure": [],
            "blood_glucose": [],
            "weight": [],
            "other_vitals": []
        }
        
        # Process blood pressure
        if 'blood_pressure' in entities:
            for entity in entities['blood_pressure']:
                if not entity.is_hypothetical and not entity.is_negated and entity.confidence >= self.confidence_threshold:
                    vital_info = {
                        "value": entity.normalized_value or entity.value,
                        "confidence": entity.confidence,
                        "temporal_context": entity.temporal_context,
                        "is_uncertain": entity.is_uncertain
                    }
                    vitals["blood_pressure"].append(vital_info)
        
        # Process blood glucose
        if 'blood_glucose' in entities:
            for entity in entities['blood_glucose']:
                if not entity.is_hypothetical and not entity.is_negated and entity.confidence >= self.confidence_threshold:
                    vital_info = {
                        "value": entity.normalized_value or entity.value,
                        "confidence": entity.confidence,
                        "temporal_context": entity.temporal_context,
                        "is_uncertain": entity.is_uncertain
                    }
                    vitals["blood_glucose"].append(vital_info)
        
        # Process weight changes
        if 'weight_change' in entities:
            for entity in entities['weight_change']:
                if not entity.is_hypothetical and not entity.is_negated and entity.confidence >= self.confidence_threshold:
                    vital_info = {
                        "value": entity.normalized_value or entity.value,
                        "confidence": entity.confidence,
                        "temporal_context": entity.temporal_context,
                        "is_uncertain": entity.is_uncertain
                    }
                    vitals["weight"].append(vital_info)
        
        # Process other vital signs
        if 'vital_signs' in entities:
            for entity in entities['vital_signs']:
                if not entity.is_hypothetical and not entity.is_negated and entity.confidence >= self.confidence_threshold:
                    vital_info = {
                        "value": entity.normalized_value or entity.value,
                        "confidence": entity.confidence,
                        "temporal_context": entity.temporal_context,
                        "is_uncertain": entity.is_uncertain
                    }
                    vitals["other_vitals"].append(vital_info)
        
        # Note contradictions in vitals
        vital_contradictions = []
        for contradiction in contradictions:
            if contradiction["entity_type"] in ["blood_pressure", "blood_glucose", "weight_change", "vital_signs"]:
                vital_contradictions.append({
                    "type": contradiction["type"],
                    "description": contradiction["description"],
                    "entity_type": contradiction["entity_type"]
                })
        
        vitals["contradictions"] = vital_contradictions
        
        return vitals
    
    def _extract_symptoms_section(self, entities, timeline):
        """Extract symptoms information"""
        symptoms = {
            "current": [],
            "resolved": [],
            "changed": []
        }
        
        # Add symptoms
        if 'symptoms' in entities:
            for entity in entities['symptoms']:
                if entity.confidence >= self.confidence_threshold:
                    symptom_info = {
                        "symptom": entity.normalized_value or entity.value,
                        "confidence": entity.confidence,
                        "temporal_context": entity.temporal_context,
                        "is_negated": entity.is_negated,
                        "is_uncertain": entity.is_uncertain
                    }
                    
                    if entity.is_negated or entity.temporal_context == "past_history":
                        symptoms["resolved"].append(symptom_info)
                    elif "better" in entity.value.lower() or "improved" in entity.value.lower() or "worse" in entity.value.lower() or "change" in entity.value.lower():
                        symptoms["changed"].append(symptom_info)
                    else:
                        symptoms["current"].append(symptom_info)
        
        return symptoms
    
    def _extract_lifestyle_section(self, entities, timeline):
        """Extract lifestyle and activity information"""
        lifestyle = {
            "exercise": [],
            "diet": [],
            "habits": []
        }
        
        # Add exercise information
        if 'exercise' in entities:
            for entity in entities['exercise']:
                if not entity.is_hypothetical and entity.confidence >= self.confidence_threshold:
                    activity_info = {
                        "activity": entity.normalized_value or entity.value,
                        "confidence": entity.confidence,
                        "temporal_context": entity.temporal_context,
                        "is_negated": entity.is_negated
                    }
                    lifestyle["exercise"].append(activity_info)
        
        # Look for diet information
        diet_patterns = [
            r'(?:diet|eating|food|nutrition|meals?|consume|consuming|intake)\s+(?:healthy|balanced|good|poor|bad|improved|better|worse|changed)',
            r'(?:carbs?|carbohydrates?|sugar|sugars|sweets?|processed\s+foods?|junk\s+foods?)',
            r'(?:vegetables?|fruits?|protein|meat|fish|dairy|whole\s+grains?)',
            r'(?:cut(?:ting)?\s+(?:down|back)|reduce|reducing|lower|lowering|increase|increasing|watching)\s+(?:salt|sodium|sugar|carbs?|fat|calories)'
        ]
        
        for pattern in diet_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                context_start = max(0, match.start() - 50)
                context_end = min(len(full_text), match.end() + 50)
                context = full_text[context_start:context_end]
                
                # Check if negated
                is_negated = any(re.search(neg_pattern, context, re.IGNORECASE) for neg_pattern in self.negation_patterns)
                
                lifestyle["diet"].append({
                    "description": match.group(0),
                    "confidence": 0.7,
                    "is_negated": is_negated,
                    "context": context
                })
        
        # Look for habits information
        habit_patterns = [
            r'(?:smoking|smoke|cigarettes?|tobacco)',
            r'(?:drinking|alcohol|beers?|wines?|spirits?|liquor)',
            r'(?:drugs?|substance|recreational)',
            r'(?:sleep|sleeping|insomnia|rest|resting)',
            r'(?:stress|anxiety|relaxation|meditation|mindfulness)'
        ]
        
        for pattern in habit_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                context_start = max(0, match.start() - 50)
                context_end = min(len(full_text), match.end() + 50)
                context = full_text[context_start:context_end]
                
                # Check if negated
                is_negated = any(re.search(neg_pattern, context, re.IGNORECASE) for neg_pattern in self.negation_patterns)
                
                lifestyle["habits"].append({
                    "description": match.group(0),
                    "confidence": 0.7,
                    "is_negated": is_negated,
                    "context": context
                })
        
        return lifestyle
    
    def _extract_plan_section(self, care_manager_text):
        """Extract plan and follow-up information"""
        plan = {
            "follow_up": [],
            "medication_changes": [],
            "recommendations": [],
            "monitoring": []
        }
        
        # Follow-up patterns
        follow_up_patterns = [
            r'(?:follow[\s-]up|come\s+back|return|visit|call|appointment|check[\s-]in)\s+(?:in|after|within|during)\s+(\d+)[\s-]*(?:days?|weeks?|months?)',
            r'(?:schedule|book|make)\s+(?:an|a|the)?\s+(?:appointment|visit|consult|checkup)',
            r'(?:see|contact|call)\s+(?:me|us|doctor|dr\.|physician|specialist|provider)\s+(?:in|after|within|during|if|when)'
        ]
        
        for pattern in follow_up_patterns:
            matches = re.finditer(pattern, care_manager_text, re.IGNORECASE)
            for match in matches:
                plan["follow_up"].append({
                    "instruction": match.group(0),
                    "confidence": 0.8
                })
        
        # Medication change patterns
        med_change_patterns = [
            r'(?:continue|keep|maintain|increase|decrease|adjust|change|start|begin|stop|discontinue)\s+(?:taking|using|with)\s+([a-z\s]+)',
            r'(?:prescription|prescribe|refill)\s+(?:for|of)?\s+([a-z\s]+)',
            r'new\s+(?:medication|prescription|drug|dosage)'
        ]
        
        for pattern in med_change_patterns:
            matches = re.finditer(pattern, care_manager_text, re.IGNORECASE)
            for match in matches:
                plan["medication_changes"].append({
                    "instruction": match.group(0),
                    "confidence": 0.8
                })
        
        # Recommendation patterns
        recommendation_patterns = [
            r'(?:recommend|advise|suggest|should|need\s+to|have\s+to|must|important\s+to|try\s+to)\s+([a-z\s]+)',
            r'(?:diet|exercise|activity|work|rest|modification|change|adjustment|improvement)',
            r'(?:increase|decrease|reduce|limit|avoid|cut\s+(?:down|back))\s+([a-z\s]+)'
        ]
        
        for pattern in recommendation_patterns:
            matches = re.finditer(pattern, care_manager_text, re.IGNORECASE)
            for match in matches:
                plan["recommendations"].append({
                    "instruction": match.group(0),
                    "confidence": 0.8
                })
        
        # Monitoring patterns
        monitoring_patterns = [
            r'(?:monitor|check|measure|test|track|record|log|journal|diary)\s+(?:your|the)?\s+([a-z\s]+)',
            r'(?:blood\s+(?:pressure|sugar|glucose)|weight|temperature|symptoms)',
            r'(?:daily|weekly|monthly|regularly|periodically|consistently|twice\s+daily|before\s+meals|after\s+meals)'
        ]
        
        for pattern in monitoring_patterns:
            matches = re.finditer(pattern, care_manager_text, re.IGNORECASE)
            for match in matches:
                plan["monitoring"].append({
                    "instruction": match.group(0),
                    "confidence": 0.8
                })
        
        return plan
    
    def _extract_concerns_section(self, patient_text):
        """Extract patient concerns and questions"""
        concerns = {
            "questions": [],
            "worries": [],
            "needs": []
        }
        
        # Question patterns
        question_patterns = [
            r'(?:what|when|where|who|why|how|is|are|can|should|could|would|will|do|does|did|am|if)(?:\s+[^.?!]+)+\?',
            r'(?:wonder|wondering|curious|want\s+to\s+know|question|asking)\s+(?:about|if|whether|why|how|what)'
        ]
        
        for pattern in question_patterns:
            matches = re.finditer(pattern, patient_text, re.IGNORECASE)
            for match in matches:
                concerns["questions"].append({
                    "question": match.group(0),
                    "confidence": 0.8
                })
        
        # Worry patterns
        worry_patterns = [
            r'(?:worried|concerned|anxious|nervous|scared|afraid|fear|stress|stressful)\s+(?:about|that|because|of|by)',
            r'(?:not\s+(?:sure|certain|confident)|unsure|uncertain|doubt|doubtful)',
            r'(?:trouble|problem|difficulty|hard|challenging|struggle|struggling)\s+(?:with|to)'
        ]
        
        for pattern in worry_patterns:
            matches = re.finditer(pattern, patient_text, re.IGNORECASE)
            for match in matches:
                concerns["worries"].append({
                    "worry": match.group(0),
                    "confidence": 0.8
                })
        
        # Need patterns
        need_patterns = [
            r'(?:need|want|would\s+like|wish|hope|trying|looking\s+for|searching\s+for)\s+(?:to|for|a|an|some|more)',
            r'(?:help|support|assistance|guidance|advice|information|resources)',
            r'(?:can\'t|cannot|having\s+trouble|difficulty|problem|issue)\s+(?:with|getting|finding|obtaining)'
        ]
        
        for pattern in need_patterns:
            matches = re.finditer(pattern, patient_text, re.IGNORECASE)
            for match in matches:
                concerns["needs"].append({
                    "need": match.group(0),
                    "confidence": 0.8
                })
        
        return concerns
    
    def _generate_narrative_summary(self, sections, entities, contradictions):
        """Generate a narrative summary from structured sections"""
        # Start with key health status information
        narrative = []
        
        # Patient identifier and visit type (generic)
        narrative.append("MEDICAL CONVERSATION SUMMARY")
        narrative.append("")
        
        # Health status summary
        health_status = sections["health_status"]
        chronic_conditions = health_status["chronic_conditions"]
        if chronic_conditions:
            condition_text = ", ".join([c["condition"] for c in chronic_conditions])
            narrative.append(f"Patient with {condition_text}.")
        
        # Current vital signs
        vital_signs = sections["vital_signs"]
        bp_entries = vital_signs["blood_pressure"]
        glucose_entries = vital_signs["blood_glucose"]
        weight_entries = vital_signs["weight"]
        
        vital_texts = []
        if bp_entries:
            bp = max(bp_entries, key=lambda x: x["confidence"])
            vital_texts.append(f"BP {bp['value']}")
        
        if glucose_entries:
            glucose = max(glucose_entries, key=lambda x: x["confidence"])
            vital_texts.append(f"glucose {glucose['value']}")
        
        if weight_entries:
            weight = max(weight_entries, key=lambda x: x["confidence"])
            vital_texts.append(f"{weight['value']}")
        
        if vital_texts:
            narrative.append(f"Current vitals: {'; '.join(vital_texts)}.")
        
        # Medications
        medications = sections["medications"]
        current_meds = medications["current"]
        changed_meds = medications["changed"]
        discontinued_meds = medications["discontinued"]
        
        if current_meds:
            med_text = ", ".join([m["medication"] for m in current_meds])
            narrative.append(f"Current medications: {med_text}.")
        
        if changed_meds or discontinued_meds:
            med_changes = []
            if changed_meds:
                change_text = ", ".join([m["medication"] for m in changed_meds])
                med_changes.append(f"changed/adjusted {change_text}")
            if discontinued_meds:
                discontinued_text = ", ".join([m["medication"] for m in discontinued_meds])
                med_changes.append(f"discontinued {discontinued_text}")
            
            narrative.append(f"Medication changes: {'; '.join(med_changes)}.")
        
        # Lifestyle
        lifestyle = sections["lifestyle"]
        exercise = lifestyle["exercise"]
        diet = lifestyle["diet"]
        
        lifestyle_texts = []
        if exercise:
            exercise_text = "; ".join([e["activity"] for e in exercise if not e["is_negated"]])
            if exercise_text:
                lifestyle_texts.append(f"Exercise: {exercise_text}")
        
        if diet:
            diet_text = "; ".join([d["description"] for d in diet if not d["is_negated"]])
            if diet_text:
                lifestyle_texts.append(f"Diet: {diet_text}")
        
        if lifestyle_texts:
            narrative.append(f"Lifestyle: {'. '.join(lifestyle_texts)}.")
        
        # Plan
        plan = sections["plan"]
        follow_up = plan["follow_up"]
        recommendations = plan["recommendations"]
        monitoring = plan["monitoring"]
        
        plan_texts = []
        if follow_up:
            follow_up_text = follow_up[0]["instruction"]
            plan_texts.append(f"Follow-up: {follow_up_text}")
        
        if recommendations:
            rec_text = "; ".join([r["instruction"] for r in recommendations[:2]])
            plan_texts.append(f"Recommendations: {rec_text}")
        
        if monitoring:
            monitoring_text = "; ".join([m["instruction"] for m in monitoring[:2]])
            plan_texts.append(f"Monitoring: {monitoring_text}")
        
        if plan_texts:
            narrative.append(f"Plan: {'. '.join(plan_texts)}.")
        
        # Highlight any contradictions or uncertainties
        if contradictions:
            contradiction_text = "; ".join([c["description"] for c in contradictions[:3]])
            narrative.append(f"Note: Possible contradictions in information: {contradiction_text}.")
        
        return "\n".join(narrative)
    
    def _generate_soap_note(self, sections, entities, timeline):
        """Generate a SOAP note from the structured sections"""
        soap_note = {
            "Subjective": self._generate_subjective_section(sections, entities),
            "Objective": self._generate_objective_section(sections, entities),
            "Assessment": self._generate_assessment_section(sections, entities, timeline),
            "Plan": self._generate_plan_section_soap(sections, entities)
        }
        
        return soap_note
    
    def _generate_subjective_section(self, sections, entities):
        """Generate the Subjective section of the SOAP note"""
        subjective = []
        
        # Patient's reported symptoms
        symptoms = sections["symptoms"]
        current_symptoms = symptoms["current"]
        if current_symptoms:
            symptom_text = ", ".join([s["symptom"] for s in current_symptoms if not s["is_negated"]])
            if symptom_text:
                subjective.append(f"Patient reports: {symptom_text}")
        
        # Changed symptoms
        changed_symptoms = symptoms["changed"]
        if changed_symptoms:
            changed_text = ", ".join([s["symptom"] for s in changed_symptoms if not s["is_negated"]])
            if changed_text:
                subjective.append(f"Changes in symptoms: {changed_text}")
        
        # Patient concerns
        concerns = sections["concerns"]
        worries = concerns["worries"]
        needs = concerns["needs"]
        
        if worries:
            worry_text = "; ".join([w["worry"] for w in worries[:2]])
            subjective.append(f"Concerns: {worry_text}")
        
        if needs:
            need_text = "; ".join([n["need"] for n in needs[:2]])
            subjective.append(f"Needs: {need_text}")
        
        # Lifestyle factors
        lifestyle = sections["lifestyle"]
        exercise = lifestyle["exercise"]
        diet = lifestyle["diet"]
        habits = lifestyle["habits"]
        
        if exercise:
            exercise_text = "; ".join([e["activity"] for e in exercise if not e["is_negated"]])
            if exercise_text:
                subjective.append(f"Exercise: {exercise_text}")
        
        if diet:
            diet_text = "; ".join([d["description"] for d in diet if not d["is_negated"]])
            if diet_text:
                subjective.append(f"Diet: {diet_text}")
        
        if habits:
            habit_text = "; ".join([h["description"] for h in habits if not h["is_negated"]])
            if habit_text:
                subjective.append(f"Habits: {habit_text}")
        
        # Medication adherence
        medications = sections["medications"]
        adherence = medications.get("adherence", [])
        if adherence:
            adherence_text = "; ".join([a["statement"] for a in adherence[:2]])
            subjective.append(f"Medication adherence: {adherence_text}")
        
        if not subjective:
            return "No subjective information reported."
        
        return "\n".join(subjective)
    
    def _generate_objective_section(self, sections, entities):
        """Generate the Objective section of the SOAP note"""
        objective = []
        
        # Vital signs
        vitals = sections["vital_signs"]
        bp_entries = vitals["blood_pressure"]
        glucose_entries = vitals["blood_glucose"]
        weight_entries = vitals["weight"]
        other_vitals = vitals["other_vitals"]
        
        if bp_entries:
            bp = max(bp_entries, key=lambda x: x["confidence"])
            objective.append(f"Blood Pressure: {bp['value']}")
        
        if glucose_entries:
            glucose = max(glucose_entries, key=lambda x: x["confidence"])
            objective.append(f"Blood Glucose: {glucose['value']}")
        
        if weight_entries:
            weight = max(weight_entries, key=lambda x: x["confidence"])
            objective.append(f"Weight Change: {weight['value']}")
        
        if other_vitals:
            vital_text = "; ".join([v["value"] for v in other_vitals])
            objective.append(f"Other Vitals: {vital_text}")
        
        # Current medications
        medications = sections["medications"]
        current_meds = medications["current"]
        
        if current_meds:
            med_text = ", ".join([m["medication"] for m in current_meds])
            objective.append(f"Current Medications: {med_text}")
        
        # Chronic conditions
        health_status = sections["health_status"]
        chronic_conditions = health_status["chronic_conditions"]
        if chronic_conditions:
            condition_text = ", ".join([c["condition"] for c in chronic_conditions])
            objective.append(f"Chronic Conditions: {condition_text}")
        
        # Vitals contradictions
        contradictions = vitals.get("contradictions", [])
        if contradictions:
            contra_text = "; ".join([c["description"] for c in contradictions])
            objective.append(f"Note: {contra_text}")
        
        if not objective:
            return "No objective findings recorded."
        
        return "\n".join(objective)
    
    def _generate_assessment_section(self, sections, entities, timeline):
        """Generate the Assessment section of the SOAP note"""
        assessment = []
        
        # Health status
        health_status = sections["health_status"]
        chronic_conditions = health_status["chronic_conditions"]
        current_state = health_status["current_state"]
        changes = health_status["changes"]
        
        if chronic_conditions:
            condition_text = ", ".join([c["condition"] for c in chronic_conditions])
            assessment.append(f"Patient with {condition_text}.")
        
        if current_state:
            state_text = "; ".join([s["finding"] for s in current_state])
            assessment.append(f"Current status: {state_text}.")
        
        if changes:
            change_text = "; ".join([c["change"] for c in changes])
            assessment.append(f"Notable changes: {change_text}.")
        
        # Diabetes management assessment
        if any("diabetes" in c["condition"].lower() for c in chronic_conditions):
            glucose_entries = sections["vital_signs"]["blood_glucose"]
            if glucose_entries:
                glucose_values = [int(re.search(r'(\d+)', g["value"]).group(1)) for g in glucose_entries if re.search(r'(\d+)', g["value"])]
                if glucose_values:
                    avg_glucose = sum(glucose_values) / len(glucose_values)
                    if avg_glucose < 70:
                        assessment.append("Diabetes: Hypoglycemic - blood sugars below target range.")
                    elif avg_glucose < 140:
                        assessment.append("Diabetes: Well-controlled - blood sugars within target range.")
                    elif avg_glucose < 180:
                        assessment.append("Diabetes: Moderately controlled - blood sugars slightly elevated.")
                    else:
                        assessment.append("Diabetes: Poorly controlled - blood sugars significantly elevated.")
        
        # Hypertension assessment
        if any("hypertension" in c["condition"].lower() or "high blood pressure" in c["condition"].lower() for c in chronic_conditions):
            bp_entries = sections["vital_signs"]["blood_pressure"]
            if bp_entries:
                systolic_values = []
                diastolic_values = []
                for bp in bp_entries:
                    bp_match = re.search(r'(\d+)/(\d+)', bp["value"])
                    if bp_match:
                        systolic_values.append(int(bp_match.group(1)))
                        diastolic_values.append(int(bp_match.group(2)))
                
                if systolic_values and diastolic_values:
                    avg_systolic = sum(systolic_values) / len(systolic_values)
                    avg_diastolic = sum(diastolic_values) / len(diastolic_values)
                    
                    if avg_systolic < 120 and avg_diastolic < 80:
                        assessment.append("Hypertension: Well-controlled - blood pressure in normal range.")
                    elif avg_systolic < 130 and avg_diastolic < 85:
                        assessment.append("Hypertension: Well-controlled - blood pressure in target range.")
                    elif avg_systolic < 140 and avg_diastolic < 90:
                        assessment.append("Hypertension: Borderline control - blood pressure slightly elevated.")
                    else:
                        assessment.append("Hypertension: Inadequate control - blood pressure remains elevated.")
        
        # Weight assessment
        weight_entries = sections["vital_signs"]["weight"]
        if weight_entries:
            for entry in weight_entries:
                if "lost" in entry["value"].lower():
                    assessment.append("Weight management: Patient reports weight loss.")
                elif "gained" in entry["value"].lower():
                    assessment.append("Weight management: Patient reports weight gain.")
                break
        
        # Medication assessment
        medications = sections["medications"]
        if medications["adherence"]:
            adherence_issues = any("missed" in a["statement"].lower() or 
                                "not taking" in a["statement"].lower() or 
                                "forgot" in a["statement"].lower() 
                                for a in medications["adherence"])
            if adherence_issues:
                assessment.append("Medication: Issues with medication adherence identified.")
            else:
                assessment.append("Medication: Patient reports good medication adherence.")
        
        # Exercise assessment
        lifestyle = sections["lifestyle"]
        if lifestyle["exercise"]:
            exercise_reported = any(not e["is_negated"] for e in lifestyle["exercise"])
            if exercise_reported:
                assessment.append("Lifestyle: Patient engaging in physical activity.")
            else:
                assessment.append("Lifestyle: Limited physical activity reported.")
        
        if not assessment:
            return "No significant assessment findings."
        
        return "\n".join(assessment)
    
    def _generate_plan_section_soap(self, sections, entities):
        """Generate the Plan section of the SOAP note"""
        plan = []
        
        # Follow-up
        follow_up = sections["plan"]["follow_up"]
        if follow_up:
            follow_up_text = "; ".join([f["instruction"] for f in follow_up])
            plan.append(f"Follow-up: {follow_up_text}")
        
        # Medication changes
        med_changes = sections["plan"]["medication_changes"]
        if med_changes:
            changes_text = "; ".join([m["instruction"] for m in med_changes])
            plan.append(f"Medication changes: {changes_text}")
        
        # Recommendations
        recommendations = sections["plan"]["recommendations"]
        if recommendations:
            rec_text = "; ".join([r["instruction"] for r in recommendations])
            plan.append(f"Recommendations: {rec_text}")
        
        # Monitoring
        monitoring = sections["plan"]["monitoring"]
        if monitoring:
            monitoring_text = "; ".join([m["instruction"] for m in monitoring])
            plan.append(f"Monitoring: {monitoring_text}")
        
        # Add specific recommendations based on assessment
        health_status = sections["health_status"]
        chronic_conditions = health_status["chronic_conditions"]
        
        # Diabetes-specific plans
        if any("diabetes" in c["condition"].lower() for c in chronic_conditions):
            glucose_entries = sections["vital_signs"]["blood_glucose"]
            if glucose_entries:
                glucose_values = [int(re.search(r'(\d+)', g["value"]).group(1)) for g in glucose_entries if re.search(r'(\d+)', g["value"])]
                if glucose_values:
                    avg_glucose = sum(glucose_values) / len(glucose_values)
                    if avg_glucose > 180:
                        plan.append("Diabetes management: Consider adjusting medication regimen for better glycemic control. Continue monitoring blood glucose levels regularly.")
        
        # Hypertension-specific plans
        if any("hypertension" in c["condition"].lower() or "high blood pressure" in c["condition"].lower() for c in chronic_conditions):
            bp_entries = sections["vital_signs"]["blood_pressure"]
            if bp_entries:
                systolic_values = []
                diastolic_values = []
                for bp in bp_entries:
                    bp_match = re.search(r'(\d+)/(\d+)', bp["value"])
                    if bp_match:
                        systolic_values.append(int(bp_match.group(1)))
                        diastolic_values.append(int(bp_match.group(2)))
                
                if systolic_values and diastolic_values:
                    avg_systolic = sum(systolic_values) / len(systolic_values)
                    avg_diastolic = sum(diastolic_values) / len(diastolic_values)
                    
                    if avg_systolic >= 140 or avg_diastolic >= 90:
                        plan.append("Hypertension management: Consider medication adjustment for blood pressure control. Continue regular blood pressure monitoring.")
        
        if not plan:
            return "Continue current management and follow up as needed."
        
        return "\n".join(plan)
    
    def _entity_to_dict(self, entity):
        """Convert MedicalEntity to dictionary for serialization"""
        return {
            "type": entity.type,
            "value": entity.value,
            "normalized_value": entity.normalized_value,
            "confidence": entity.confidence,
            "temporal_context": entity.temporal_context,
            "source": entity.source,
            "position": entity.position,
            "is_negated": entity.is_negated,
            "is_uncertain": entity.is_uncertain,
            "is_hypothetical": entity.is_hypothetical,
            "is_about_family": entity.is_about_family,
            "measurement_unit": entity.measurement_unit,
            "metadata": entity.metadata
        }
    
    def process_segment(self, segment, conversation_context=None):
        """Process a single conversation segment for real-time analysis"""
        text = segment.get("text", "")
        speaker = segment.get("speaker", "unknown")
        
        # Extract entities from segment
        entities = self._extract_medical_entities(text)
        
        # Process entities for context
        entities = self._process_entity_modifiers(entities, text)
        
        # Detect contradictions within this segment
        contradictions = self.contradiction_detector.detect_contradictions(entities, text)
        
        # Construct timeline for the segment
        timeline = self._construct_timeline(entities, text)
        
        return {
            "entities": entities,
            "contradictions": contradictions,
            "timeline": timeline,
            "segment_id": segment.get("id"),
            "speaker": speaker,
            "text": text
        }
    
    def update_analysis(self, new_segment_result, conversation_context):
        """Update ongoing analysis with new segment results"""
        # Combine entities from all segments
        all_entities = {}
        
        # Extract entities from all context segments
        for segment in conversation_context:
            if "entities" in segment:
                for entity_type, entity_list in segment["entities"].items():
                    if entity_type not in all_entities:
                        all_entities[entity_type] = []
                    all_entities[entity_type].extend(entity_list)
        
        # Merge similar entities
        merged_entities = self._merge_similar_entities(all_entities)
        
        # Construct comprehensive timeline
        full_text = " ".join([segment.get("text", "") for segment in conversation_context])
        timeline = self._construct_timeline(merged_entities, full_text)
        
        # Check for contradictions across all segments
        contradictions = self.contradiction_detector.detect_contradictions(merged_entities, full_text)
        
        # Extract key sections
        care_manager_text = " ".join([segment.get("text", "") for segment in conversation_context 
                                    if segment.get("speaker") == "SPEAKER_00"])
        patient_text = " ".join([segment.get("text", "") for segment in conversation_context 
                                if segment.get("speaker") == "SPEAKER_01"])
        
        sections = self._extract_key_sections(merged_entities, timeline, contradictions, 
                                            full_text, care_manager_text, patient_text)
        
        # Generate updated summary
        narrative_summary = self._generate_narrative_summary(sections, merged_entities, contradictions)
        
        # Generate updated SOAP note
        soap_note = self._generate_soap_note(sections, merged_entities, timeline)
        
        return {
            "narrative_summary": narrative_summary,
            "structured_summary": sections,
            "soap_note": soap_note,
            "timeline": timeline,
            "entities": {
                entity_type: [self._entity_to_dict(e) for e in entity_list]
                for entity_type, entity_list in merged_entities.items()
            },
            "contradictions": contradictions
        }