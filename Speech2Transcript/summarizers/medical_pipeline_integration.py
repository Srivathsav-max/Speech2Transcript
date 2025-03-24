# medical_pipeline_integration.py
import os
import json
import torch
import logging
import numpy as np
import concurrent.futures
import re
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, field

# Import the enhanced medical summarizer
from Speech2Transcript.summarizers.medical_summarizer_advanced import AdvancedMedicalSummarizer, MedicalEntity

class MedicalTranscriptSummarizer:
    """Integration layer for medical transcript summarization with enhanced entity recognition,
    temporal processing, and scalability improvements"""
    
    def __init__(
            self,
            base_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            ner_model: str = "emilyalsentzer/Bio_ClinicalBERT",
            qa_model: str = "dmis-lab/biobert-base-cased-v1.1-squad",
            sentence_model: str = "sentence-transformers/paraphrase-mpnet-base-v2",
            device: Optional[str] = None,
            compute_type: str = "float16",
            cache_dir: Optional[str] = None,
            confidence_threshold: float = 0.65,
            use_hybrid_ner: bool = True,
            use_enhanced_temporal: bool = True,
            use_vectorized_contradictions: bool = True,
            max_workers: int = 4,
            logger: Optional[logging.Logger] = None
        ):
        self.logger = logger or logging.getLogger(__name__)
        
        # Feature flags for enabling/disabling enhancements
        self.use_hybrid_ner = use_hybrid_ner
        self.use_enhanced_temporal = use_enhanced_temporal
        self.use_vectorized_contradictions = use_vectorized_contradictions
        
        # Set up device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.compute_type = compute_type
        
        self.logger.info(f"Initializing Enhanced Medical Transcript Summarizer on {device} with {compute_type}")
        
        # Initialize the advanced summarizer
        try:
            if self.use_vectorized_contradictions:
                self.logger.info(f"Loading sentence embedding model: {sentence_model}")
            
            self.summarizer = AdvancedMedicalSummarizer(
                base_model=base_model,
                ner_model=ner_model,
                qa_model=qa_model,
                sentence_model=sentence_model if self.use_vectorized_contradictions else None,
                device=device,
                compute_type=compute_type,
                cache_dir=cache_dir,
                confidence_threshold=confidence_threshold,
                use_enhanced_temporal=use_enhanced_temporal
            )
            self.logger.info("Medical summarizer initialized successfully")
            
            # Initialize thread pool for parallelism
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            self.logger.info(f"Thread pool initialized with {max_workers} workers")
            
        except Exception as e:
            self.logger.error(f"Error initializing medical summarizer: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def process_transcript(
            self,
            transcript_data: Union[str, Dict, List],
            output_path: Optional[str] = None,
            text_column: str = "transcription",
            speaker_column: str = "speaker",
            chunk_size: int = 5000,
            chunk_overlap: int = 500
        ) -> Dict[str, Any]:
        """Process a medical transcript and generate a comprehensive summary using enhanced architecture"""
        self.logger.info(f"Processing medical transcript: {transcript_data}")
        
        try:
            # Process the transcript with the advanced summarizer
            result = self.summarizer.process_medical_transcript(
                transcript_data=transcript_data,
                text_column=text_column,
                speaker_column=speaker_column,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            self.logger.info(f"Medical transcript processing complete")
            
            # Create a simplified summary for easy consumption
            simplified = {
                "summary": result["narrative_summary"],
                "soap_note": result["soap_note"],
                "key_findings": self._extract_key_findings(result),
                "plan": result["soap_note"]["Plan"],
                "processing_stats": result.get("processing_stats", {})
            }
            
            # Save simplified summary if output path is provided
            if output_path:
                simple_output = output_path.replace(".json", "_simple.json")
                with open(simple_output, "w") as f:
                    json.dump(simplified, f, indent=2)
                
                # Also save as text
                text_output = output_path.replace(".json", "_summary.txt")
                with open(text_output, "w") as f:
                    f.write("MEDICAL CONVERSATION SUMMARY\n")
                    f.write("==========================\n\n")
                    f.write(result["narrative_summary"])
                    f.write("\n\n")
                    f.write("SOAP NOTE\n")
                    f.write("=========\n\n")
                    for section, content in result["soap_note"].items():
                        f.write(f"{section}:\n")
                        f.write(f"{content}\n\n")
                
                self.logger.info(f"Saved simplified summary to {simple_output}")
                self.logger.info(f"Saved text summary to {text_output}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing medical transcript: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def process_transcript_async(
            self,
            transcript_data: Union[str, Dict, List],
            callback: callable,
            text_column: str = "transcription",
            speaker_column: str = "speaker"
        ) -> concurrent.futures.Future:
        """Process a medical transcript asynchronously and call the callback when done"""
        future = self.executor.submit(
            self.process_transcript,
            transcript_data=transcript_data,
            output_path=None,
            text_column=text_column,
            speaker_column=speaker_column
        )
        
        future.add_done_callback(
            lambda f: callback(f.result() if not f.exception() else f.exception())
        )
        
        return future
    
    def analyze_conversation_in_real_time(
            self,
            new_segment: Dict[str, Any],
            conversation_context: List[Dict[str, Any]],
            streaming_results_callback: Optional[callable] = None
        ) -> Dict[str, Any]:
        """Real-time analysis of conversation segments as they come in"""
        self.logger.info("Processing new conversation segment")
        
        # Add new segment to conversation context
        updated_context = conversation_context + [new_segment]
        
        try:
            # Process just the new segment
            segment_result = self.summarizer.process_segment(
                segment=new_segment,
                conversation_context=updated_context
            )
            
            # Incorporate the segment results into the ongoing analysis
            cumulative_result = self.summarizer.update_analysis(
                new_segment_result=segment_result,
                conversation_context=updated_context
            )
            
            # Extract key findings for the new segment
            segment_findings = self._extract_key_findings_from_segment(segment_result)
            
            # If a streaming callback is provided, send the intermediate results
            if streaming_results_callback and callable(streaming_results_callback):
                streaming_results_callback({
                    "segment_result": segment_result,
                    "segment_findings": segment_findings,
                    "cumulative_result": cumulative_result
                })
            
            return {
                "segment_result": segment_result,
                "segment_findings": segment_findings,
                "cumulative_result": cumulative_result,
                "updated_context": updated_context
            }
            
        except Exception as e:
            self.logger.error(f"Error in real-time analysis: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _extract_key_findings(self, result):
        """Extract key findings from the result for simplified output"""
        key_findings = []
        
        # Extract critical vitals
        vitals = result["structured_summary"]["vital_signs"]
        if vitals["blood_pressure"]:
            for bp in vitals["blood_pressure"]:
                key_findings.append({
                    "type": "blood_pressure",
                    "value": bp["value"],
                    "importance": "high",
                    "source": bp.get("source", "unknown"),
                    "temporal_context": bp.get("temporal_context", "current"),
                    "confidence": bp.get("confidence", 0.7)
                })
        
        if vitals["blood_glucose"]:
            for glucose in vitals["blood_glucose"]:
                key_findings.append({
                    "type": "blood_glucose",
                    "value": glucose["value"],
                    "importance": "high",
                    "source": glucose.get("source", "unknown"),
                    "temporal_context": glucose.get("temporal_context", "current"),
                    "confidence": glucose.get("confidence", 0.7)
                })
        
        # Extract medications
        medications = result["structured_summary"]["medications"]
        if medications["current"]:
            for med in medications["current"]:
                key_findings.append({
                    "type": "medication",
                    "value": med["medication"],
                    "importance": "high",
                    "temporal_context": "current",
                    "confidence": med.get("confidence", 0.7)
                })
        
        # Extract symptoms
        symptoms = result["structured_summary"]["symptoms"]
        if symptoms["current"]:
            for symptom in symptoms["current"]:
                if not symptom["is_negated"]:
                    key_findings.append({
                        "type": "symptom",
                        "value": symptom["symptom"],
                        "importance": "medium",
                        "temporal_context": symptom.get("temporal_context", "current"),
                        "confidence": symptom.get("confidence", 0.7)
                    })
        
        # Extract contradictions
        if result["contradictions"]:
            for contradiction in result["contradictions"][:3]:
                key_findings.append({
                    "type": "contradiction",
                    "value": contradiction["description"],
                    "importance": "high",
                    "entity_type": contradiction.get("entity_type", "unknown"),
                    "confidence": 0.9  # Contradictions have high confidence by default
                })
        
        # Extract significant changes from timeline
        timeline = result.get("timeline", {})
        if "recent_past" in timeline and timeline["recent_past"]:
            for event in timeline["recent_past"][:3]:
                if "change" in event["value"].lower() or "improved" in event["value"].lower() or "worse" in event["value"].lower():
                    key_findings.append({
                        "type": "change",
                        "value": event["value"],
                        "importance": "high",
                        "temporal_context": "recent_past",
                        "confidence": event.get("confidence", 0.7)
                    })
        
        # Sort findings by importance and confidence
        key_findings.sort(key=lambda x: (-{"high": 3, "medium": 2, "low": 1}.get(x["importance"], 0), 
                                         -x.get("confidence", 0)))
        
        return key_findings
    
    def _extract_key_findings_from_segment(self, segment_result):
        """Extract key findings from a single conversation segment"""
        key_findings = []
        
        # Extract entities from segment
        if "entities" in segment_result:
            for entity_type, entities in segment_result["entities"].items():
                for entity in entities:
                    # Skip negated, uncertain, or low-confidence entities
                    if entity.get("is_negated", False) or entity.get("is_uncertain", False) or entity.get("confidence", 1.0) < 0.6:
                        continue
                    
                    # Add relevant entity types
                    if entity_type in ["blood_pressure", "blood_glucose", "medications", "symptoms"]:
                        key_findings.append({
                            "type": entity_type,
                            "value": entity.get("normalized_value", entity.get("value", "")),
                            "importance": "high" if entity_type in ["blood_pressure", "blood_glucose", "medications"] else "medium",
                            "temporal_context": entity.get("temporal_context", "current"),
                            "confidence": entity.get("confidence", 0.7)
                        })
        
        # Extract contradictions
        if "contradictions" in segment_result and segment_result["contradictions"]:
            for contradiction in segment_result["contradictions"][:2]:
                key_findings.append({
                    "type": "contradiction",
                    "value": contradiction["description"],
                    "importance": "high",
                    "confidence": 0.9
                })
        
        return key_findings

class EnhancedMedicalTranscriptSummarizer(MedicalTranscriptSummarizer):
    """Enhanced version with additional features for medical transcript analysis"""
    
    def __init__(
            self,
            base_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            ner_model: str = "emilyalsentzer/Bio_ClinicalBERT",
            qa_model: str = "dmis-lab/biobert-base-cased-v1.1-squad",
            sentence_model: str = "sentence-transformers/paraphrase-mpnet-base-v2",
            temporal_model: Optional[str] = None,
            terminology_mapping: Optional[str] = None,
            device: Optional[str] = None,
            compute_type: str = "float16",
            cache_dir: Optional[str] = None,
            confidence_threshold: float = 0.65,
            use_hybrid_ner: bool = True,
            use_enhanced_temporal: bool = True,
            use_vectorized_contradictions: bool = True,
            max_workers: int = 4,
            logger: Optional[logging.Logger] = None
        ):
        # Call parent constructor
        super().__init__(
            base_model=base_model,
            ner_model=ner_model,
            qa_model=qa_model,
            sentence_model=sentence_model,
            device=device,
            compute_type=compute_type,
            cache_dir=cache_dir,
            confidence_threshold=confidence_threshold,
            use_hybrid_ner=use_hybrid_ner,
            use_enhanced_temporal=use_enhanced_temporal,
            use_vectorized_contradictions=use_vectorized_contradictions,
            max_workers=max_workers,
            logger=logger
        )
        
        # Additional models and resources
        self.temporal_model_path = temporal_model
        self.terminology_mapping_path = terminology_mapping
        
        # Initialize additional components
        if self.temporal_model_path:
            self.logger.info(f"Loading temporal sequence model: {temporal_model}")
            try:
                self.summarizer.set_temporal_model(temporal_model)
                self.logger.info("Temporal model loaded successfully")
            except Exception as e:
                self.logger.error(f"Error loading temporal model: {e}")
        
        if self.terminology_mapping_path:
            self.logger.info(f"Loading terminology mapping: {terminology_mapping}")
            try:
                self.summarizer.set_terminology_mapping(terminology_mapping)
                self.logger.info("Terminology mapping loaded successfully")
            except Exception as e:
                self.logger.error(f"Error loading terminology mapping: {e}")
    
    def generate_medical_reports(
            self,
            result: Dict[str, Any],
            output_path: Optional[str] = None,
            report_types: List[str] = ["summary", "soap", "timeline", "followup"]
        ) -> Dict[str, Any]:
        """Generate various medical reports from analysis results"""
        self.logger.info("Generating medical reports")
        
        reports = {}
        
        try:
            if "summary" in report_types:
                reports["summary"] = result["narrative_summary"]
            
            if "soap" in report_types:
                reports["soap"] = result["soap_note"]
            
            if "timeline" in report_types:
                # Generate enhanced timeline visualization
                timeline_report = self._generate_timeline_report(result["timeline"])
                reports["timeline"] = timeline_report
            
            if "followup" in report_types:
                # Generate follow-up recommendations
                followup_report = self._generate_followup_recommendations(result)
                reports["followup"] = followup_report
            
            # Save reports if output path is provided
            if output_path:
                reports_dir = os.path.dirname(output_path)
                os.makedirs(reports_dir, exist_ok=True)
                
                for report_type, report_content in reports.items():
                    report_path = os.path.join(reports_dir, f"{os.path.basename(output_path).split('.')[0]}_{report_type}.txt")
                    
                    with open(report_path, "w") as f:
                        if isinstance(report_content, dict):
                            # Format dictionary content
                            for key, value in report_content.items():
                                f.write(f"{key}:\n{value}\n\n")
                        else:
                            # Write string content directly
                            f.write(report_content)
                    
                    self.logger.info(f"Saved {report_type} report to {report_path}")
            
            return reports
            
        except Exception as e:
            self.logger.error(f"Error generating medical reports: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _generate_timeline_report(self, timeline):
        """Generate a formatted timeline report"""
        report = []
        
        report.append("MEDICAL TIMELINE\n")
        report.append("===============\n\n")
        
        # Past history
        if "past_history" in timeline and timeline["past_history"]:
            report.append("PAST MEDICAL HISTORY:\n")
            for event in timeline["past_history"]:
                report.append(f"- {event['value']}")
            report.append("\n")
        
        # Recent past
        if "recent_past" in timeline and timeline["recent_past"]:
            report.append("RECENT HISTORY:\n")
            for event in timeline["recent_past"]:
                report.append(f"- {event['value']}")
            report.append("\n")
        
        # Current
        if "current" in timeline and timeline["current"]:
            report.append("CURRENT STATUS:\n")
            for event in timeline["current"]:
                report.append(f"- {event['value']}")
            report.append("\n")
        
        # Future plans
        if ("immediate_future" in timeline and timeline["immediate_future"]) or ("distant_future" in timeline and timeline["distant_future"]):
            report.append("PLANNED CARE:\n")
            
            if "immediate_future" in timeline:
                for event in timeline["immediate_future"]:
                    report.append(f"- {event['value']}")
            
            if "distant_future" in timeline:
                for event in timeline["distant_future"]:
                    report.append(f"- {event['value']} (long term)")
        
        return "\n".join(report)
    
    def _generate_followup_recommendations(self, result):
        """Generate follow-up recommendations based on analysis results"""
        recommendations = []
        
        recommendations.append("FOLLOW-UP RECOMMENDATIONS\n")
        recommendations.append("=========================\n\n")
        
        # Include plan from SOAP note
        if "soap_note" in result and "Plan" in result["soap_note"]:
            recommendations.append("PLAN FROM MEDICAL ASSESSMENT:\n")
            recommendations.append(result["soap_note"]["Plan"])
            recommendations.append("\n")
        
        # Include specific follow-up recommendations
        if "structured_summary" in result and "plan" in result["structured_summary"]:
            plan = result["structured_summary"]["plan"]
            
            if "follow_up" in plan and plan["follow_up"]:
                recommendations.append("FOLLOW-UP APPOINTMENTS:\n")
                for follow_up in plan["follow_up"]:
                    recommendations.append(f"- {follow_up.get('instruction', '')}")
                recommendations.append("\n")
            
            if "monitoring" in plan and plan["monitoring"]:
                recommendations.append("MONITORING RECOMMENDATIONS:\n")
                for monitoring in plan["monitoring"]:
                    recommendations.append(f"- {monitoring.get('instruction', '')}")
                recommendations.append("\n")
        
        # Add reminders based on key conditions
        if "structured_summary" in result and "health_status" in result["structured_summary"]:
            health_status = result["structured_summary"]["health_status"]
            
            if "chronic_conditions" in health_status and health_status["chronic_conditions"]:
                for condition in health_status["chronic_conditions"]:
                    condition_name = condition.get("condition", "").lower()
                    
                    if "diabetes" in condition_name:
                        recommendations.append("DIABETES MANAGEMENT REMINDERS:\n")
                        recommendations.append("- Continue regular blood glucose monitoring")
                        recommendations.append("- Schedule regular HbA1c testing (every 3-6 months)")
                        recommendations.append("- Annual eye examination recommended")
                        recommendations.append("- Regular foot examinations")
                        recommendations.append("\n")
                    
                    if "hypertension" in condition_name or "high blood pressure" in condition_name:
                        recommendations.append("HYPERTENSION MANAGEMENT REMINDERS:\n")
                        recommendations.append("- Continue regular blood pressure monitoring")
                        recommendations.append("- Maintain low sodium diet")
                        recommendations.append("- Regular cardiovascular check-ups")
                        recommendations.append("\n")
        
        return "\n".join(recommendations)

# Factory function to create the appropriate summarizer based on settings
def create_medical_summarizer(
        enhanced: bool = True,
        base_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        ner_model: str = "emilyalsentzer/Bio_ClinicalBERT",
        qa_model: str = "dmis-lab/biobert-base-cased-v1.1-squad",
        sentence_model: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        device: Optional[str] = None,
        compute_type: str = "float16",
        cache_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ) -> Union[MedicalTranscriptSummarizer, EnhancedMedicalTranscriptSummarizer]:
    """Create the appropriate medical summarizer based on settings"""
    if enhanced:
        return EnhancedMedicalTranscriptSummarizer(
            base_model=base_model,
            ner_model=ner_model,
            qa_model=qa_model,
            sentence_model=sentence_model,
            device=device,
            compute_type=compute_type,
            cache_dir=cache_dir,
            logger=logger,
            **kwargs
        )
    else:
        return MedicalTranscriptSummarizer(
            base_model=base_model,
            ner_model=ner_model,
            qa_model=qa_model,
            sentence_model=sentence_model,
            device=device,
            compute_type=compute_type,
            cache_dir=cache_dir,
            logger=logger,
            **kwargs
        )