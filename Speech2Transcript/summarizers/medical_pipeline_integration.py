"""
Medical transcript summarization with enhanced processing for telehealth conversations.

This module provides backward compatibility with the original MedicalTranscriptSummarizer
while using the new modular architecture for improved accuracy and maintainability.
"""
import os
import torch
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Union

from .medical_transcript_processor import MedicalTranscriptProcessor

class MedicalTranscriptSummarizer:
    """
    A comprehensive medical conversation summarization pipeline that extracts structured
    information from healthcare conversations to populate telehealth progress notes.
    
    This implementation uses the newer modular architecture internally while maintaining
    the same interface as the original class for backward compatibility.
    """
    
    def __init__(
        self,
        base_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        ner_model: str = "emilyalsentzer/Bio_ClinicalBERT",
        qa_model: str = "dmis-lab/biobert-base-cased-v1.1-squad",
        device: str = None,
        compute_type: str = "float16",
        cache_dir: str = None,
        confidence_threshold: float = 0.65,
        logger = None
    ):
        """Initialize the medical summarization pipeline with appropriate models.

        Args:
            base_model: Model name for general medical language understanding
            ner_model: Model name for NER (medical entity extraction)
            qa_model: Model name for QA (targeted information extraction)
            device: 'cuda', 'cpu', or None (auto-detect)
            compute_type: Computation precision ("float16", "float32", "int8")
            cache_dir: Directory to cache downloaded models
            confidence_threshold: Minimum confidence for entity extraction
            logger: Optional logger for messages
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        
        # Store original parameters for logging purposes
        self.base_model = base_model
        self.ner_model = ner_model
        self.qa_model = qa_model
        self.compute_type = compute_type
        self.confidence_threshold = confidence_threshold
        
        # Log initialization
        self._log(f"Initializing Medical Transcript Summarizer")
        self._log(f"Using device: {self.device}, compute type: {compute_type}")
        
        # Initialize the new processor that handles all the work
        self.processor = MedicalTranscriptProcessor(
            ner_model=ner_model,
            device=device,
            compute_type=compute_type,
            cache_dir=cache_dir,
            confidence_threshold=confidence_threshold,
            logger=logger
        )
    
    def _log(self, message, level="info"):
        """Log messages if logger is available."""
        if self.logger:
            if level == "info":
                self.logger.info(message)
            elif level == "error":
                self.logger.error(message)
            elif level == "warning":
                self.logger.warning(message)
    
    def process_transcript(
        self,
        transcript_path: str = None,
        transcript_data: Dict = None,
        output_path: str = None,
        text_column: str = "transcription",
        speaker_column: str = "speaker"
    ) -> Dict:
        """
        Process a conversation transcript to extract medical information.

        Args:
            transcript_path: Path to transcript file (JSON)
            transcript_data: Alternatively, provide transcript data directly
            output_path: Optional path to save results
            text_column: Column name containing the transcription text
            speaker_column: Column name containing the speaker ID

        Returns:
            Dictionary with structured clinical information
        """
        # Delegate to the new processor
        return self.processor.process_transcript(
            transcript_path=transcript_path,
            transcript_data=transcript_data,
            output_path=output_path,
            text_column=text_column,
            speaker_column=speaker_column
        )
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        output_path: str = None,
        text_column: str = "transcription",
        speaker_column: str = "speaker"
    ) -> Dict:
        """
        Process transcript data from a pandas DataFrame.
        
        Args:
            df: DataFrame containing transcript data
            output_path: Optional path to save results
            text_column: Column name containing the transcription text
            speaker_column: Column name containing the speaker ID
            
        Returns:
            Dictionary with structured clinical information
        """
        # Convert DataFrame to the expected format
        segments = []
        for _, row in df.iterrows():
            segment = {
                text_column: row.get(text_column, ""),
                speaker_column: row.get(speaker_column, "")
            }
            segments.append(segment)
        
        transcript_data = {"segments": segments}
        
        # Delegate to the main processing method
        return self.process_transcript(
            transcript_data=transcript_data,
            output_path=output_path,
            text_column=text_column,
            speaker_column=speaker_column
        )
    
    def fill_template(self, results: Dict, template_path: str) -> str:
        """
        Fill a provided template with extracted medical data.
        
        Args:
            results: Extracted medical information
            template_path: Path to template file
            
        Returns:
            Filled template text
        """
        # Generate a new telehealth note with the provided template
        return self.processor.note_generator.generate_telehealth_note(results, template_path)
