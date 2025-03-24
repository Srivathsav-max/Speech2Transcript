"""
Medical transcript summarization modules for extracting structured information
from speech-to-text transcriptions and generating medical documentation.
"""

# Import the main processor for easy access
from .medical_transcript_processor import MedicalTranscriptProcessor

# Import specialized extractors for direct use if needed
from .speaker_identifier import SpeakerIdentifier
from .vital_sign_extractor import VitalSignExtractor
from .medication_extractor import MedicationExtractor
from .condition_symptom_extractor import ConditionSymptomExtractor
from .note_generator import NoteGenerator
from .base_extractor import BaseExtractor

# Import utility functions from telehealth_template
from .telehealth_template import generate_telehealth_note

# For backward compatibility, provide the MedicalTranscriptSummarizer class
# This will use the new system but maintain the old interface
from .medical_pipeline_integration import MedicalTranscriptSummarizer

__all__ = [
    'MedicalTranscriptProcessor',
    'SpeakerIdentifier', 
    'VitalSignExtractor',
    'MedicationExtractor',
    'ConditionSymptomExtractor',
    'NoteGenerator',
    'BaseExtractor',
    'generate_telehealth_note',
    'MedicalTranscriptSummarizer'
]
