"""
Transcript summarization modules for extracting structured information
from speech-to-text transcriptions and generating detailed summaries.

This package provides:
- BaseSummarizer: Abstract base class for all summarizers
- EnterpriseSummarizer: Enterprise-grade, HIPAA-compliant medical summarizer
- GeminiSummarizer: LLM-powered summarizer using Google's Gemini API
- SmartHIPAAProcessor: Advanced HIPAA compliance with intelligent redaction
- ClinicalNarrativeGenerator: Professional medical documentation generator
- TelemedicalDetector: Utilities for identifying healthcare-related content
"""

# Base class
from .base_summarizer import BaseSummarizer

# Import enterprise summarizer components
from .enterprise_summarizer import EnterpriseSummarizer
from .clinical_narrative import ClinicalNarrativeGenerator
from .smart_hipaa import SmartHIPAAProcessor


__all__ = [
    'BaseSummarizer',
    'EnterpriseSummarizer',
    'ClinicalNarrativeGenerator',
    'SmartHIPAAProcessor',
]