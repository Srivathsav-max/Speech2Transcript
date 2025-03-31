"""
Transcript summarization modules for extracting structured information
from speech-to-text transcriptions and generating detailed summaries.

This package provides advanced transcript summarization:
- BaseSummarizer: Abstract base class for all summarizers
- AdvancedMedicalSummarizer: Enterprise-grade medical summarizer with optional NLP capabilities 
- GeminiSummarizer: LLM-powered summarizer using Google's Gemini API
"""

# Base class
from .base_summarizer import BaseSummarizer

# Import the advanced medical summarizer (previous implementation)
from .advanced_summarizer import AdvancedMedicalSummarizer

# Import Gemini-powered summarizer
from .gemini_summarizer import GeminiSummarizer

# For backward compatibility, alias the TransformerMedicalSummarizer to AdvancedMedicalSummarizer
TransformerMedicalSummarizer = AdvancedMedicalSummarizer

__all__ = [
    'BaseSummarizer',
    'AdvancedMedicalSummarizer',
    'TransformerMedicalSummarizer',  # For backward compatibility
    'GeminiSummarizer'
]