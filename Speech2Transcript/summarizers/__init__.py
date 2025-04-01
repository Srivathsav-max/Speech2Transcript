"""
Transcript summarization modules for extracting structured information
from speech-to-text transcriptions and generating detailed summaries.

This package provides:
- BaseSummarizer: Abstract base class for all summarizers
- GeminiSummarizer: Enterprise-grade LLM-powered summarizer using Google's Gemini API
- TelemedicalDetector: Utilities for identifying healthcare-related content
"""

# Base class
from .base_summarizer import BaseSummarizer

# Import Gemini-powered summarizer
from .gemini_summarizer import GeminiSummarizer

# Import telemedical detection utilities
from .telemedical_detector import (
    detect_telemedical_content,
    analyze_conversation,
    get_non_telemedical_message
)

__all__ = [
    'BaseSummarizer',
    'GeminiSummarizer',
    'detect_telemedical_content',
    'analyze_conversation',
    'get_non_telemedical_message'
]