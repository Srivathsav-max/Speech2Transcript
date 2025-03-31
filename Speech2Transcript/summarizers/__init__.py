"""
Transcript summarization modules for extracting structured information
from speech-to-text transcriptions and generating detailed summaries.

This package provides transcript summarization:
- BaseSummarizer: Abstract base class for all summarizers
- GeminiSummarizer: Enterprise-grade LLM-powered summarizer using Google's Gemini API
"""

# Base class
from .base_summarizer import BaseSummarizer

# Import Gemini-powered summarizer
from .gemini_summarizer import GeminiSummarizer

__all__ = [
    'BaseSummarizer',
    'GeminiSummarizer'
]