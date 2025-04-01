# Speech2Transcript

A comprehensive speech processing system for transcription, diarization, and medical summarization.

## Current Architecture Implementation Progress
![image](https://github.com/user-attachments/assets/5967bbdb-5f6a-4833-979c-e3c978301dc8)

## Overview

Speech2Transcript is a Python toolkit for processing speech audio, performing speaker diarization, transcription, and advanced summarization particularly focused on medical conversations. The system can work with both pre-recorded audio files and real-time audio input.

## Getting Started

For quick setup instructions, see [SETUP.md](SETUP.md).

To verify your installation, run the test script:
```bash
python test_setup.py --audio audios/audio1.wav
```

## Key Features

- **Speaker Diarization**: Identify different speakers in audio using pyannote.audio
- **Speech Transcription**: Convert speech to text using faster-whisper
- **Real-time Processing**: Capture and process live audio
- **Medical Conversation Analysis**: Enhanced processing for medical and telehealth conversations
- **Multiple Output Formats**: JSON, CSV, RTTM formats supported
- **Customizable Templates**: Generate formatted medical notes using templates

## Core Components

- **DiarizationPipeline**: Speaker identification
- **TranscriptionPipeline**: Speech-to-text conversion
- **RealTimeSTT**: Real-time audio processing
- **MedicalTranscriptProcessor**: Advanced medical conversation analysis

## Transcript Summarization

The system offers several approaches to transcript summarization:

### Advanced Transformer-based Medical Summarization
- **State-of-the-art NLP**: Uses transformer models for high-accuracy information extraction
- **Medical Entity Recognition**: Specialized medical entity recognition with confidence scores
- **Question Answering**: Uses QA models to extract precise information from patient-provider dialogue
- **Sentiment Analysis**: Analyzes patient sentiment regarding medications, treatment, and health status
- **Rich Dialogue Summarization**: Generates coherent and contextual summaries of clinical conversations
- **Health Status Assessment**: Automated assessment of patient health with concerns and positives
- **Confidence Metrics**: Provides confidence scores for extracted information to assist with validation

### Gemini-powered Narrative Summarization
- **LLM-powered Natural Summaries**: Leverages Google's Gemini API for high-quality, coherent summaries
- **Conversational Understanding**: Produces summaries that capture the nuance and flow of the dialogue
- **Narrative Structure**: Creates summaries that read like a medical note with natural paragraph structure
- **Contextual Integration**: Integrates information across the entire conversation for comprehensive coverage
- **Medical Language**: Uses professional medical terminology and structure appropriate for healthcare documentation

### HIPAA Compliance Features
- **Protected Health Information (PHI) Detection**: Automatically detects and redacts PHI in transcripts
- **De-identification**: Replaces identifiable information with generic placeholders
- **Compliance Verification**: Ensures summaries adhere to HIPAA requirements
- **Minimal Necessary Information**: Extracts only clinically relevant details
- **Secure Output**: Generates HIPAA-compliant medical documentation

### Enhanced Nurse-Style Documentation
- **SOAP Format**: Structures summaries in Subjective, Objective, Assessment, Plan format
- **Professional Clinical Terminology**: Uses proper medical abbreviations and phrasing
- **Comprehensive Assessment**: Detailed evaluation organized by body systems
- **Clear Treatment Plans**: Well-structured follow-up instructions and recommendations
- **Realistic Nursing Narrative**: Documentation that reads as if written by a healthcare professional

## Installation

1. Clone this repository
```bash
git clone https://github.com/yourusername/Speech2Transcript.git
cd Speech2Transcript
```

2. Install requirements
```bash
pip install -r requirements.txt
```

## Usage Examples

### Process Audio File with Diarization and Transcription

```python
from Speech2Transcript.diarization import DiarizationPipeline
from Speech2Transcript.transcription import TranscriptionPipeline
from Speech2Transcript.utils import export_to_json

# Initialize diarization
diarization = DiarizationPipeline()

# Process audio
diarization_results = diarization.process_audio("path/to/audio.wav")

# Initialize transcription
transcription = TranscriptionPipeline()

# Process diarization results
results = transcription.process_diarization(
    diarization_results,
    "path/to/audio.wav"
)

# Export to JSON
export_to_json(results, "output.json")
```

### Transcript Summarization

```python
from Speech2Transcript.summarizers import SimpleSummarizer, DetailedSummarizer, TransformerMedicalSummarizer

# ---- Simple Summarization ----
# Initialize simple summarizer
simple_summarizer = SimpleSummarizer()

# Process a transcript
simple_results = simple_summarizer.process_transcript(
    transcript_path="path/to/transcript.json",
    output_path="path/to/simple_summary.json"
)

# Access the simple summary
print(f"Summary: {simple_results['summary']}")
print(f"Number of speakers: {simple_results['speaker_count']}")

# ---- Detailed Summarization ----
# Initialize detailed summarizer
detailed_summarizer = DetailedSummarizer()

# Process a transcript with comprehensive analysis
detailed_results = detailed_summarizer.process_transcript(
    transcript_path="path/to/transcript.json",
    output_path="path/to/detailed_summary.json"
)

# Access detailed summary information
print(f"Detailed Summary: {detailed_results['detailed_summary']}")

# Access structured data
extracted_data = detailed_results['extracted_data']
print(f"Patient: {extracted_data['patient_info']['name']}")
print(f"Conditions: {extracted_data['conditions']}")
print(f"Medications: {[m['name'] for m in extracted_data['medications']['medications']]}")

# ---- Advanced Transformer-based Medical Summarization ----
# Initialize transformer-based summarizer with custom models (optional)
transformer_summarizer = TransformerMedicalSummarizer(
    ner_model="emilyalsentzer/Bio_ClinicalBERT",
    qa_model="distilbert-base-cased-distilled-squad",
    sentiment_model="bhadresh-savani/distilbert-base-uncased-emotion",
    summarization_model="philschmid/bart-large-cnn-samsum"
)

# Process a transcript with advanced NLP techniques
transformer_results = transformer_summarizer.process_transcript(
    transcript_path="path/to/transcript.json",
    output_path="path/to/transformer_summary.json"
)

# Access dialogue summary
print(f"Dialogue Summary: {transformer_results['extracted_data']['dialogue_summary']}")

# Access medical entities with confidence scores
entities = transformer_results['extracted_data']['medical_entities']
print(f"Conditions: {[(c['name'], c['confidence']) for c in entities['conditions']]}")
print(f"Medications: {[(m['name'], m['confidence']) for m in entities['medications']]}")

# Access sentiment analysis and health assessment
print(f"Medication adherence: {transformer_results['extracted_data']['medication_adherence']['status']}")
print(f"Sentiment: {transformer_results['extracted_data']['medication_adherence']['sentiment']['primary']}")
print(f"Health status: {transformer_results['extracted_data']['health_assessment']['status']}")
```

### Real-time Processing

```python
from Speech2Transcript.diarization import DiarizationPipeline
from Speech2Transcript.transcription import TranscriptionPipeline
from Speech2Transcript.realtimestt import RealTimeSTT

# Initialize components
diarization = DiarizationPipeline()
transcription = TranscriptionPipeline()

# Create real-time processor
realtime = RealTimeSTT(
    diarization_pipeline=diarization,
    transcription_pipeline=transcription,
    output_dir="./outputs"
)

# Start processing (will use default microphone)
realtime.start()

# To stop processing (e.g., in another thread or after some time)
# realtime.stop()
```

## Command Line Usage

```bash
# Process an audio file with diarization and transcription
python -m Speech2Transcript.main --audio /path/to/audio.wav --diarize --transcribe --output ./outputs

# Use real-time processing
python -m Speech2Transcript.main --realtime --output ./outputs

# Generate a traditional extraction-based summary from a transcript
python -m Speech2Transcript.main --summarize --transcript_file /path/to/transcript.json --output ./outputs

# Generate a detailed comprehensive summary with structured information
python -m Speech2Transcript.main --summarize --detailed --transcript_file /path/to/transcript.json --output ./outputs

# Generate a narrative summary using Gemini API (requires API key)
python -m Speech2Transcript.main --summarize --gemini-api-key YOUR_API_KEY --transcript_file /path/to/transcript.json --output ./outputs

# Generate a nurse-like clinical summary with HIPAA compliance
python -m Speech2Transcript.main --summarize --transcript_file /path/to/transcript.json --output ./outputs

# Generate a standard summary without nurse formatting
python -m Speech2Transcript.main --summarize --disable-nurse-style --transcript_file /path/to/transcript.json --output ./outputs

# Generate a summary without HIPAA compliance features
python -m Speech2Transcript.main --summarize --disable-hipaa --transcript_file /path/to/transcript.json --output ./outputs

# Alternatively, set the GOOGLE_API_KEY environment variable
export GOOGLE_API_KEY=your_api_key_here
python -m Speech2Transcript.main --summarize --transcript_file /path/to/transcript.json --output ./outputs

# Run standalone Gemini test
python -m Speech2Transcript.gemini_test --transcript /path/to/transcript.json --api-key YOUR_API_KEY
```

## License

[Your License Information]
