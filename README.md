# Speech2Transcript

A comprehensive speech processing system for transcription, diarization, and medical summarization.

## Current Architecture Implementation Progress
![image](https://github.com/user-attachments/assets/5967bbdb-5f6a-4833-979c-e3c978301dc8)

## Overview

Speech2Transcript is a Python toolkit for processing speech audio, performing speaker diarization, transcription, and advanced summarization particularly focused on medical conversations. The system can work with both pre-recorded audio files and real-time audio input.

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

## Enhanced Medical Summarization

The medical summarization system has been redesigned with a modular architecture for better accuracy, maintainability, and extensibility:

- **Specialized Extractors**: Individual components for extracting different types of medical information
- **Improved Speaker Identification**: Better differentiation between care managers and patients
- **Context-aware Extraction**: More accurate extraction of vital signs, medications, conditions, etc.
- **Dynamic Note Generation**: Flexible generation of SOAP notes and telehealth progress notes

### Medical Summarization Architecture

The medical summarization system consists of these key components:

```
MedicalTranscriptProcessor (Main coordinator)
├── SpeakerIdentifier (Identifies care manager and patient)
├── VitalSignExtractor (Extracts BP, glucose, weight, etc.)
├── MedicationExtractor (Extracts medications and adherence info)
├── ConditionSymptomExtractor (Extracts conditions and symptoms)
└── NoteGenerator (Generates SOAP notes and telehealth notes)
```

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

### Medical Transcript Analysis

```python
from Speech2Transcript.summarizers import MedicalTranscriptProcessor

# Initialize processor
processor = MedicalTranscriptProcessor()

# Process a transcript
results = processor.process_transcript(
    transcript_path="path/to/transcript.json",
    output_path="path/to/output.json",
    template_path="path/to/template.txt"
)

# Access structured data
print(f"Patient: {results['patient_info']['patient_name']}")
print(f"Medications: {[m['name'] for m in results['medications']['medications']]}")

# Access generated notes
soap_note = results["soap_note"]
telehealth_note = results["telehealth_note"]
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

# Generate a medical summary from a transcript
python -m Speech2Transcript.main --advanced_medical_summary --transcript_file /path/to/transcript.json --output ./outputs
```

## License

[Your License Information]
