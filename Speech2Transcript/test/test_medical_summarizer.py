"""
Test script for the enhanced medical transcript processing system.

This script demonstrates how to use the new modular architecture to process
a medical transcript and generate structured information and formatted notes.
"""
import os
import sys
import json
import logging

# Add parent directory to path to allow importing Speech2Transcript
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the processor
from Speech2Transcript.summarizers import MedicalTranscriptProcessor
from Speech2Transcript.summarizers import SpeakerIdentifier, VitalSignExtractor, MedicationExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('medical_summarizer_test')

def test_process_transcript(transcript_path, output_dir, template_path=None):
    """
    Process a transcript file and generate outputs.
    
    Args:
        transcript_path: Path to the transcript JSON file
        output_dir: Directory to save outputs
        template_path: Optional path to telehealth note template
    """
    logger.info(f"Processing transcript: {transcript_path}")
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the processor
    processor = MedicalTranscriptProcessor(
        ner_model="emilyalsentzer/Bio_ClinicalBERT",
        device=None,  # Auto-detect
        logger=logger
    )
    
    # Process the transcript
    basename = os.path.splitext(os.path.basename(transcript_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_summary.json")
    
    results = processor.process_transcript(
        transcript_path=transcript_path,
        output_path=output_path,
        template_path=template_path
    )
    
    # Print summary of results
    logger.info("=" * 60)
    logger.info("Processing Results:")
    logger.info("=" * 60)
    
    # Patient and care manager
    logger.info(f"Patient: {results['patient_info']['patient_name']}")
    logger.info(f"Care Manager: {results['patient_info']['care_manager_name']}")
    
    # Health status
    logger.info("\nHealth Status:")
    if results['health_status']['has_symptoms']:
        logger.info(f"Symptoms: {results['health_status']['symptom_text']}")
    else:
        logger.info("No symptoms reported")
    
    # Conditions
    if results['health_status'].get('conditions'):
        logger.info("\nConditions:")
        for condition in results['health_status']['conditions']:
            logger.info(f"- {condition['name']}")
    
    # Vital signs
    vitals = results['health_status']['vital_signs']
    logger.info("\nVital Signs:")
    if vitals['blood_pressure']:
        bp = vitals['blood_pressure'][0]['full']
        logger.info(f"- BP: {bp}")
    if vitals['glucose']:
        glucose = vitals['glucose'][0]['value']
        logger.info(f"- Glucose: {glucose} mg/dL")
    
    # Medications
    if results['medications'].get('medications'):
        logger.info("\nMedications:")
        for med in results['medications']['medications']:
            logger.info(f"- {med['name']}")
    
    logger.info(f"\nAdherence: {results['medications'].get('adherence', 'Unknown')}")
    
    # Plan
    logger.info("\nPlan:")
    follow_up = results['plan'].get('follow_up', {})
    if isinstance(follow_up, dict) and follow_up.get('timeframe'):
        logger.info(f"Follow-up: {follow_up.get('type', 'appointment')} in {follow_up['timeframe']}")
    
    logger.info("\nOutputs saved to:")
    logger.info(f"- JSON Summary: {output_path}")
    logger.info(f"- Text Summary: {output_path.replace('.json', '_summary.txt')}")
    logger.info(f"- Telehealth Note: {output_path.replace('.json', '_telehealth_note.txt')}")

def test_individual_extractors(transcript_path):
    """
    Demonstrate using individual extractors directly.
    
    Args:
        transcript_path: Path to the transcript JSON file
    """
    logger.info("\n" + "=" * 60)
    logger.info("Testing Individual Extractors:")
    logger.info("=" * 60)
    
    # Load the transcript
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    
    segments = transcript_data.get('segments', [])
    if not segments:
        logger.error("No segments found in transcript")
        return
    
    # Extract all text for processing
    all_text = " ".join([segment.get("transcription", "") for segment in segments 
                         if "transcription" in segment])
    
    # Test speaker identifier
    speaker_identifier = SpeakerIdentifier(logger=logger)
    speaker_info = speaker_identifier.extract(segments)
    
    logger.info("\nSpeaker Identification Results:")
    logger.info(f"Care Manager: {speaker_info['speakers']['care_manager']}")
    logger.info(f"Patient: {speaker_info['speakers']['patient']}")
    logger.info(f"Care Manager Name: {speaker_info['names']['care_manager_name']}")
    logger.info(f"Patient Name: {speaker_info['names']['patient_name']}")
    
    # Test vital sign extractor
    vital_extractor = VitalSignExtractor(logger=logger)
    vitals = vital_extractor.extract(all_text)
    
    logger.info("\nVital Sign Extraction Results:")
    if vitals['blood_pressure']:
        for bp in vitals['blood_pressure']:
            logger.info(f"Blood Pressure: {bp['full']} (Context: '{bp['context'][:30]}...')")
    
    if vitals['glucose']:
        for glucose in vitals['glucose']:
            logger.info(f"Glucose: {glucose['value']} mg/dL (Context: '{glucose['context'][:30]}...')")
    
    # Test medication extractor
    med_extractor = MedicationExtractor(logger=logger)
    med_info = med_extractor.extract(all_text)
    
    logger.info("\nMedication Extraction Results:")
    for med in med_info['medications']:
        logger.info(f"Medication: {med['name']}")
        if med.get('dosage'):
            logger.info(f"  Dosage: {med['dosage']}")
        if med.get('frequency'):
            logger.info(f"  Frequency: {med['frequency']}")
        if med.get('category'):
            logger.info(f"  Category: {med['category']}")
    
    logger.info(f"\nAdherence: {med_info['adherence']}")
    logger.info(f"Side Effects: {med_info['side_effects']}")

if __name__ == "__main__":
    # Default paths - adjust as needed
    transcript_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "outputs", 
        "audio1_diarization.json"
    )
    
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "outputs"
    )
    
    template_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "templates",
        "telehealth_template.txt"
    )
    
    # Process the transcript
    test_process_transcript(transcript_path, output_dir, template_path)
    
    # Demonstrate individual extractors
    test_individual_extractors(transcript_path)
