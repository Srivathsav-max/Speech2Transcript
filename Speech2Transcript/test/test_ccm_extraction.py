#!/usr/bin/env python
"""
Test script for the CCM template extraction and generation.

This script demonstrates the new NLP-based CCM (Chronic Care Management) template 
extraction functionality for medical transcript processing.
"""
import os
import sys
import json
import logging

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Speech2Transcript.summarizers import MedicalTranscriptSummarizer
from Speech2Transcript.summarizers import CCMTemplateExtractor
from Speech2Transcript.summarizers import generate_ccm_note

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CCM_TEST")

def test_ccm_extraction(transcript_path, output_dir=None, ccm_template_path=None):
    """
    Test the CCM template extraction on a transcript.
    
    Args:
        transcript_path: Path to the transcript JSON file
        output_dir: Directory to save the output files
        ccm_template_path: Path to the CCM template file
    """
    logger.info(f"Testing CCM extraction on transcript: {transcript_path}")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the summarizer
    logger.info("Initializing summarizer...")
    summarizer = MedicalTranscriptSummarizer(
        logger=logger
    )
    
    # Process the transcript
    logger.info("Processing transcript...")
    results = summarizer.process_transcript(
        transcript_path=transcript_path,
        output_path=os.path.join(output_dir, "transcript_results.json") if output_dir else None,
        ccm_template_path=ccm_template_path
    )
    
    # Print some key information
    if results:
        logger.info("=" * 50)
        logger.info("CCM EXTRACTION RESULTS SUMMARY:")
        logger.info("=" * 50)
        
        # Print patient info
        patient_info = results.get("patient_info", {})
        logger.info(f"Patient: {patient_info.get('patient_name', 'Unknown')}")
        logger.info(f"Care Manager: {patient_info.get('care_manager_name', 'Unknown')}")
        
        # Print CCM data summary
        ccm_data = results.get("ccm_data", {})
        
        # Chronic conditions
        conditions = ccm_data.get("chronic_conditions", [])
        logger.info(f"Chronic conditions extracted: {len(conditions)}")
        for condition in conditions:
            logger.info(f"- {condition.get('name')}")
        
        # Medication management
        med_management = ccm_data.get("medication_management", {})
        medications = med_management.get("medications", [])
        logger.info(f"Medications extracted: {len(medications)}")
        for med in medications:
            logger.info(f"- {med.get('name')} ({med.get('dosage', 'N/A')}, {med.get('frequency', 'N/A')})")
        
        # Time spent
        time_spent = ccm_data.get("time_spent", {})
        logger.info(f"Time spent: {time_spent.get('total_minutes', 'Unknown')} minutes")
        
        # Billing code
        billing_codes = ccm_data.get("billing_codes", {})
        logger.info(f"Suggested billing codes: {', '.join(billing_codes.get('suggested_codes', ['None']))}")
        
        # Save CCM note to file
        if output_dir and "ccm_note" in results:
            ccm_note_path = os.path.join(output_dir, "ccm_note.txt")
            with open(ccm_note_path, "w") as f:
                f.write(results["ccm_note"])
            logger.info(f"CCM note saved to: {ccm_note_path}")
        
        logger.info("=" * 50)
        return True
    else:
        logger.error("Failed to process transcript")
        return False

def test_standalone_ccm_extraction(transcript_path, output_dir=None, ccm_template_path=None):
    """
    Test the standalone CCM template extractor and generator.
    
    Args:
        transcript_path: Path to the transcript JSON file
        output_dir: Directory to save the output files
        ccm_template_path: Path to the CCM template file
    """
    logger.info(f"Testing standalone CCM extraction on transcript: {transcript_path}")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load transcript data
    try:
        with open(transcript_path, 'r') as f:
            transcript_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading transcript: {e}")
        return False
    
    # Extract segments
    segments = transcript_data.get("segments", [])
    if not segments:
        logger.error("No segments found in transcript")
        return False
    
    # Combine text
    all_text = " ".join([segment.get("transcription", "") for segment in segments if "transcription" in segment])
    
    # Initialize CCM extractor
    logger.info("Initializing CCM extractor...")
    ccm_extractor = CCMTemplateExtractor(
        logger=logger
    )
    
    # Extract CCM data
    logger.info("Extracting CCM data...")
    ccm_data = ccm_extractor.extract(all_text, segments=segments)
    
    # Create a dummy results dictionary
    results = {
        "patient_info": {
            "patient_name": "Test Patient",
            "care_manager_name": "Test Provider"
        },
        "ccm_data": ccm_data
    }
    
    # Generate CCM note
    logger.info("Generating CCM note...")
    ccm_note = generate_ccm_note(results, ccm_template_path, logger)
    
    # Save CCM note to file
    if output_dir:
        ccm_note_path = os.path.join(output_dir, "standalone_ccm_note.txt")
        with open(ccm_note_path, "w") as f:
            f.write(ccm_note)
        logger.info(f"Standalone CCM note saved to: {ccm_note_path}")
    
    # Print summary
    logger.info("=" * 50)
    logger.info("STANDALONE CCM EXTRACTION SUMMARY:")
    logger.info("=" * 50)
    
    # Chronic conditions
    conditions = ccm_data.get("chronic_conditions", [])
    logger.info(f"Chronic conditions extracted: {len(conditions)}")
    for condition in conditions:
        logger.info(f"- {condition.get('name')}")
    
    # Medication management
    med_management = ccm_data.get("medication_management", {})
    medications = med_management.get("medications", [])
    logger.info(f"Medications extracted: {len(medications)}")
    for med in medications:
        logger.info(f"- {med.get('name')}")
    
    logger.info("=" * 50)
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test CCM template extraction")
    parser.add_argument("--transcript", "-t", required=True, help="Path to transcript JSON file")
    parser.add_argument("--output", "-o", default="./outputs", help="Output directory")
    parser.add_argument("--template", "-tp", default=None, help="Path to CCM template file")
    parser.add_argument("--standalone", "-s", action="store_true", help="Test standalone extractor")
    
    args = parser.parse_args()
    
    if args.standalone:
        test_standalone_ccm_extraction(args.transcript, args.output, args.template)
    else:
        test_ccm_extraction(args.transcript, args.output, args.template)
