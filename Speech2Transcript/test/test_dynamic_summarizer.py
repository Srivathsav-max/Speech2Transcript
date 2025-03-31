"""
Test script for the enhanced dynamic medical transcript summarizer.

This script demonstrates the new dynamic summarization capabilities.
"""
import os
import sys
import json
import logging

# Add parent directory to path to allow importing Speech2Transcript
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the processor
from summarizers.medical_transcript_processor import MedicalTranscriptProcessor
from summarizers.medical_pipeline_integration import MedicalTranscriptSummarizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dynamic_summarizer_test')

def test_dynamic_summarizer(transcript_path, output_dir):
    """
    Process a transcript file using the dynamic summarizer.
    
    Args:
        transcript_path: Path to the transcript JSON file
        output_dir: Directory to save outputs
    """
    logger.info(f"Processing transcript with dynamic summarizer: {transcript_path}")
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the processor without NER model to avoid dependency issues
    processor = MedicalTranscriptProcessor(
        ner_model=None,  # Skip NER model to avoid dependency issues
        device="cpu",
        logger=logger
    )
    
    # Process the transcript
    basename = os.path.splitext(os.path.basename(transcript_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_dynamic_summary.json")
    
    results = processor.process_transcript(
        transcript_path=transcript_path,
        output_path=output_path
    )
    
    # Print summary of results
    logger.info("=" * 60)
    logger.info("Dynamic Summarization Results:")
    logger.info("=" * 60)
    
    # Print the narrative summary
    logger.info("\nNarrative Summary:")
    logger.info(results["narrative_summary"])
    
    # Print transcript analysis
    logger.info("\nTranscript Analysis:")
    transcript_analysis = results.get("transcript_analysis", {})
    
    logger.info(f"Conversation Type: {transcript_analysis.get('conversation_type', 'N/A')}")
    logger.info(f"Sentiment: {transcript_analysis.get('sentiment', 'N/A')}")
    
    if transcript_analysis.get("key_topics"):
        logger.info(f"Key Topics: {', '.join(transcript_analysis['key_topics'])}")
    
    if transcript_analysis.get("patient_concerns"):
        logger.info(f"Patient Concerns: {', '.join(transcript_analysis['patient_concerns'])}")
    
    if transcript_analysis.get("provider_recommendations"):
        logger.info(f"Provider Recommendations: {', '.join(transcript_analysis['provider_recommendations'])}")
    
    # Print SOAP note
    logger.info("\nSOAP Note:")
    for section, content in results["soap_note"].items():
        logger.info(f"\n{section}:")
        logger.info(content)
    
    logger.info("\nOutputs saved to:")
    logger.info(f"- JSON Summary: {output_path}")
    logger.info(f"- Text Summary: {output_path.replace('.json', '_summary.txt')}")

def test_pipeline_integration(transcript_path, output_dir):
    """
    Test the MedicalTranscriptSummarizer integration class.
    
    Args:
        transcript_path: Path to the transcript JSON file
        output_dir: Directory to save outputs
    """
    logger.info("\n" + "=" * 60)
    logger.info("Testing Pipeline Integration:")
    logger.info("=" * 60)
    
    # Initialize the summarizer without NER model to avoid dependency issues
    summarizer = MedicalTranscriptSummarizer(
        ner_model=None,  # Skip NER model to avoid dependency issues
        device="cpu",
        logger=logger
    )
    
    # Process the transcript
    basename = os.path.splitext(os.path.basename(transcript_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_pipeline_summary.json")
    
    results = summarizer.process_transcript(
        transcript_path=transcript_path,
        output_path=output_path
    )
    
    # Print the narrative summary
    logger.info("\nPipeline Integration Narrative Summary:")
    logger.info(results["narrative_summary"])
    
    logger.info(f"\nPipeline integration outputs saved to: {output_path}")

if __name__ == "__main__":
    # Default paths
    transcript_path = os.path.join(
        os.path.dirname(__file__), 
        "sample_transcript.json"
    )
    
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "outputs"
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the transcript with dynamic summarizer
    test_dynamic_summarizer(transcript_path, output_dir)
    
    # Test pipeline integration
    test_pipeline_integration(transcript_path, output_dir)