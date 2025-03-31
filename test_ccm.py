#!/usr/bin/env python3
"""
Test script for the CCM template extraction capabilities.

This script demonstrates the enhanced CCM template functionality
using the new dynamic content generation approach.
"""
import os
import sys
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CCM_TEST")

def main():
    """Run CCM template test"""
    parser = argparse.ArgumentParser(description="Test CCM template extraction")
    parser.add_argument("--transcript", "-t", required=True, help="Path to transcript JSON file")
    parser.add_argument("--output", "-o", default="./outputs", help="Path to save results")
    parser.add_argument("--telehealth", help="Path to telehealth template file")
    parser.add_argument("--ccm", help="Path to CCM template file")
    args = parser.parse_args()
    
    # Ensure the output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Build the command to run the main script
    cmd = [
        sys.executable,
        "-m", "Speech2Transcript.main",
        "--advanced_medical_summary",
        "--transcript_file", args.transcript,
        "--output", args.output
    ]
    
    # Add template paths if provided
    if args.telehealth:
        cmd.extend(["--telehealth_template", args.telehealth])
    
    if args.ccm:
        cmd.extend(["--ccm_template", args.ccm])
    
    # Display the command
    logger.info("Running command: %s", " ".join(cmd))
    
    # Run the command
    try:
        import subprocess
        subprocess.run(cmd, check=True)
        logger.info("CCM template extraction completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error("CCM template extraction failed with error: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()