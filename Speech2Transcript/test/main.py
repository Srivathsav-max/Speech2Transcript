# Add to main.py
from Speech2Transcript.summarizers.medical_pipeline_integration import MedicalTranscriptSummarizer

# Add a new argument group for advanced medical summarization
advanced_med_group = parser.add_argument_group("Advanced Medical Summarization")
advanced_med_group.add_argument("--advanced_medical_summary", action="store_true", 
                            help="Enable advanced medical transcript summarization")
advanced_med_group.add_argument("--med_base_model", 
                            default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", 
                            help="Base language model for medical summarization")
advanced_med_group.add_argument("--med_ner_model", 
                            default="emilyalsentzer/Bio_ClinicalBERT", 
                            help="NER model for medical entity recognition")
advanced_med_group.add_argument("--med_qa_model", 
                            default="dmis-lab/biobert-base-cased-v1.1-squad", 
                            help="QA model for medical information extraction")
advanced_med_group.add_argument("--med_confidence", type=float, default=0.65, 
                            help="Confidence threshold for medical entity extraction")
advanced_med_group.add_argument("--med_cache_dir", default=None, 
                            help="Cache directory for medical models")

# Then in your main function, add:
if args.advanced_medical_summary:
    if args.transcript_file is None and not os.path.exists(os.path.join(args.output, f"{basename}_diarization.json")):
        log.error("No transcript file available for advanced medical summarization")
    else:
        transcript_file = args.transcript_file or os.path.join(args.output, f"{basename}_diarization.json")
        
        try:
            log.info("Initializing Advanced Medical Transcript Summarizer")
            medical_summarizer = MedicalTranscriptSummarizer(
                base_model=args.med_base_model,
                ner_model=args.med_ner_model,
                qa_model=args.med_qa_model,
                device=args.device,
                compute_type=args.compute_type,
                cache_dir=args.med_cache_dir,
                confidence_threshold=args.med_confidence,
                logger=log
            )
            
            output_path = os.path.join(args.output, f"{basename}_medical_summary.json")
            
            log.info(f"Processing medical transcript: {transcript_file}")
            result = medical_summarizer.process_transcript(
                transcript_path=transcript_file,
                output_path=output_path,
                text_column="transcription",
                speaker_column="speaker"
            )
            
            log.info("=" * 60)
            log.info("Medical Summary:")
            log.info("=" * 60)
            log.info(result["narrative_summary"])
            
            log.info("\n" + "=" * 60)
            log.info("SOAP Note:")
            log.info("=" * 60)
            for section, content in result["soap_note"].items():
                log.info(f"{section}:")
                log.info(content)
                log.info("-" * 40)
            
            log.info(f"Full medical summary saved to: {output_path}")
            log.info(f"Simple summary saved to: {output_path.replace('.json', '_simple.json')}")
            log.info(f"Text summary saved to: {output_path.replace('.json', '_summary.txt')}")
            
        except Exception as e:
            log.error(f"Error in advanced medical summarization: {e}")
            import traceback
            traceback.print_exc()