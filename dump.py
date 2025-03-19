import os
import torch
import logging as log
import argparse
import signal
import sys
from diarization import DiarizationPipeline 
from utils import export_to_rttm, export_to_json, merge_speakers, get_device, audio_devices
from transcription import TranscriptionPipeline
from realtime_stt import EnhancedRealtimePipeline

log.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=log.INFO
)

# Global variable to store the realtime pipeline instance
realtime_instance = None

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully stop the realtime pipeline"""
    global realtime_instance
    if realtime_instance is not None:
        log.info("Stopping recording...")
        realtime_instance.stop()
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(
        description = "Speech to Text Transcription",
        formatter_class= argparse.ArgumentDefaultsHelpFormatter 
    )

    parser.add_argument("--audio", "-a", type=str, help="Path to the audio file")
    parser.add_argument("--output", "-o", default="./outputs", help="Path to the output directory")
    parser.add_argument("--use_auth_token", "-hf_token", default=None, help="Hugging Face authentication token")
    parser.add_argument("--device", "-d", default=get_device(), choices=["cuda", "mps", "cpu"], help="Device to use")

    diar_group = parser.add_argument_group("Diarization")
    diar_group.add_argument("--diarize", action="store_true", help="Perform Diarization")
    diar_group.add_argument("--diarization_model", "-dm", default="pyannote/speaker-diarization-3.1", help="Model name")
    diar_group.add_argument("--num_speakers", "-ns", type=int, default=None, help="Number of speakers")
    diar_group.add_argument("--min_speakers", "-min_ns", type=int, default=None, help="Minimum number of speakers")
    diar_group.add_argument("--max_speakers", "-max_ns", type=int, default=None, help="Maximum number of speakers")
    diar_group.add_argument("--merge_threshold", type=float, default=0.5, help="Threshold to merge consecutive segments from same speaker (in seconds)")

    trans_group = parser.add_argument_group('Transcription')
    trans_group.add_argument("--transcribe", action="store_true", help="Perform transcription")
    trans_group.add_argument("--transcription_model", "-tm", default="large-v3", help="Transcription model name (large-v3, medium, small, tiny)")
    trans_group.add_argument("--language", "-l", default=None, help="Language code for transcription (e.g., 'en', 'fr')")
    trans_group.add_argument("--compute_type", "-tcompute",default="float16", choices=["float16", "float32", "int8"], help="Compute type for transcription model")
    trans_group.add_argument("--chunk_length", type=int, default=30, help="Chunk length for transcription (in seconds)")
    trans_group.add_argument("--batch_size", type=int, default=8, help="Batch size for transcription")
    trans_group.add_argument("--beam_size", type=int, default=5, help="Beam size for transcription")
    trans_group.add_argument("--min_segment_length", type=float, default=0.5, help="Minimum segment length to transcribe (in seconds)")

    format_group = parser.add_argument_group('Output Formats')
    format_group.add_argument("--format", "-f", choices=["rttm", "json", "csv", "all"], default="all", help="Output format(s)")

    audio_group = parser.add_argument_group('Audio Devices')
    audio_group.add_argument("--list_devices", action="store_true", help="List available audio devices")
    
    # New realtime mode arguments
    realtime_group = parser.add_argument_group('Real-time Processing')
    realtime_group.add_argument("--realtime", action="store_true", help="Enable real-time audio processing")
    realtime_group.add_argument("--device_index", type=int, default=None, help="Input device index for real-time audio")
    realtime_group.add_argument("--buffer_duration", type=float, default=15.0, help="Audio buffer duration in seconds")
    realtime_group.add_argument("--processing_interval", type=float, default=2.0, help="How often to process the buffer in seconds")
    realtime_group.add_argument("--vad_threshold", type=float, default=0.02, help="Voice activity detection threshold")
    realtime_group.add_argument("--use_adaptive_vad", action="store_true", help="Use adaptive VAD threshold")
    realtime_group.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")

    args = parser.parse_args()

    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Handle device listing request first
    if args.list_devices:
        devices = audio_devices()
        log.info("Available Audio Input Devices:")
        for device in devices:
            log.info(f"Device Index: {device['index']}, Name: {device['name']}, Max Channels: {device['channels']}")
        return

    # Set up output directory
    os.makedirs(args.output, exist_ok=True)

    # Initialize pipelines
    if args.realtime or args.diarize or args.transcribe:
        # Initialize the diarization pipeline
        diarization = DiarizationPipeline(
            model_name=args.diarization_model,
            use_auth_token=args.use_auth_token,
            device=args.device
        )
        log.info(f"Diarization pipeline initialized, using device: {args.device}")
        
        # Initialize the transcription pipeline if needed
        if args.realtime or args.transcribe:
            transcription = TranscriptionPipeline(
                model_name=args.transcription_model,
                device=args.device,
                compute_type=args.compute_type,
                chunk_length=args.chunk_length,
                batch_size=args.batch_size,
                language=args.language,
                beam_size=args.beam_size
            )
            log.info(f"Transcription pipeline initialized with model: {args.transcription_model}")
    
    # Handle real-time mode
    if args.realtime:
        global realtime_instance
        
        log.info("Starting real-time audio processing mode")
        
        # Check if we need to recommend listing devices
        if args.device_index is None:
            devices = audio_devices()
            if len(devices) > 1:
                log.info("Multiple audio input devices detected. You may want to specify one with --device_index")
                log.info("Available devices:")
                for device in devices:
                    log.info(f"  Index: {device['index']}, Name: {device['name']}")
        
        # Initialize the real-time pipeline
        realtime_instance = EnhancedRealtimePipeline(
            diarization_pipeline=diarization,
            transcription_pipeline=transcription,
            sample_rate=args.sample_rate,
            buffer_duration=args.buffer_duration,
            processing_interval=args.processing_interval,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            merge_threshold=args.merge_threshold,
            device_index=args.device_index,
            output_dir=args.output,
            vad_threshold=args.vad_threshold,
            use_adaptive_vad=args.use_adaptive_vad
        )
        
        try:
            # Start the pipeline
            realtime_instance.start()
            
            # Keep the main thread alive while the pipeline runs
            # The user can stop by pressing Ctrl+C, which will be caught by our signal handler
            while realtime_instance.is_running:
                import time
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            log.info("Keyboard interrupt detected, stopping...")
            realtime_instance.stop()
        except Exception as e:
            log.error(f"Error in real-time processing: {e}")
            if realtime_instance is not None:
                realtime_instance.stop()
        
        # Display summary when stopped
        log.info("Real-time processing complete")
        speaker_stats = realtime_instance.get_speaker_statistics()
        log.info(f"Detected {len(speaker_stats)} speakers")
        for speaker, stats in speaker_stats.items():
            log.info(f"  {speaker}: {stats['total_speech_duration']:.1f} seconds of speech")
        
        log.info(f"Results saved to: {args.output}")
        return
        
    # Handle file-based processing mode
    if args.audio:
        if not os.path.exists(args.audio):
            log.error(f"Audio file not found: {args.audio}")
            return
            
        basename = os.path.splitext(os.path.basename(args.audio))[0]
        
        diarization_results = None
        
        if args.diarize:
            log.info(f"Processing audio file: {args.audio}")
            
            diarization_results = diarization.process_audio(
                args.audio,
                num_speakers=args.num_speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers
            )
            
            num_speakers = diarization_results["speaker"].nunique()
            total_duration = diarization_results["end"].max()
            log.info(f"Detected {num_speakers} speakers in total duration {total_duration:.2f} seconds")
            
            if args.merge_threshold > 0:
                diarization_results = merge_speakers(diarization_results, threshold=args.merge_threshold)
        
        if args.transcribe:
            if diarization_results is None:
                log.error("Transcription requires diarization results. Please enable diarization.")
                return
            
            log.info("Transcribing audio segments...")
            temp_dir = os.path.join(args.output, "temp")
            
            results = transcription.process_diarization(
                diarization_results,
                args.audio,
                min_segment_length=args.min_segment_length,
                temp_dir=temp_dir
            )
            
            # Replace diarization results with transcribed results
            diarization_results = results
            
            log.info("Transcription completed")
            
        # Save results in requested formats
        if diarization_results is not None:
            formats = [args.format] if args.format != "all" else ["csv", "rttm", "json"]
            
            for fmt in formats:
                if fmt == "csv":
                    output_path = os.path.join(args.output, f"{basename}_diarization.csv")
                    results_copy = diarization_results.copy()
                    if 'segment' in results_copy.columns:
                        results_copy['segment_str'] = results_copy['segment'].apply(str)
                        results_copy.drop('segment', axis=1).to_csv(output_path, index=False)
                    else:
                        results_copy.to_csv(output_path, index=False)
                    log.info(f"Saved CSV results to: {output_path}")
                    
                elif fmt == "rttm":
                    output_path = os.path.join(args.output, f"{basename}_diarization.rttm")
                    export_to_rttm(diarization_results, output_path, args.audio)
                    log.info(f"Saved RTTM results to: {output_path}")
                    
                elif fmt == "json":
                    output_path = os.path.join(args.output, f"{basename}_diarization.json")
                    export_to_json(diarization_results, output_path)
                    log.info(f"Saved JSON results to: {output_path}")
        
        log.info("Processing finished successfully")

if __name__ == "__main__":
    main()