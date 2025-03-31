import os
import sys
import signal
import logging as log
import argparse
from diarization import DiarizationPipeline
from transcription import TranscriptionPipeline
from realtimestt import RealTimeSTT

from summarizers import AdvancedMedicalSummarizer, GeminiSummarizer

from utils import export_to_rttm, export_to_json, merge_speakers, get_device, audio_devices

log.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=log.INFO
)

realtime_instance = None

def signal_handler(sig, frame):
    global realtime_instance
    if realtime_instance is not None:
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
    parser.add_argument("--transcript_file", "-tf", type=str, help="Path to the Transcript file")

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
    trans_group.add_argument("--language", "-l", default="en", help="Language code for transcription (e.g., 'en', 'fr')")
    trans_group.add_argument("--compute_type", "-tcompute",default="float16", choices=["float16", "float32", "int8"], help="Compute type for transcription model")
    trans_group.add_argument("--chunk_length", type=int, default=30, help="Chunk length for transcription (in seconds)")
    trans_group.add_argument("--batch_size", type=int, default=8, help="Batch size for transcription")
    trans_group.add_argument("--beam_size", type=int, default=5, help="Beam size for transcription")
    trans_group.add_argument("--min_segment_length", type=float, default=0.5, help="Minimum segment length to transcribe (in seconds)")

    format_group = parser.add_argument_group('Output Formats')
    format_group.add_argument("--format", "-f", choices=["rttm", "json", "csv", "all"], default="all", help="Output format(s)")

    realtime_group = parser.add_argument_group('Real Time Audio')
    realtime_group.add_argument("--list_devices", action="store_true", help="List available audio devices")
    realtime_group.add_argument("--realtime", "-rt", action="store_true", help="Enable real-time audio processing")
    realtime_group.add_argument("--device_index", "-di", type=int, default=None, help="Input device index for real-time audio")
    realtime_group.add_argument("--buffer_duration", type=float, default=15.0, help="Audio buffer duration in seconds")
    realtime_group.add_argument("--processing_interval", type=float, default=2.0, help="How often to process the buffer in seconds")
    realtime_group.add_argument("--vad_threshold", type=float, default=0.005, help="Voice activity detection threshold")
    realtime_group.add_argument("--use_adaptive_vad", action="store_true", help="Use adaptive VAD threshold")
    realtime_group.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    realtime_group.add_argument("--chunk_size", type=int, default=2048, help="Audio chunk size for realtime processing")

    # Summarizer options
    summarizer_group = parser.add_argument_group("Transcript Summarization")
    summarizer_group.add_argument("--summarize", action="store_true", help="Generate a summary of the transcript")
    summarizer_group.add_argument("--detailed", action="store_true", help="Generate a more detailed summary with comprehensive analysis")
    summarizer_group.add_argument("--summary-output", default="summary", help="Output filename for the summary (without extension)")
    summarizer_group.add_argument("--use-gemini", action="store_true", help="Use Google Gemini API for generating summaries")
    summarizer_group.add_argument("--gemini-api-key", type=str, help="API key for Google Gemini (defaults to GOOGLE_API_KEY environment variable)")
    summarizer_group.add_argument("--gemini-model", default="gemini-pro", help="Gemini model to use for summarization")

    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    if os.path.exists(args.transcript_file) and args.transcript_file is not None:
        basename = os.path.splitext(os.path.basename(args.transcript_file))[0]
    elif args.audio:
        basename = os.path.splitext(os.path.basename(args.audio))[0]
    else:
        basename = "output"

    if args.summarize or args.detailed or args.use_gemini:
        if args.transcript_file is None and not os.path.exists(os.path.join(args.output, f"{basename}_diarization.json")):
            log.error("No transcript file available for summarization")
        else:
            transcript_file = args.transcript_file or os.path.join(args.output, f"{basename}_diarization.json")

            # Determine output filename
            summary_basename = args.summary_output if args.summary_output != "summary" else basename

            try:
                # Choose appropriate summarizer based on flags
                if args.use_gemini:
                    log.info("Initializing Gemini-powered summarizer")
                    summarizer = GeminiSummarizer(
                        api_key=args.gemini_api_key,
                        model_name=args.gemini_model,
                        logger=log
                    )
                    output_path = os.path.join(args.output, f"{summary_basename}_gemini_summary.json")
                    
                    log.info(f"Processing transcript with Gemini: {transcript_file}")
                    result = summarizer.process_transcript(
                        transcript_path=transcript_file,
                        output_path=output_path,
                        text_column="transcription",
                        speaker_column="speaker"
                    )
                    
                    log.info("=" * 60)
                    log.info("Gemini-Generated Summary:")
                    log.info("=" * 60)
                    log.info(result["summary"])
                    log.info("=" * 60)
                    
                    log.info(f"Summary saved to: {output_path}")
                    log.info(f"Text summary saved to: {output_path.replace('.json', '_summary.txt')}")
                    
                else:
                    # Use the AdvancedMedicalSummarizer for traditional extraction-based summaries
                    log.info("Initializing transcript summarizer")
                    summarizer = AdvancedMedicalSummarizer(logger=log)
                    output_path = os.path.join(args.output, f"{summary_basename}_enhanced_summary.json")

                    log.info(f"Processing transcript with comprehensive analysis: {transcript_file}")
                    result = summarizer.process_transcript(
                        transcript_path=transcript_file,
                        output_path=output_path,
                        text_column="transcription",
                        speaker_column="speaker"
                    )

                    log.info("=" * 60)
                    log.info("Enhanced Transcript Summary:")
                    log.info("=" * 60)
                    log.info(result["detailed_summary"])
                    log.info("=" * 60)

                    # Print health assessment if available
                    if "health_assessment" in result["extracted_data"]:
                        assessment = result["extracted_data"]["health_assessment"]
                        log.info("Health Status: %s (confidence: %.2f)",
                                assessment.get("status", "Unknown").upper(),
                                assessment.get("confidence", 0.0))

                    log.info(f"Comprehensive analysis saved to: {output_path}")
                    log.info(f"Text summary saved to: {output_path.replace('.json', '_summary.txt')}")

            except Exception as e:
                log.error(f"Error in transcript summarization: {e}")
                import traceback
                traceback.print_exc()


    if args.list_devices:
        devices = audio_devices()
        for device in devices:
            log.info("Device Index: %s, Name: %s, Max Channels: %s", device["index"], device["name"], device["channels"])
        return

    if args.realtime or args.diarize or args.transcribe:
        diarization = DiarizationPipeline(
            model_name=args.diarization_model,
            use_auth_token=args.use_auth_token,
            device=args.device
        )
        log.info(f"Diarization pipeline initialized, using device: {args.device}")

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

            try:
                original_transcribe = transcription.transcribe_audio

                def patched_transcribe_audio(self, audio, sample_rate=None, return_timestamps=False):
                    """Patched version to ensure VAD filter is disabled"""
                    if hasattr(self.model, 'transcribe'):
                        try:
                            segments, info = self.model.transcribe(
                                audio,
                                language=self.language,
                                beam_size=self.beam_size,
                                word_timestamps=return_timestamps,
                                vad_filter=False
                            )

                            result = {
                                "text": "",
                                "chunks": [],
                                "language": info.language,
                                "language_probability": info.language_probability
                            }
                            for segment in segments:
                                result["text"] += segment.text + " "
                                if return_timestamps and segment.words:
                                    for word in segment.words:
                                        result["chunks"].append({
                                            "text": word.word,
                                            "timestamp": [word.start, word.end]
                                        })

                            result["text"] = result["text"].strip()
                            return result

                        except Exception as e:
                            log.error(f"Direct transcription failed: {e}")
                            return original_transcribe(audio, sample_rate, return_timestamps)
                    else:
                        return original_transcribe(audio, sample_rate, return_timestamps)

                transcription.transcribe_audio = patched_transcribe_audio.__get__(transcription, type(transcription))
                log.info("Patched transcription pipeline to disable VAD filtering")

            except Exception as e:
                log.warning(f"Could not patch transcription pipeline: {e}")

    if args.realtime:
        global realtime_instance

        log.info("Starting real-time audio processing mode")

        if args.device_index is None:
            devices = audio_devices()
            if len(devices) > 1:
                log.info("Multiple audio input devices detected. You may want to specify one with --device_index")
                log.info("Available devices:")
                for device in devices:
                    log.info(f"  Index: {device['index']}, Name: {device['name']}")

        realtime_instance = RealTimeSTT(
            diarization_pipeline=diarization,
            transcription_pipeline=transcription,
            sample_rate=args.sample_rate,
            chunk_size=args.chunk_size,
            vad_threshold=args.vad_threshold,
            language=args.language,
            device_index=args.device_index,
            output_dir=args.output
        )

        try:
            realtime_instance.start()
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

        log.info("Real-time processing complete")

        log.info(f"Results saved to: {args.output}")
        return

    if args.audio:
        if not os.path.exists(args.audio):
            log.error("Audio File Not Found: %s", args.audio)
            return

        os.makedirs(args.output, exist_ok=True)

        log.info("Initializing Diarization Pipeline")
        log.info("Using Device: %s", args.device)

        if args.diarize:
            diarization = DiarizationPipeline(
                model_name=args.diarization_model,
                use_auth_token=args.use_auth_token,
                device=args.device
            )

            log.info("Processing Audio: %s", args.audio)

            diarization_results = diarization.process_audio(
                args.audio,
                num_speakers=args.num_speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers
            )

            # if args.merge_threshold:
            #     diarization_results = merge_speakers(diarization_results, threshold=args.merge_threshold)

            num_speakers = diarization_results["speaker"].nunique()
            total_duration = diarization_results["end"].max()
            log.info("Detected %s speakers in Total Duration %.2f seconds", num_speakers if num_speakers else "Failed In Prediction", total_duration)

        if args.transcribe:
            if diarization_results is None:
                log.error("Transcription requires diarization results. Please enable diarization.")
                return

            log.info("Initializing Transcription Pipeline")
            transcription = TranscriptionPipeline(
                model_name=args.transcription_model,
                device=args.device,
                compute_type=args.compute_type,
                chunk_length=args.chunk_length,
                batch_size=args.batch_size,
                language=args.language,
                beam_size=args.beam_size
            )

            log.info("Processing Audio for Transcription using: %s", args.transcription_model)
            temp_dir = os.path.join(args.output, "temp")

            results = transcription.process_diarization(
                diarization_results,
                args.audio,
                min_segment_length=args.min_segment_length,
                temp_dir=temp_dir
            )

            diarization_results = results

            log.info("Transcription completed")

        formats = [args.format] if args.format != "all" else ["csv", "rttm", "json"]

        for fmt in formats:
            if fmt == "csv":
                output_path = os.path.join(args.output, f"{basename}_diarization.csv")
                results_copy = diarization_results.copy()
                results_copy['segment_str'] = results_copy['segment'].apply(str)
                results_copy.drop('segment', axis=1).to_csv(output_path, index=False)
                log.info(f"Saved CSV results to: {output_path}")

            elif fmt == "rttm":
                output_path = os.path.join(args.output, f"{basename}_diarization.rttm")
                export_to_rttm(diarization_results, output_path, args.audio)
                log.info(f"Saved RTTM results to: {output_path}")

            elif fmt == "json":
                output_path = os.path.join(args.output, f"{basename}_diarization.json")
                export_to_json(diarization_results, output_path)
                log.info(f"Saved JSON results to: {output_path}")

        log.info(f"Processing Finished Successfully")

if __name__ == "__main__":
    main()