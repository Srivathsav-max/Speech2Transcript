import os
import torch
import logging as log
import argparse
from diarization import DiarizationPipeline 
from utils import export_to_rttm, export_to_json, merge_speakers, get_device, audio_devices
from transcription import TranscriptionPipeline

log.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=log.INFO
)

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

    format_group = parser.add_argument_group('Real Time Audio')
    format_group.add_argument("--list_devices", action="store_true", help="List available audio devices")

    args = parser.parse_args()

    if args.audio:
        if not os.path.exists(args.audio):
            log.error("Audio File Not Found: %s", args.audio)
            return

        os.makedirs(args.output, exist_ok=True)

        basename = os.path.splitext(os.path.basename(args.audio))[0]

        log.info("Initializing Diarization Pipeline")
        log.info("Using Device: %s", args.device)

    if args.list_devices:
        devices = audio_devices()
        for device in devices:
            log.info("Device Index: %s, Name: %s, Max Channels: %s", device["index"], device["name"], device["channels"])
        return
    
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
        
        log.info("Processing Audio for Transcription: %s", args.audio)
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

    formats = [args.format] if args.format != "all" else ["csv", "rttm", "json"]

    for fmt in formats:
        if fmt == "csv":
            # For CSV, we need to handle the segment object
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


