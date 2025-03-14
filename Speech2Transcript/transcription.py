import os 
import torch
import numpy as np
import pandas as pd
from pydub import AudioSegment
import faster_whisper
from typing import Optional, Union, Dict, Any

class TranscriptionPipeline:
    def __init__(
            self,
            model_name: str = "large-v3",
            device: Optional[Union[str, torch.device]] = None,
            compute_type: Optional[str] = "float16",
            chunk_length: Optional[int] = 30,
            batch_size: int = 8,
            language: Optional[str] = "en",
            beam_size: int = 5
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if isinstance(device, torch.device):
            device_type = device.type
            device_index = device.index if device_type == "cuda" else 0
        else:
            device_type = device
            device_index = 0
        
        if device_type == "cuda":
            fw_device = "cuda"
        elif device_type == "mps":
            fw_device = "cpu"
            compute_type = "float32"
        else:
            fw_device = "cpu"
            compute_type = "float32"

        self.model = faster_whisper.WhisperModel(
            model_name,
            device = fw_device,
            device_index = device_index,
            compute_type = compute_type,
            download_root = None,
            local_files_only= False,
        )

        self.language = language
        self.device = device_type
        self.chunk_length = chunk_length
        self.batch_size = batch_size
        self.beam_size = beam_size
    
    def transcribe_audio(
            self,
            audio: Union[str, np.ndarray], 
            sample_rate: Optional[int] = None,
            return_timestamps: bool = False
    ) -> Dict[str, Any]:
        
        transcribe_options = {
            "beam_size": self.beam_size,
            "word_timestamps": return_timestamps,
            "language": self.language,
            "vad_filter" : True,
            "vad_parameters": {
                "min_silence_duration_ms": 500 # minimum slience duration means the minimum duration of silence in milliseconds in the given audio 500 represents 0.5 seconds so if the audio has silence of 0.5 seconds or more it will be removed
            }
        }

        if isinstance(audio, str):
            segments, info = self.model.transcribe(
                audio,
                **transcribe_options
            )
        else:
            if sample_rate is None:
                raise ValueError("sample_rate is required when passing an audio array")
            segments, info = self.model.transcribe(
                audio,
                **transcribe_options
            )
        
        result = {
            "text": "",
            "chunks": [],
            "language": self.language,
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
    
    def extract_segment(
            self,
            audio_path: str,
            start_time: float,
            end_time: float,
            temp_dir: str = "./temp"
    ) -> str:
        
        os.makedirs(temp_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        temp_audio_path = os.path.join(temp_dir, f"{base_name}_{start_time}_{end_time}_segment.wav")

        if os.path.exists(temp_audio_path):
            return temp_audio_path
        
        audio = AudioSegment.from_file(audio_path)
        segment = audio[int(start_time * 1000): int(end_time * 1000)]
        segment.export(temp_audio_path, format="wav")

        return temp_audio_path

    def process_diarization(
            self,
            diarization_df: pd.DataFrame,
            audio_path: str,
            min_segment_length: float = 0.5,
            temp_dir: str = "./temp",
            cleanup: bool = True
    ) -> pd.DataFrame:
        
        results_df = diarization_df.copy()

        results_df["transcription"] = ""
        results_df["word_timestamps"] = None

        for idx, row in results_df.iterrows():
            start_time = row["start"]
            end_time = row["end"]

            if end_time - start_time < min_segment_length:
                continue

            segment_path = self.extract_segment(audio_path, start_time, end_time, temp_dir)

            try:
                transcription = self.transcribe_audio(segment_path)

                results_df.at[idx, "transcription"] = transcription["text"]
                results_df.at[idx, "language"] = transcription["language"]
                results_df.at[idx, "language_probability"] = transcription["language_probability"]

                if "chunks" in transcription:

                    adjusted_timestamps = []
                    for chunk in transcription["chunks"]:
                        if isinstance(chunk, dict) and "timestamp" in chunk:
                            ts = chunk["timestamp"]
                            if isinstance(ts, list) and len(ts) == 2:
                                adjusted_timestamps.append({
                                    "word": chunk.get("text", ""),
                                    "start": ts[0] + start_time,
                                    "end": ts[1] + start_time
                                })
                    
                    results_df.at[idx, "word_timestamps"] = adjusted_timestamps # so if a segment has a sentence "Hello World" then the word_timestamps will be a list of dictionaries containing the start and end time of each word in the sentence
            except Exception as e:
                raise Exception(f"Error transcribing segment {start_time} - {end_time}: {e}")
    
        if cleanup:
            for file in os.listdir(temp_dir):
                if file.startswith(os.path.splitext(os.path.basename(audio_path))[0]):
                    os.remove(os.path.join(temp_dir, file))
        
        return results_df





