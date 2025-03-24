import os
import pandas as pd
import torch
import pyaudio 
import json
from typing import List, Dict

from summarize import SummarizationPipeline
from typing import Optional, Union, List, Dict, Any


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def segments_to_df(segments: List, labels: List, speakers: List) -> pd.DataFrame:

    df = pd.DataFrame({
        "segment": segments,
        "label": labels,
        "speaker": speakers
    })

    df["start"] = df["segment"].apply(lambda x: x.start)
    df["end"] = df["segment"].apply(lambda x: x.end)
    
    return df

def export_to_rttm(diarization_df: pd.DataFrame, output_path: str, audio_file: str):
    with open(output_path, "w") as f:
        for _ , row in diarization_df.iterrows():
            duration = row["end"] - row["start"]
            f.write(
                f"SPEAKER {audio_file} {row['start']:.3f} {row['end']:.3f} {duration:.3f} {row['speaker']} <NA> <NA> {row['label']} <NA> <NA>\n"
            )

def export_to_json(diarization_df: pd.DataFrame, output_path: str):
    result = {
        "speakers": list(diarization_df["speaker"].unique()),
        "segments": []
    }

    for _, row in diarization_df.iterrows():
        segment_data = {
            "start": row["start"],
            "end": row["end"],
            "speaker": row["speaker"],
            "label": row["label"]
        }

        if "transcription" in row and row["transcription"]:
            segment_data["transcription"] = row["transcription"]
        
        if "word_timestamps" in row and row["word_timestamps"]:
            segment_data["word_timestamps"] = row["word_timestamps"]
        
        if "language" in row and row["language"]:
            segment_data["language"] = row["language"]
        
        if "language_probability" in row and row["language_probability"]:
            segment_data["language_probability"] = row["language_probability"]
        
        result["segments"].append(segment_data)
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

def merge_speakers(diarization_df: pd.DataFrame, merge_threshold: float = 0.5) -> pd.DataFrame:
    
    df = diarization_df.sort_values("start").reset_index(drop=True)

    merged_segments = []

    current_speaker = None
    current_start = None 
    current_end = None
    current_label = None
    current_segment = None
    current_transcription = "" 
    current_word_timestamps = []
    current_language = None
    current_language_probability = None

    for idx, row in df.iterrows():
        if current_speaker is None:
            current_speaker = row["speaker"]
            current_start = row["start"]
            current_end = row["end"]
            current_label = row["label"]
            current_segment = row["segment"]
            current_transcription = row.get("transcription", "")
            current_word_timestamps = row.get("word_timestamps", [])
            current_language = row["language"]
            current_language_probability = row["language_probability"]
        
        elif (row["speaker"] == current_speaker and 
              row["start"] - current_end <= merge_threshold):
            current_end = row["end"]
            if "transcription" in row and row["transcription"]:
                if current_transcription:
                    current_transcription += " " + row["transcription"]
                else:
                    current_transcription = row["transcription"]
            if "word_timestamps" in row and row["word_timestamps"]:
                current_word_timestamps.extend(row["word_timestamps"])
            if "language" in row and row["language"]:
                current_language = row["language"]
            if "language_probability" in row and row["language_probability"]:
                current_language_probability = row["language_probability"]

        else:
            # Add current segment to merged segments
            merged_segments.append({
                "segment": current_segment,
                "label": current_label,
                "speaker": current_speaker,
                "start": current_start,
                "end": current_end,
                "transcription": current_transcription,
                "word_timestamps": current_word_timestamps if current_word_timestamps else None,
                "language": current_language,
                "language_probability": current_language_probability
            })
            
            # Start a new segment
            current_speaker = row["speaker"]
            current_start = row["start"]
            current_end = row["end"]
            current_label = row["label"]
            current_segment = row["segment"]
            current_transcription = row.get("transcription", "")
            current_word_timestamps = row.get("word_timestamps", [])
            current_language = row.get("language", None)
            current_language_probability = row.get("language_probability", None)
    
    # Add the last segment
    if current_speaker is not None:
        merged_segments.append({
            "segment": current_segment,
            "label": current_label,
            "speaker": current_speaker,
            "start": current_start,
            "end": current_end,
            "transcription": current_transcription,
            "word_timestamps": current_word_timestamps if current_word_timestamps else None,
            "language": current_language,
            "language_probability": current_language_probability
        })
    
    return pd.DataFrame(merged_segments)


def audio_devices() -> List[Dict]:

    pa = pyaudio.PyAudio()

    devices = []

    device_count = pa.get_device_count()
    try: 
        for i in range(device_count):
            device_info = pa.get_device_info_by_index(i)
            devices.append({
                "index": i,
                "name": device_info["name"],
                "device_type": device_info["hostApi"],
                "channels": device_info["maxInputChannels"],
                "sample_rate": device_info["defaultSampleRate"]
            })
    finally:
        pa.terminate()
    
    return devices