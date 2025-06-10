import os
import io
import pandas as pd
import numpy as np
from google.cloud import speech
from pydub import AudioSegment
from typing import Optional, Union, Dict, Any
import warnings

from Speech2Transcript.utils import segments_to_df

class GoogleDiarizationPipeline:
    """
    Google Speech-to-Text based diarization pipeline.
    Uses Google Cloud Speech-to-Text API for both transcription and speaker diarization.
    """
    def __init__(
            self,
            language: str = "en-US",
            device: Optional[str] = None,  # Kept for compatibility but not used
    ):
        # Initialize Google Speech-to-Text client
        self.client = speech.SpeechClient()
        self.language = language

        # For compatibility with original implementation
        self.device = "cloud"  # Processing happens in the cloud

    def process_audio(
            self,
            audio: Union[str, np.ndarray],
            sample_rate: Optional[int] = None,
            num_speakers: Optional[int] = None,
            min_speakers: Optional[int] = 2,
            max_speakers: Optional[int] = 6
    ) -> pd.DataFrame:
        """
        Process audio for diarization using Google Speech-to-Text API.

        Args:
            audio: Path to audio file or numpy array
            sample_rate: Sample rate (required if audio is numpy array)
            num_speakers: Exact number of speakers (optional)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers

        Returns:
            DataFrame with diarization results
        """
        # Prepare audio content
        if isinstance(audio, str):
            # Load audio file
            with io.open(audio, "rb") as audio_file:
                content = audio_file.read()

            # Determine audio format from file extension
            audio_format = speech.RecognitionConfig.AudioEncoding.LINEAR16  # Default
            if audio.lower().endswith('.mp3'):
                audio_format = speech.RecognitionConfig.AudioEncoding.MP3
            elif audio.lower().endswith('.flac'):
                audio_format = speech.RecognitionConfig.AudioEncoding.FLAC
            elif audio.lower().endswith('.wav'):
                audio_format = speech.RecognitionConfig.AudioEncoding.LINEAR16

            # Get sample rate if not provided
            if sample_rate is None:
                try:
                    import soundfile as sf
                    info = sf.info(audio)
                    sample_rate = info.samplerate
                except (ImportError, RuntimeError):
                    try:
                        audio_segment = AudioSegment.from_file(audio)
                        sample_rate = audio_segment.frame_rate
                    except Exception:
                        sample_rate = 44100  # Default
        else:
            # Convert numpy array to bytes
            if sample_rate is None:
                raise ValueError("sample_rate is required when passing an audio array")

            # Convert to 16-bit PCM
            audio_array = (audio * 32767).astype(np.int16)
            content = audio_array.tobytes()
            audio_format = speech.RecognitionConfig.AudioEncoding.LINEAR16

        # Configure diarization
        diarization_config = speech.SpeakerDiarizationConfig(
            enable_speaker_diarization=True,
            min_speaker_count=min_speakers,
            max_speaker_count=max_speakers,
        )

        # Configure recognition request
        config = speech.RecognitionConfig(
            encoding=audio_format,
            sample_rate_hertz=sample_rate,
            language_code=self.language,
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
            use_enhanced=True,
            model="latest_long",
            diarization_config=diarization_config
        )

        # Create recognition audio
        audio_obj = speech.RecognitionAudio(content=content)

        # Perform recognition
        try:
            response = self.client.recognize(config=config, audio=audio_obj)
        except Exception as e:
            warnings.warn(f"Google Speech-to-Text API error: {e}")
            return pd.DataFrame(columns=['segment', 'label', 'speaker', 'start', 'end'])

        # Process results to extract speaker segments
        segments = []

        for i, result in enumerate(response.results):
            alternative = result.alternatives[0]

            if hasattr(alternative, 'words') and alternative.words:
                # Group words by speaker
                current_speaker = None
                current_segment = None

                for word_info in alternative.words:
                    speaker_tag = getattr(word_info, 'speaker_tag', 0)
                    speaker_id = f"SPEAKER_{speaker_tag}"

                    start_time = word_info.start_time.total_seconds()
                    end_time = word_info.end_time.total_seconds()

                    if current_speaker != speaker_id:
                        # Save previous segment if exists
                        if current_segment is not None:
                            segments.append(current_segment)

                        # Start new segment
                        current_segment = {
                            'segment': len(segments),
                            'label': 'SPEAKER',
                            'speaker': speaker_id,
                            'start': start_time,
                            'end': end_time
                        }
                        current_speaker = speaker_id
                    else:
                        # Extend current segment
                        current_segment['end'] = end_time

                # Add the last segment
                if current_segment is not None:
                    segments.append(current_segment)

        # If no segments found, return empty DataFrame
        if not segments:
            warnings.warn("No speech segments detected in audio")
            return pd.DataFrame(columns=['segment', 'label', 'speaker', 'start', 'end'])

        return pd.DataFrame(segments)

# For backward compatibility, keep the original class name
class DiarizationPipeline(GoogleDiarizationPipeline):
    """
    Backward-compatible wrapper for GoogleDiarizationPipeline.
    Maintains the same interface as the original pyannote-based implementation.
    """
    def __init__(
            self,
            model_name: str = "pyannote/speaker-diarization-3.1",  # Keep original param name for compatibility
            use_auth_token: Optional[str] = None,  # Keep original param for compatibility
            device: Optional[str] = None,
    ):
        # Map old parameters to new ones
        super().__init__(
            language="en-US",  # Default language
            device=device
        )
        # Store original parameters for reference
        self.original_model_name = model_name
        self.original_use_auth_token = use_auth_token


# Alias for backward compatibility
LightDiarizationPipeline = GoogleDiarizationPipeline
