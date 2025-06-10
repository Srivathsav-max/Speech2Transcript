import os 
import torch
import numpy as np
import pandas as pd
from pydub import AudioSegment
from google.cloud import speech
import io
from typing import Optional, Union, Dict, Any

class TranscriptionPipeline:
    def __init__(
            self,
            model_name: str = "default",  # Not used with Google STT but kept for compatibility
            device: Optional[Union[str, torch.device]] = None,  # Not used with Google STT but kept for compatibility
            compute_type: Optional[str] = None,  # Not used with Google STT but kept for compatibility
            chunk_length: Optional[int] = 60,  # Maximum length in seconds for each audio chunk
            batch_size: int = 1,  # Not used with Google STT but kept for compatibility
            language: Optional[str] = "en-US",  # Language code for Google STT
            beam_size: int = 1  # Not used with Google STT but kept for compatibility
    ):
        # Initialize Google Speech-to-Text client
        self.client = speech.SpeechClient()
        
        # Store configuration
        self.language = language
        self.chunk_length = chunk_length
        
        # For compatibility with original implementation
        self.device = "cpu"  # Processing happens in the cloud
        self.batch_size = batch_size
        self.beam_size = beam_size
    
    def transcribe_audio(
            self,
            audio: Union[str, np.ndarray], 
            sample_rate: Optional[int] = None,
            return_timestamps: bool = False,
            enable_speaker_diarization: bool = True,
            min_speaker_count: int = 2,
            max_speaker_count: int = 6
    ) -> Dict[str, Any]:
        """
        Transcribe audio using Google Speech-to-Text API.
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
                # Try to detect sample rate from the audio file
                try:
                    import soundfile as sf
                    info = sf.info(audio)
                    sample_rate = info.samplerate
                except (ImportError, RuntimeError):
                    try:
                        from pydub import AudioSegment
                        audio_segment = AudioSegment.from_file(audio)
                        sample_rate = audio_segment.frame_rate
                    except Exception:
                        # Use default sample rate as last resort
                        sample_rate = 44100
        else:
            # Convert numpy array to bytes
            if sample_rate is None:
                raise ValueError("sample_rate is required when passing an audio array")
            
            # Convert to 16-bit PCM
            audio_array = (audio * 32767).astype(np.int16)
            content = audio_array.tobytes()
            audio_format = speech.RecognitionConfig.AudioEncoding.LINEAR16
        
        # Configure recognition request
        diarization_config = None
        if enable_speaker_diarization:
            diarization_config = speech.SpeakerDiarizationConfig(
                enable_speaker_diarization=True,
                min_speaker_count=min_speaker_count,
                max_speaker_count=max_speaker_count,
            )
            
        config = speech.RecognitionConfig(
            encoding=audio_format,
            sample_rate_hertz=sample_rate,
            language_code=self.language,
            enable_word_time_offsets=return_timestamps,
            enable_automatic_punctuation=True,
            use_enhanced=True,
            model="latest_long",
            diarization_config=diarization_config
        )
        
        # Create recognition audio
        audio = speech.RecognitionAudio(content=content)
        
        # Perform recognition
        try:
            response = self.client.recognize(config=config, audio=audio)
        except Exception as e:
            return {
                "text": f"Error in Google STT API: {str(e)}",
                "chunks": [],
                "language": self.language,
                "language_probability": 0.0,
                "speaker_segments": []
            }
        
        # Process results
        result = {
            "text": "",
            "language": self.language,
            "language_probability": 1.0,
            "chunks": [],
            "speaker_segments": []
        }
        
        # Process results and extract speaker information
        for i, res in enumerate(response.results):
            alternative = res.alternatives[0]
            result["text"] += alternative.transcript + " "
            
            # Extract word timestamps and speaker tags
            if return_timestamps and hasattr(alternative, 'words') and alternative.words:
                for word_info in alternative.words:
                    speaker_tag = getattr(word_info, 'speaker_tag', 0)
                    word_data = {
                        "text": word_info.word,
                        "timestamp": [
                            word_info.start_time.total_seconds(),
                            word_info.end_time.total_seconds()
                        ],
                        "speaker": f"SPEAKER_{speaker_tag}" if enable_speaker_diarization else "SPEAKER_0"
                    }
                    result["chunks"].append(word_data)
                    
        # Create speaker segments from word chunks
        if enable_speaker_diarization and result["chunks"]:
            # Group words by speaker
            speaker_words = {}
            for chunk in result["chunks"]:
                speaker = chunk["speaker"]
                if speaker not in speaker_words:
                    speaker_words[speaker] = []
                speaker_words[speaker].append(chunk)
            
            # Create segments for each speaker
            for speaker, words in speaker_words.items():
                # Skip SPEAKER_0 if we have other speakers identified
                if speaker == "SPEAKER_0" and len(speaker_words) > 1:
                    continue
                    
                # Sort words by start time
                words.sort(key=lambda x: x["timestamp"][0])
                
                # Group consecutive words into segments
                segments = []
                current_segment = {
                    "speaker": speaker,
                    "start": words[0]["timestamp"][0],
                    "end": words[0]["timestamp"][1],
                    "text": words[0]["text"]
                }
                
                for i in range(1, len(words)):
                    word = words[i]
                    prev_word = words[i-1]
                    
                    # If words are close together (less than 1.5 seconds gap), add to current segment
                    if word["timestamp"][0] - prev_word["timestamp"][1] < 1.5:
                        current_segment["text"] += " " + word["text"]
                        current_segment["end"] = word["timestamp"][1]
                    else:
                        # Start a new segment
                        segments.append(current_segment)
                        current_segment = {
                            "speaker": speaker,
                            "start": word["timestamp"][0],
                            "end": word["timestamp"][1],
                            "text": word["text"]
                        }
                
                # Add the last segment
                segments.append(current_segment)
                
                # Add all segments to result
                result["speaker_segments"].extend(segments)
        
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
            min_segment_length: float = 1.0,
            temp_dir: str = "./temp",
            cleanup: bool = True
    ) -> pd.DataFrame:
        
        results_df = diarization_df.copy()

        results_df["transcription"] = ""
        results_df["word_timestamps"] = None
        
        batch_size = 5  
        for i in range(0, len(results_df), batch_size):
            batch_df = results_df.iloc[i:i+batch_size]
            
            for idx, row in batch_df.iterrows():
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
                        
                        results_df.at[idx, "word_timestamps"] = adjusted_timestamps
                except Exception as e:
                    results_df.at[idx, "transcription"] = f"Error: {str(e)[:100]}"
                
                if cleanup:
                    try:
                        os.remove(segment_path)
                    except:
                        pass
                    
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
        if cleanup:
            for file in os.listdir(temp_dir):
                if file.startswith(os.path.splitext(os.path.basename(audio_path))[0]):
                    try:
                        os.remove(os.path.join(temp_dir, file))
                    except:
                        pass
        
        return results_df
