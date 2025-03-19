import os
import torch 
import time
import numpy as np 
import pandas as pd
import pyaudio
import threading
import wave
import queue
import json
from typing import Dict, List, Any
from collections import deque

class RealTimeSTT:
    def __init__(
            self,
            diarization_pipeline,
            transcription_pipeline,
            sample_rate=16000,
            chunk_size=2048,
            vad_threshold=0.005,
            language="en",
            device_index=None,
            output_dir="./outputs"
    ):
        # Core components
        self.diarization_pipeline = diarization_pipeline
        self.transcription_pipeline = transcription_pipeline
        
        # Audio settings
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.vad_threshold = vad_threshold
        self.device_index = device_index
        self.language = language
        self.output_dir = output_dir
        
        # Audio capture state
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.is_running = False
        
        # Audio processing queues and buffers
        self.audio_queue = queue.Queue()
        self.speech_segments_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Processing state
        self.active_speech_buffer = None
        self.speech_active = False
        self.silence_frames = 0
        self.max_silence_frames = int(1.0 * sample_rate / chunk_size)  # 1 second of silence
        self.speech_frame_count = 0
        self.speech_min_frames = int(0.3 * sample_rate / chunk_size)   # 300ms minimum speech
        
        # Threads
        self.capture_thread = None
        self.vad_thread = None
        self.diarization_thread = None
        self.transcription_thread = None
        self.display_thread = None
        
        # Stats and counters
        self.speaker_counter = 1
        self.segments = []
        self.diarization_failures = 0
        self.transcription_failures = 0
        self.successful_segments = 0
        
        # Make output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "temp"), exist_ok=True)
        
        print(f"Initialized Robust Speech Processor:")
        print(f"  - Device index: {self.device_index}")
        print(f"  - Sample rate: {self.sample_rate} Hz")
        print(f"  - Chunk size: {self.chunk_size} samples ({self.chunk_size/self.sample_rate:.3f}s)")
        print(f"  - Language: {self.language}")
        print(f"  - VAD threshold: {self.vad_threshold}")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        if status:
            print(f"Audio callback status: {status}")
            
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put((audio_data, time.time()))
        
        return (in_data, pyaudio.paContinue)
    
    def _process_audio_chunks(self):
        print("Starting audio processing thread...")
        
        while self.is_running:
            try:
                while not self.audio_queue.empty():
                    audio_chunk, timestamp = self.audio_queue.get(block=False)

                    energy = np.sqrt(np.mean(np.square(audio_chunk)))

                    if energy > self.vad_threshold:
                        if not self.speech_active:
                            self.speech_active = True
                            self.active_speech_buffer = []
                            self.speech_frame_count = 0
                            print("Speech started")
                            
                        self.active_speech_buffer.append(audio_chunk)
                        self.speech_frame_count += 1
                        self.silence_frames = 0
                    else:
                        if self.speech_active:
                            self.silence_frames += 1
                            self.active_speech_buffer.append(audio_chunk)
                            if self.silence_frames >= self.max_silence_frames:
                                if self.speech_frame_count >= self.speech_min_frames:
                                    speech_audio = np.concatenate(self.active_speech_buffer)
                                    segment_id = int(time.time() * 1000)
                                    self.speech_segments_queue.put({
                                        "id": segment_id,
                                        "audio": speech_audio,
                                        "timestamp": timestamp - (len(self.active_speech_buffer) * self.chunk_size / self.sample_rate),
                                        "duration": len(speech_audio) / self.sample_rate
                                    })
                                    print(f"Speech segment complete: {len(speech_audio)/self.sample_rate:.2f}s")
                                else:
                                    print(f"Speech segment too short: {self.speech_frame_count * self.chunk_size / self.sample_rate:.2f}s")
                                self.speech_active = False
                                self.active_speech_buffer = None
                if self.audio_queue.empty():
                    time.sleep(0.01)
                    
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"Error in audio processing thread: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
    
    def _create_fallback_diarization(self, segment, speaker_id=None):
        # If no speaker_id is provided, use a default one
        if speaker_id is None:
            speaker_id = f"Speaker_{self.speaker_counter}"
            self.speaker_counter += 1
            
        # Create a simple one-speaker diarization result
        data = {
            "segment": None,  # Placeholder for segment object
            "label": "SPEAKER",
            "speaker": speaker_id,
            "start": 0.0,  # Start at beginning of segment
            "end": segment["duration"],  # End at end of segment
            "duration": segment["duration"]
        }
        
        return pd.DataFrame([data])
    
    def _process_speech_segments(self):
        """
        Thread 2: Process speech segments for diarization with robust error handling
        """
        print("Starting diarization thread...")
        
        while self.is_running:
            try:
                # Process speech segments as they become available
                if not self.speech_segments_queue.empty():
                    segment = self.speech_segments_queue.get(block=False)
                    
                    print(f"Processing segment {segment['id']}: {segment['duration']:.2f}s")
                    
                    try:
                        # Save audio to temporary WAV for processing
                        temp_path = os.path.join(self.output_dir, "temp", f"segment_{segment['id']}.wav")
                        
                        # Scale to int16 for WAV
                        max_value = max(np.max(np.abs(segment['audio'])), 1e-10)
                        scaling = 0.9 * 32767 / max_value
                        audio_int16 = (segment['audio'] * scaling).astype(np.int16)
                        
                        with wave.open(temp_path, 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)  # 2 bytes for int16
                            wf.setframerate(self.sample_rate)
                            wf.writeframes(audio_int16.tobytes())
                        
                        # TRY diarization with error handling
                        try:
                            # Process diarization (speaker identification)
                            diarization_results = self.diarization_pipeline.process_audio(
                                segment['audio'],
                                sample_rate=self.sample_rate,
                                min_speakers=1,
                                max_speakers=3
                            )
                            
                            # Check if we got valid results
                            if diarization_results is not None and not diarization_results.empty:
                                # Adjust timestamps to be absolute
                                diarization_results['start'] += segment['timestamp']
                                diarization_results['end'] += segment['timestamp']
                                diarization_results['duration'] = diarization_results['end'] - diarization_results['start']
                                
                                # Success!
                                print(f"Diarization found {len(diarization_results)} speaker segments")
                            else:
                                print("Empty diarization result, using fallback")
                                diarization_results = self._create_fallback_diarization(segment)
                                self.diarization_failures += 1
                        
                        except Exception as e:
                            print(f"Diarization failed: {e}")
                            diarization_results = self._create_fallback_diarization(segment)
                            self.diarization_failures += 1

                        self.transcription_queue.put({
                            "id": segment['id'],
                            "audio_path": temp_path,
                            "diarization": diarization_results,
                            "timestamp": segment['timestamp'],
                            "duration": segment['duration']
                        })
                            
                    except Exception as e:
                        print(f"Error processing segment: {e}")
                        import traceback
                        traceback.print_exc()
                        
                else:
                    time.sleep(0.1)
                    
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in diarization thread: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)
    
    def _transcribe_segment(self, audio_path):
        try:
            result = self.transcription_pipeline.transcribe_audio(
                audio_path, 
                return_timestamps=False
            )
            
            if result and "text" in result and result["text"].strip():
                return result["text"]
            print("Empty transcription, retrying with different parameters...")
            try:
                text = ""
                segments, _ = self.transcription_pipeline.model.transcribe(
                    audio_path,
                    language=self.language,
                    vad_filter=False,
                    beam_size=5
                )
                for segment in segments:
                    text += segment.text + " "
                
                return text.strip()
            except Exception as inner_e:
                print(f"Direct transcription also failed: {inner_e}")
                return ""
                
        except Exception as e:
            print(f"Error in transcription: {e}")
            return ""
    
    def _process_transcriptions(self):
        print("Starting transcription thread...")
        
        while self.is_running:
            try:
                if not self.transcription_queue.empty():
                    segment = self.transcription_queue.get(block=False)
                    
                    try:
                        full_text = self._transcribe_segment(segment['audio_path'])
                        
                        if full_text and full_text.strip():
                            results = segment['diarization'].copy()
                            if len(results) == 1:
                                main_speaker_idx = 0
                            else:
                                results['duration'] = results['end'] - results['start']
                                main_speaker_idx = results['duration'].idxmax()
                            if 'transcription' not in results.columns:
                                results['transcription'] = ""
                            results.at[main_speaker_idx, 'transcription'] = full_text
                            self.result_queue.put({
                                "id": segment['id'],
                                "results": results[results['transcription'] != ""]
                            })
                            
                            print(f"Transcription for segment {segment['id']}: '{full_text}'")
                            self.successful_segments += 1
                        else:
                            print("No transcription found for segment")
                            self.transcription_failures += 1
                            
                    except Exception as e:
                        print(f"Error in transcription processing: {e}")
                        import traceback
                        traceback.print_exc()
                        self.transcription_failures += 1
                        
                else:
                    time.sleep(0.1)
                    
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in transcription thread: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)
    
    def _display_results(self):
        print("Starting results display thread...")
        
        while self.is_running:
            try:
                if not self.result_queue.empty():
                    result = self.result_queue.get(block=False)
                    for _, row in result['results'].iterrows():
                        speaker = row['speaker']
                        text = row['transcription']
                        start = row.get('start', 0)
                        end = row.get('end', 0)

                        print("\n" + "-" * 50)
                        print(f"[{start:.2f}-{end:.2f}] {speaker}:")
                        print(f"  \"{text}\"")
                        print("-" * 50)
                        
                        self.segments.append({
                            "speaker": speaker,
                            "text": text,
                            "start": start,
                            "end": end,
                            "segment_id": result['id'],
                            "timestamp": time.time()
                        })
                    
                    self._save_results()
                    
                else:
                    time.sleep(0.1)
                    
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in results thread: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)
    
    def _save_results(self):
        if not self.segments:
            return
            
        try:
            df = pd.DataFrame(self.segments)
            csv_path = os.path.join(self.output_dir, "realtime_transcript.csv")
            df.to_csv(csv_path, index=False)

            json_path = os.path.join(self.output_dir, "realtime_transcript.json")
            with open(json_path, 'w') as f:
                json.dump({
                    "segments": self.segments,
                    "stats": {
                        "successful_segments": self.successful_segments,
                        "diarization_failures": self.diarization_failures,
                        "transcription_failures": self.transcription_failures,
                    },
                    "updated_at": time.time()
                }, f, indent=2)
                
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def start(self):
        if self.is_running:
            print("Already running!")
            return
            
        self.is_running = True

        self.segments = []
        self.speech_active = False
        self.active_speech_buffer = None
        self.diarization_failures = 0
        self.transcription_failures = 0
        self.successful_segments = 0

        while not self.audio_queue.empty():
            self.audio_queue.get()
        while not self.speech_segments_queue.empty():
            self.speech_segments_queue.get()
        while not self.transcription_queue.empty():
            self.transcription_queue.get()
        while not self.result_queue.empty():
            self.result_queue.get()
        
        try:
            self.stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback,
                input_device_index=self.device_index
            )
            print("Audio stream opened successfully")

            self.vad_thread = threading.Thread(target=self._process_audio_chunks)
            self.vad_thread.daemon = True
            self.vad_thread.start()
            
            self.diarization_thread = threading.Thread(target=self._process_speech_segments)
            self.diarization_thread.daemon = True
            self.diarization_thread.start()
            
            self.transcription_thread = threading.Thread(target=self._process_transcriptions)
            self.transcription_thread.daemon = True
            self.transcription_thread.start()
            
            self.display_thread = threading.Thread(target=self._display_results)
            self.display_thread.daemon = True
            self.display_thread.start()

            self.stream.start_stream()
            
            print("\n" + "-" * 50)
            print("🎤 Recording... Speak clearly into your microphone")
            print("Press Ctrl+C to stop recording")
            print("-" * 50 + "\n")
            
        except Exception as e:
            self.is_running = False
            print(f"Error starting speech processor: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def stop(self):
        if not self.is_running:
            print("Not running!")
            return
            
        print("Stopping speech processor...")
        self.is_running = False
        print("Removed all temp files")
        os.rmdir(os.path.join(self.output_dir, "temp"))

        try:
            if self.stream and self.stream.is_active():
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
        except Exception as e:
            print(f"Error stopping audio stream: {e}")

        for thread in [self.vad_thread, self.diarization_thread, 
                       self.transcription_thread, self.display_thread]:
            if thread and thread.is_alive():
                try:
                    thread.join(timeout=2.0)
                except Exception as e:
                    print(f"Error joining thread: {e}")

        self._save_results()

        print("\n" + "=" * 50)
        print("Session Summary:")
        print(f"  Successful transcriptions: {self.successful_segments}")
        print(f"  Diarization failures: {self.diarization_failures}")
        print(f"  Transcription failures: {self.transcription_failures}")
        print(f"  Total segments processed: {self.successful_segments + self.transcription_failures}")
        print("=" * 50)
        
        print("\nSpeech processor stopped.")
        print("\n" + "-" * 50)
        print(f"Recording stopped. Results saved to {self.output_dir}")
        print("-" * 50 + "\n")