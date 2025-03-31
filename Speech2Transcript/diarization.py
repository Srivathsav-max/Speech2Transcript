import torch
import numpy as np
import pandas as pd
from pyannote.audio import Pipeline
from typing import Optional, Union

from Speech2Transcript.utils import segments_to_df

class DiarizationPipeline:
    """
    A class to handle speaker diarization using the specified model.
    """
    def __init__(
            self,
            model_name: str = "pyannote/speaker-diarization-3.1",
            use_auth_token: Optional[str] = None,
            device: Optional[Union[str, torch.device]] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if isinstance(device, str):
            device = torch.device(device)
        
        self.model = Pipeline.from_pretrained(
            model_name,
            use_auth_token=use_auth_token
        ).to(device)

    def process_audio(
            self,
            audio: Union[str, np.ndarray], 
            sample_rate: Optional[int] = None,
            num_speakers: Optional[int] = None,
            min_speakers: Optional[int] = None,
            max_speakers: Optional[int] = None
      ) -> pd.DataFrame:                                    # expected ruturn type is a DataFrame
          
        if isinstance(audio, str):
            audio_input = audio
        else:
            audio_input = {
                "waveform": torch.from_numpy(audio[None, :]),
                "sample_rate": sample_rate
            }
        
        segment = self.model(
            audio_input,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )

        segments, labels, speakers = zip(*[(segment, label, speaker) for segment, label, speaker in segment.itertracks(yield_label=True)])

        diarization_df = segments_to_df(segments, labels, speakers)
        
        return diarization_df
