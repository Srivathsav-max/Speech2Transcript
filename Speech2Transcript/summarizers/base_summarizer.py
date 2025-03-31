import json
import os
import abc
import logging
from typing import Dict, Any, Optional, Union, List


class BaseSummarizer(abc.ABC):
    """
    Abstract base class for all summarizers defining the common interface.
    
    All summarizer implementations should inherit from this class and
    implement the abstract methods.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize a base summarizer.
        
        Args:
            logger: Optional logger for messages
        """
        self.logger = logger
        self._log("Summarizer initialized")
    
    def _log(self, message: str, level: str = "info") -> None:
        """
        Log messages if logger is available.
        
        Args:
            message: Message to log
            level: Log level (info, error, warning)
        """
        if self.logger:
            if level == "info":
                self.logger.info(message)
            elif level == "error":
                self.logger.error(message)
            elif level == "warning":
                self.logger.warning(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def load_transcript(self, transcript_path: Optional[str] = None, 
                        transcript_data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Load transcript data from a file or directly from provided data.
        
        Args:
            transcript_path: Path to a JSON transcript file
            transcript_data: Direct transcript data
            
        Returns:
            Dictionary containing transcript data or None if loading fails
        """
        if transcript_path and not transcript_data:
            try:
                with open(transcript_path, 'r') as f:
                    transcript_data = json.load(f)
                self._log(f"Loaded transcript from {transcript_path}")
                return transcript_data
            except Exception as e:
                self._log(f"Error loading transcript: {e}", level="error")
                return None
        
        return transcript_data
    
    def save_results(self, results: Dict[str, Any], output_path: Optional[str] = None) -> bool:
        """
        Save results to file.
        
        Args:
            results: Results to save
            output_path: Path to save results
            
        Returns:
            True if save successful, False otherwise
        """
        if not output_path:
            return False
            
        try:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            
            # Save JSON results
            with open(output_path, 'w', encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            
            # Save text summary if available
            if "summary" in results or "detailed_summary" in results:
                summary_text = results.get("detailed_summary") or results.get("summary", "")
                summary_path = output_path.replace('.json', '_summary.txt')
                with open(summary_path, 'w', encoding="utf-8") as f:
                    f.write(summary_text)
                
                self._log(f"Saved summary to {summary_path}")
            
            self._log(f"Saved results to {output_path}")
            return True
            
        except Exception as e:
            self._log(f"Error saving results: {e}", level="error")
            return False
    
    @abc.abstractmethod
    def extract_text_from_segments(self, segments: List[Dict[str, Any]], 
                                  text_column: str = "transcription", 
                                  speaker_column: str = "speaker") -> Dict[str, Any]:
        """
        Extract and format text from transcript segments.
        
        Args:
            segments: List of transcript segments
            text_column: Name of the column containing the transcript text
            speaker_column: Name of the column containing the speaker identifier
            
        Returns:
            Dictionary with extracted text information
        """
        pass
    
    @abc.abstractmethod
    def process_transcript(self, transcript_path: Optional[str] = None,
                          transcript_data: Optional[Dict[str, Any]] = None,
                          output_path: Optional[str] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Process a transcript and generate a summary.
        
        Args:
            transcript_path: Path to transcript file (JSON)
            transcript_data: Alternatively, provide transcript data directly
            output_path: Optional path to save results
            **kwargs: Additional parameters for specific summarizer implementations
            
        Returns:
            Dictionary with generated summary and extracted data
        """
        pass
