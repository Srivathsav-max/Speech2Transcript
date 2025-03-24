"""
Base extractor module that provides common functionality for all specialized extractors.
"""
from typing import Any, Dict, List, Optional

class BaseExtractor:
    """
    Base class for all extractors in the medical summarization pipeline.
    Provides common logging and utility functions.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the extractor.
        
        Args:
            logger: Optional logger for messages
        """
        self.logger = logger
    
    def _log(self, message: str, level: str = "info") -> None:
        """
        Log messages if logger is available.
        
        Args:
            message: The message to log
            level: Log level (info, error, warning)
        """
        if self.logger:
            if level == "info":
                self.logger.info(message)
            elif level == "error":
                self.logger.error(message)
            elif level == "warning":
                self.logger.warning(message)
    
    def extract(self, *args, **kwargs) -> Any:
        """
        The main extraction method to be implemented by subclasses.
        
        Returns:
            Extracted information
        """
        raise NotImplementedError("Subclasses must implement the extract method.")
