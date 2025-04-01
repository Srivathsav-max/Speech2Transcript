import os
import json
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import base summarizer
from .base_summarizer import BaseSummarizer

# Import Gemini helpers
from .gemini_helpers import (
    extract_conversation_structure,
    identify_speakers,
    extract_basic_info,
    generate_gemini_prompt,
    prepare_gemini_extraction_prompt
)

# Import telemedical detection utilities
from .telemedical_detector import detect_telemedical_content, get_non_telemedical_message

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class GeminiSummarizer(BaseSummarizer):
    """
    Gemini-powered medical transcript summarizer.

    Uses Google's Gemini API to generate natural, cohesive summaries
    from medical conversation transcripts.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-pro",
        logger: Optional[logging.Logger] = None,
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
        top_p: float = 0.95,
        top_k: int = 64
    ):
        """
        Initialize Gemini summarizer.

        Args:
            api_key: Google API key for Gemini access (default: uses GOOGLE_API_KEY env var)
            model_name: Gemini model to use
            logger: Optional logger for messages
            temperature: Model temperature for generation (lower = more deterministic)
            max_output_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
        """
        super().__init__(logger)

        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.top_k = top_k
        
        # Check if Gemini is available and configure it
        if GEMINI_AVAILABLE and self.api_key:
            try:
                # Initialize with client-based approach
                self._log(f"Initializing Gemini API client with model: {model_name}")
                self.client = genai.Client(api_key=self.api_key)

                # Try to list models to confirm connection
                try:
                    models = self.client.models.list_models()
                    model_names = [model.name for model in models]
                    self._log(f"Available Gemini models: {', '.join(model_names)}")
                    
                    # Check if our model is available
                    if not any(self.model_name in model_name for model_name in model_names):
                        closest_match = next((m for m in model_names if "gemini" in m.lower()), None)
                        if closest_match:
                            self._log(f"Model {self.model_name} not found, using {closest_match} instead", level="warning")
                            self.model_name = closest_match
                except Exception as e:
                    self._log(f"Could not retrieve list of available models: {e}", level="warning")

                self.gemini_available = True
            except Exception as e:
                self._log(f"Error initializing Gemini API: {e}", level="error")
                self.gemini_available = False
        else:
            self._log("Gemini API not available or no API key provided", level="warning")
            self.gemini_available = False

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
        # Use our helper function for consistent conversation extraction
        return extract_conversation_structure(segments, text_column, speaker_column)

    def _generate_summary_with_gemini(self, prompt: str) -> str:
        """
        Generate summary using Gemini API.

        Args:
            prompt: Formatted prompt for Gemini

        Returns:
            Generated summary text
        """
        if not self.gemini_available:
            error_msg = "Gemini API not available. Please check your API key and connection."
            self._log(error_msg, level="error")
            return error_msg

        try:
            self._log("Sending prompt to Gemini API...")
            self._log(f"Using model: {self.model_name}")

            # Generate content using the simplest approach
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )

            self._log("Successfully generated content with Gemini")

            # Extract the text from the response
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'parts'):
                parts_text = ''.join([part.text for part in response.parts if hasattr(part, 'text')])
                if parts_text:
                    return parts_text

            # Last resort - convert to string
            return str(response)

        except Exception as e:
            error_msg = f"Error generating content with Gemini: {str(e)}"
            self._log(error_msg, level="error")
            self._log(f"Error details: {type(e).__name__}: {str(e)}", level="error")

            # Return a fallback message
            return f"Error generating content: {str(e)}"
            
    def _extract_entities_with_gemini(self, conversation_text: str) -> Dict[str, Any]:
        """
        Extract structured entities from the conversation using Gemini.
        
        Args:
            conversation_text: The full conversation text
            
        Returns:
            Dictionary with extracted entities
        """
        # Create a specialized prompt for entity extraction
        extraction_prompt = prepare_gemini_extraction_prompt(conversation_text)
        
        try:
            self._log("Extracting entities from conversation with Gemini...")
            
            # The extraction uses the same approach as regular generation
            # Note: We can't change temperature directly anymore with the simplified approach
            self._log("Using fixed parameters for entity extraction")
            
            # Generate extraction response
            response_text = self._generate_summary_with_gemini(extraction_prompt)
            
            # Parse the JSON response
            try:
                # Clean up the response - sometimes Gemini includes markdown code blocks
                if response_text.startswith("```"):
                    # Extract content between code block markers
                    start_idx = response_text.find("\n") + 1
                    end_idx = response_text.rfind("```")
                    if end_idx > start_idx:
                        response_text = response_text[start_idx:end_idx].strip()
                    else:
                        # Just remove the first markdown marker if no ending marker
                        response_text = response_text[start_idx:].strip()
                        
                # Sometimes JSON responses include the language identifier
                if response_text.startswith("json"):
                    response_text = response_text[4:].strip()
                        
                self._log("Attempting to parse JSON response")
                extracted_data = json.loads(response_text)
                self._log("Successfully extracted structured data from conversation")
                return extracted_data
            except json.JSONDecodeError as e:
                self._log(f"Error parsing Gemini extraction response as JSON: {e}", level="error")
                self._log(f"Raw response: {response_text}", level="error")
                return {}
                
        except Exception as e:
            self._log(f"Error in entity extraction: {e}", level="error")
            return {}

    def process_transcript(self,
                          transcript_path: Optional[str] = None,
                          transcript_data: Optional[Dict[str, Any]] = None,
                          output_path: Optional[str] = None,
                          text_column: str = "transcription",
                          speaker_column: str = "speaker",
                          force_process: bool = False) -> Dict[str, Any]:
        """
        Process a transcript and generate a comprehensive summary.

        Args:
            transcript_path: Path to transcript file (JSON)
            transcript_data: Alternatively, provide transcript data directly
            output_path: Optional path to save results
            text_column: Column name containing the transcription text
            speaker_column: Column name containing the speaker ID
            force_process: If True, process even when content is not telemedical

        Returns:
            Dictionary with the summary and extracted information
        """
        # Load transcript
        data = self.load_transcript(transcript_path, transcript_data)
        if not data:
            self._log("No transcript data available", level="error")
            return {"error": "No transcript data available"}

        # Extract segments
        segments = data.get("segments", [])
        if not segments:
            self._log("No segments found in transcript", level="error")
            return {"error": "No segments found in transcript"}

        self._log(f"Processing transcript with {len(segments)} segments")

        # Extract text from segments
        text_data = self.extract_text_from_segments(segments, text_column, speaker_column)
        
        # Check if the conversation is telemedical before proceeding with expensive API calls
        is_telemedical, matched_categories = detect_telemedical_content(text_data["full_text"])
        self._log(f"Telemedical detection: {is_telemedical}")
        
        # If content is not telemedical and we're not forcing processing, return early
        if not is_telemedical and not force_process:
            self._log("Content is not telemedical - skipping Gemini API calls to save costs", level="warning")
            return {
                "summary": get_non_telemedical_message(),
                "extracted_info": {
                    "is_telemedical": False,
                    "matched_categories": list(matched_categories.keys()) if matched_categories else []
                }
            }
        
        # Content is telemedical or we're forcing processing
        if not is_telemedical and force_process:
            self._log("Content is not telemedical, but force_process=True - proceeding with Gemini API", level="info")

        # Identify speakers
        speakers = identify_speakers(text_data["conversation"])
        self._log(f"Identified speakers: Patient={speakers.get('patient')}, Care Manager={speakers.get('care_manager')}")

        # Extract basic information for context (fallback for prompting)
        basic_info = extract_basic_info(text_data["conversation"], speakers)
        
        # Use Gemini for detailed entity extraction
        self._log("Extracting detailed entities using Gemini")
        extracted_entities = self._extract_entities_with_gemini(text_data["full_text"])
        
        # Merge extracted entities with basic info, preferring Gemini results if available
        merged_info = {}
        
        # Get patient name
        if extracted_entities and "patient_name" in extracted_entities and extracted_entities["patient_name"].get("value"):
            merged_info["patient_name"] = extracted_entities["patient_name"]["value"]
        else:
            merged_info["patient_name"] = basic_info["patient_name"]
            
        # Get provider name
        if extracted_entities and "provider_name" in extracted_entities and extracted_entities["provider_name"].get("value"):
            merged_info["provider_name"] = extracted_entities["provider_name"]["value"]
        else:
            merged_info["provider_name"] = basic_info["provider_name"]
            
        # Get conditions
        if extracted_entities and "conditions" in extracted_entities and extracted_entities["conditions"].get("list"):
            merged_info["conditions"] = extracted_entities["conditions"]["list"]
        else:
            merged_info["conditions"] = basic_info["conditions"]
            
        # Get medications
        if extracted_entities and "medications" in extracted_entities and extracted_entities["medications"].get("list"):
            merged_info["medications"] = [med["name"] for med in extracted_entities["medications"]["list"]]
        else:
            merged_info["medications"] = basic_info["medications"]
            
        # Copy vital signs
        merged_info["vital_signs"] = {}
        if extracted_entities and "vital_signs" in extracted_entities:
            for key, value in extracted_entities["vital_signs"].items():
                if value and key != "confidence":
                    merged_info["vital_signs"][key] = value
        else:
            merged_info["vital_signs"] = basic_info["vital_signs"]

        # Generate prompt for Gemini
        summary_prompt = generate_gemini_prompt(text_data["full_text"], merged_info, "summary")

        # Generate summary with Gemini
        summary = self._generate_summary_with_gemini(summary_prompt)

        # Create result object
        result = {
            "summary": summary,
            "extracted_info": {
                **merged_info,
                "is_telemedical": True,  # Mark as telemedical since we processed it
                "speakers": speakers,
                "detailed_entities": extracted_entities
            }
        }

        # Save results if output path provided
        if output_path:
            self.save_results(result, output_path)

        return result
