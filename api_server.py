import os
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import the Speech2Transcript components
from Speech2Transcript.transcription import TranscriptionPipeline
from Speech2Transcript.summarizers import EnterpriseSummarizer

from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='outputs')
CORS(app, resources={
    r"/*": {  # Allow CORS for all routes
        "origins": ["http://localhost:3000", "http://localhost:5173", "https://watchrx.srivathsav.me"],  # Frontend URLs
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "expose_headers": ["Content-Type", "Content-Disposition"],
        "supports_credentials": True
    }
})

# Configure error handling
@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': str(e)}), 400

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': str(e)}), 500

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'aac', 'm4a', 'audio', 'audio/wav', 'audio/mp3', 'audio/ogg', 'audio/flac', 'audio/aac', 'audio/m4a'}

# Helper function to check allowed file extensions
def allowed_file(filename):
    if '.' not in filename:
        extension = filename.split(';')[0].split('/')[1] if ';' in filename else filename.split('/')[-1]
        return extension.lower() in ALLOWED_EXTENSIONS
    return filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


# Initialize pipelines at startup
logger.info("Initializing pipelines at startup...")

logger.info("Initializing Google Speech-to-Text transcription pipeline")
transcription_pipeline = TranscriptionPipeline(
    model_name="default",  # Not used with Google STT but kept for compatibility
    device="cpu",  # Not used with Google STT but kept for compatibility
    compute_type=None,  # Not used with Google STT
    chunk_length=60,  # Maximum length for Google STT
    batch_size=1,  # Not used with Google STT but kept for compatibility
    language="en-US",  # Language code for Google STT
    beam_size=1  # Not used with Google STT but kept for compatibility
)

logger.info("Initializing Enterprise summarizer")
# Get API key from environment variables
api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
logger.info(f"API Key available: {api_key is not None}")

if not api_key:
    logger.warning("No API key found in environment variables. Please set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file")
else:
    logger.info("API key loaded from environment variables")

# Use a more resource-efficient model considering memory constraints (4GB RAM, 2 CPUs)
summarizer = EnterpriseSummarizer(
    api_key=api_key,
    model_name="gemini-1.5-flash",  # Updated to use current model
    temperature=0.2,
    max_output_tokens=2048,  # Reduced from 4096 to save memory
    enforce_hipaa=True,
    clinical_format=True,
    logger=logger  # Pass the logger for better debugging
)

logger.info("All pipelines initialized successfully")

@app.route('/')
def index():
    return jsonify({"message": "Speech2Transcript API Server is running"})

@app.route('/api/process', methods=['POST'])
def process_audio():
    logger.info("Received process request")
    logger.info(f"Files: {request.files.keys()}")
    logger.info(f"Form data: {request.form}")

    # Check if the post request has the file part
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    logger.info(f"Received file: {file.filename}")

    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Secure the filename to prevent security issues
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get parameters from request
        params = {
            'diarize': request.form.get('diarize', 'true').lower() == 'true',
            'transcribe': request.form.get('transcribe', 'true').lower() == 'true',
            'summarize': request.form.get('summarize', 'true').lower() == 'true',
            'force_process': request.form.get('force_process', 'false').lower() == 'true',
            'num_speakers': int(request.form.get('num_speakers', '0')) if request.form.get('num_speakers', '0').isdigit() else None
        }

        # Log if we're forcing processing
        if params['force_process']:
            logger.info("Force processing enabled - will process even if not telemedical")

        try:
            logger.info(f"Processing file: {filename}")
            result = {"status": "success", "filename": filename, "outputs": {}}

            # Validate audio file before processing
            try:
                # Try soundfile first (more reliable for format detection)
                try:
                    import soundfile as sf
                    info = sf.info(filepath)
                    duration = info.duration
                    sample_rate = info.samplerate
                    channels = info.channels
                    frames = info.frames
                except ImportError:
                    # Fallback to pydub if soundfile not available
                    from pydub import AudioSegment
                    audio = AudioSegment.from_file(filepath)
                    duration = len(audio) / 1000.0  # Convert ms to seconds
                    sample_rate = audio.frame_rate
                    channels = audio.channels
                    frames = len(audio.raw_data) // (audio.sample_width * channels)

                logger.info(f"Audio file info: {frames} frames, {sample_rate} Hz, {channels} channels, {duration:.2f}s")

                # Check if file is too short
                if duration < 0.1:
                    logger.warning(f"Audio file is very short ({duration:.2f}s), may not produce good results")
                elif duration < 1.0:
                    logger.warning(f"Audio file is short ({duration:.2f}s), diarization may be less accurate")

            except Exception as audio_error:
                logger.error(f"Invalid audio file format: {audio_error}")
                return jsonify({
                    "status": "error",
                    "error": f"Invalid audio file format: {str(audio_error)}. Please ensure the file is a valid audio file (WAV, MP3, etc.)"
                }), 400

            # Run transcription with Google Speech-to-Text (includes diarization)
            if params['transcribe'] or params['diarize']:
                logger.info("Starting Google Speech-to-Text with speaker diarization...")

                try:
                    # Determine speaker count parameters
                    min_speakers = 2
                    max_speakers = 6
                    if params['num_speakers']:
                        min_speakers = max_speakers = params['num_speakers']

                    transcription = transcription_pipeline.transcribe_audio(
                        filepath,
                        return_timestamps=True,
                        enable_speaker_diarization=True,
                        min_speaker_count=min_speakers,
                        max_speaker_count=max_speakers
                    )
                    
                    segments = []
                    if transcription.get("speaker_segments"):
                        # Process speaker segments
                        for i, speaker_segment in enumerate(transcription["speaker_segments"]):
                            segment = {
                                "segment": i,
                                "label": "SPEAKER",
                                "speaker": speaker_segment["speaker"],
                                "start": speaker_segment["start"],
                                "end": speaker_segment["end"],
                                "transcription": speaker_segment["text"],
                                "language": transcription["language"],
                                "language_probability": transcription["language_probability"],
                                "word_timestamps": []
                            }
                            segments.append(segment)
                    else:
                        # Fallback to a single segment with the entire transcription
                        segment = {
                            "segment": 0,
                            "label": "SPEAKER",
                            "speaker": "SPEAKER_0",
                            "start": 0.0,
                            "end": 0.0,  # Will be updated with the last word timestamp
                            "transcription": transcription["text"],
                            "language": transcription["language"],
                            "language_probability": transcription["language_probability"],
                            "word_timestamps": []
                        }
                        segments.append(segment)
                    
                    # Process word timestamps if available
                    if transcription["chunks"]:
                        # Group word timestamps by speaker segment
                        for chunk in transcription["chunks"]:
                            word_timestamp = {
                                "word": chunk["text"],
                                "start": chunk["timestamp"][0],
                                "end": chunk["timestamp"][1]
                            }
                            
                            # Find the segment this word belongs to
                            for segment in segments:
                                if (chunk["speaker"] == segment["speaker"] and 
                                    chunk["timestamp"][0] >= segment["start"] and 
                                    chunk["timestamp"][1] <= segment["end"]):
                                    segment["word_timestamps"].append(word_timestamp)
                                    break
                            else:
                                # If no matching segment found, add to the first segment as fallback
                                if segments:
                                    segments[0]["word_timestamps"].append(word_timestamp)
                        
                        # Update segment end times if needed
                        for segment in segments:
                            if segment["word_timestamps"] and segment["end"] == 0.0:
                                segment["end"] = segment["word_timestamps"][-1]["end"]
                    
                    # Create diarization data structure
                    diarization_data = {
                        "segments": segments,
                        "num_speakers": len(set(s["speaker"] for s in segments)),
                        "duration": max([s["end"] for s in segments]) if segments else 0.0
                    }
                    
                    # Store results
                    result["outputs"]["diarization"] = diarization_data
                    result["outputs"]["transcription"] = {"segments": segments}
                    
                    # Create diarization_results for summarization
                    import pandas as pd
                    
                    if segments:
                        # Create DataFrame for summarization
                        diarization_results = pd.DataFrame(segments)

                        # Store results for both diarization and transcription outputs
                        result["outputs"]["diarization"] = {
                            "segments": segments,
                            "num_speakers": len(set(s["speaker"] for s in segments)),
                            "duration": max([s["end"] for s in segments]) if segments else 0.0
                        }
                        result["outputs"]["transcription"] = {"segments": segments}
                    else:
                        logger.warning("No transcription returned from Google Speech-to-Text")
                        result["outputs"]["diarization"] = {"segments": [], "num_speakers": 0, "duration": 0.0}
                        result["outputs"]["transcription"] = {"segments": []}
                        diarization_results = pd.DataFrame()
                        
                except Exception as e:
                    logger.error(f"Error in transcription: {str(e)}")
                    result["outputs"]["diarization"] = {"segments": [], "num_speakers": 0, "duration": 0.0}
                    result["outputs"]["transcription"] = {"segments": []}
                    diarization_results = pd.DataFrame()

                # Run summarization if requested (requires transcription data)
                if params['summarize'] and (params['transcribe'] or params['diarize']):
                    if len(diarization_results) == 0:
                        logger.warning("Cannot summarize: no transcribed segments available")
                        result["outputs"]["summary"] = {
                            "summary": "No speech content detected in the audio file for summarization.",
                            "extracted_info": {
                                "is_telemedical": False,
                                "patient_name": "Unknown",
                                "provider_name": "Unknown",
                                "conditions": [],
                                "medications": [],
                                "vital_signs": {}
                            }
                        }
                        result["is_telemedical"] = False
                    else:
                        logger.info("Starting summarization with Gemini...")

                        # Create transcript data for summarizer
                        transcript_data = {
                            "segments": diarization_results.to_dict('records')
                        }
                        
                        # Process the transcript with the summarizer
                        summary_result = summarizer.process_transcript(
                            transcript_data=transcript_data,
                            text_column="transcription",
                            speaker_column="speaker",
                            force_process=params['force_process']
                        )

                        # Add summary to response
                        result["outputs"]["summary"] = {
                            "summary": summary_result["summary"],
                            "extracted_info": summary_result["extracted_info"]
                        }

                        # Add telemedical flag to the top level for easy access
                        if "is_telemedical" in summary_result["extracted_info"]:
                            result["is_telemedical"] = summary_result["extracted_info"]["is_telemedical"]
            else:
                # If neither transcription nor diarization is requested
                logger.info("No processing requested - use 'diarize' and/or 'transcribe' parameters")
                result["message"] = "No processing performed. Use 'diarize=true' and/or 'transcribe=true' parameters to process audio."

            return jsonify(result)

        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({"status": "error", "error": str(e)}), 500

    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/outputs', methods=['GET'])
def list_outputs():
    """List all available output files"""
    files = []
    for filename in os.listdir(app.config['OUTPUT_FOLDER']):
        if os.path.isfile(os.path.join(app.config['OUTPUT_FOLDER'], filename)):
            files.append({
                "filename": filename,
                "url": f"/outputs/{filename}"
            })
    return jsonify({"files": files})

@app.route('/outputs/<path:filename>')
def serve_output(filename):
    """Serve output files"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

@app.route('/api/regenerate', methods=['POST'])
def regenerate_summary():
    """Regenerate summary for an existing processed transcript"""
    logger.info("Received regenerate summary request")

    # Check if the request has the required data
    if not request.json:
        logger.error("No JSON data in request")
        return jsonify({'error': 'No JSON data provided in request'}), 400

    # Extract data from request
    data = request.json

    # Validate required fields
    if 'transcription' not in data:
        logger.error("No transcription data provided")
        return jsonify({'error': 'No transcription data provided'}), 400

    # Check if transcription segments exist
    if 'segments' not in data['transcription']:
        logger.error("No segments in transcription data")
        return jsonify({'error': 'No segments in transcription data'}), 400

    try:
        logger.info("Starting regeneration of summary...")

        # Create transcript data for summarizer
        transcript_data = {
            "segments": data['transcription']['segments']
        }

        # Optional parameters
        text_column = data.get('text_column', 'transcription')
        speaker_column = data.get('speaker_column', 'speaker')

        # Optional Gemini parameters with defaults from the original implementation
        temperature = data.get('temperature', 0.1)
        max_output_tokens = data.get('max_output_tokens', 2048)
        
        # Get force_process parameter
        force_process = data.get('force_process', False)
        if force_process:
            logger.info("Force processing enabled - will process even if not telemedical")

        # Update Gemini parameters if provided
        if temperature != summarizer.temperature:
            logger.info(f"Using custom temperature: {temperature}")
            summarizer.temperature = temperature

        if max_output_tokens != summarizer.max_output_tokens:
            logger.info(f"Using custom max_output_tokens: {max_output_tokens}")
            summarizer.max_output_tokens = max_output_tokens

        # Generate new summary with force_process parameter
        summary_result = summarizer.process_transcript(
            transcript_data=transcript_data,
            text_column=text_column,
            speaker_column=speaker_column,
            force_process=force_process
        )
        print(summary_result)

        # Return regenerated summary
        result = {
            "status": "success",
            "outputs": {
                "summary": {
                    "summary": summary_result["summary"],
                    "extracted_info": summary_result["extracted_info"]
                }
            }
        }
        print(result)
        # Add telemedical flag to the top level for easy access
        if "is_telemedical" in summary_result["extracted_info"]:
            result["is_telemedical"] = summary_result["extracted_info"]["is_telemedical"]
            
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error regenerating summary: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=True)
