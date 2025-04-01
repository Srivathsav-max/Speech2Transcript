import os
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import the Speech2Transcript components
from Speech2Transcript.diarization import DiarizationPipeline
from Speech2Transcript.transcription import TranscriptionPipeline
from Speech2Transcript.summarizers import GeminiSummarizer, detect_telemedical_content, analyze_conversation

from Speech2Transcript.utils import get_device

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

device = get_device()
logger.info(f"Using device: {device}")

logger.info("Initializing diarization pipeline")
diarization_pipeline = DiarizationPipeline(
    model_name="pyannote/speaker-diarization-3.1",
    device=device
)

logger.info("Initializing transcription pipeline")
transcription_pipeline = TranscriptionPipeline(
    model_name="large-v3",
    device=device,
    compute_type="float16",
    chunk_length=30,
    batch_size=8,
    language="en"
)

logger.info("Initializing Gemini summarizer")
gemini_summarizer = GeminiSummarizer(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model_name="gemini-2.0-flash",
    temperature=0.1,
    max_output_tokens=2048
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

        # Generate a base filename for outputs
        basename = os.path.splitext(filename)[0]

        try:
            logger.info(f"Processing file: {filename}")
            result = {"status": "success", "filename": filename, "outputs": {}}

            # Run diarization if requested
            if params['diarize']:
                logger.info("Starting diarization...")
                diarization_results = diarization_pipeline.process_audio(
                    filepath,
                    num_speakers=params['num_speakers']
                )

                # Convert diarization results to dictionary
                diarization_data = {
                    "num_speakers": diarization_results["speaker"].nunique(),
                    "duration": float(diarization_results["end"].max()),
                    "segments": diarization_results.to_dict('records')
                }

                result["outputs"]["diarization"] = diarization_data

                # Run transcription if requested
                if params['transcribe']:
                    logger.info("Starting transcription...")

                    transcription_results = transcription_pipeline.process_diarization(
                        diarization_results,
                        filepath,
                        min_segment_length=0.5,
                        temp_dir=os.path.join(app.config['OUTPUT_FOLDER'], "temp")
                    )

                    # Update the diarization results with transcriptions
                    diarization_results = transcription_results

                    result["outputs"]["transcription"] = {
                        "segments": diarization_results.to_dict('records')
                    }

                # Run summarization if requested
                if params['summarize'] and params['transcribe']:
                    logger.info("Starting summarization with Gemini...")

                    # Create transcript data for summarizer
                    transcript_data = {
                        "segments": diarization_results.to_dict('records')
                    }

                    # Pass force_process parameter to the summarizer
                    summary_result = gemini_summarizer.process_transcript(
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
        if temperature != gemini_summarizer.temperature:
            logger.info(f"Using custom temperature: {temperature}")
            gemini_summarizer.temperature = temperature

        if max_output_tokens != gemini_summarizer.max_output_tokens:
            logger.info(f"Using custom max_output_tokens: {max_output_tokens}")
            gemini_summarizer.max_output_tokens = max_output_tokens

        # Generate new summary with force_process parameter
        summary_result = gemini_summarizer.process_transcript(
            transcript_data=transcript_data,
            text_column=text_column,
            speaker_column=speaker_column,
            force_process=force_process
        )

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
    port = int(os.environ.get('PORT', 5512))
    app.run(host='0.0.0.0', port=port, debug=True)