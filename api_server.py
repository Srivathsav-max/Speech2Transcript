import os
import json
import logging
import tempfile
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pandas as pd

# Import the Speech2Transcript components
from Speech2Transcript.diarization import DiarizationPipeline
from Speech2Transcript.transcription import TranscriptionPipeline
from Speech2Transcript.summarizers import GeminiSummarizer

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
        "origins": "http://localhost:3000",  # Frontend URL
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "expose_headers": ["Content-Type", "Content-Disposition"],
        "supports_credentials": True,
        "send_wildcard": False
    }
})

# Enable CORS preflight for all routes
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response

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


# Initialize pipelines (lazy loading)
diarization_pipeline = None
transcription_pipeline = None
gemini_summarizer = None

def init_diarization_pipeline():
    global diarization_pipeline
    if diarization_pipeline is None:
        logger.info("Initializing diarization pipeline")
        diarization_pipeline = DiarizationPipeline(
            model_name="pyannote/speaker-diarization-3.1",
            device="cpu"  # Change to "cuda" if GPU is available
        )
    return diarization_pipeline

def init_transcription_pipeline():
    global transcription_pipeline
    if transcription_pipeline is None:
        logger.info("Initializing transcription pipeline")
        transcription_pipeline = TranscriptionPipeline(
            model_name="large-v3",
            device="cpu",  # Change to "cuda" if GPU is available
            compute_type="float16",
            chunk_length=30,
            batch_size=8,
            language="en"
        )
    return transcription_pipeline

def init_gemini_summarizer():
    global gemini_summarizer
    if gemini_summarizer is None:
        logger.info("Initializing Gemini summarizer")
        gemini_summarizer = GeminiSummarizer(
            model_name="gemini-2.0-pro",
            temperature=0.1,
            max_output_tokens=2048
        )
    return gemini_summarizer

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
            'num_speakers': int(request.form.get('num_speakers', '0')) if request.form.get('num_speakers', '0').isdigit() else None
        }

        # Generate a base filename for outputs
        basename = os.path.splitext(filename)[0]
        
        try:
            logger.info(f"Processing file: {filename}")
            result = {"status": "success", "filename": filename, "outputs": {}}
            
            # Run diarization if requested
            if params['diarize']:
                logger.info("Starting diarization...")
                pipeline = init_diarization_pipeline()
                
                diarization_results = pipeline.process_audio(
                    filepath,
                    num_speakers=params['num_speakers']
                )
                
                # Export results to various formats
                json_output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{basename}_diarization.json")
                from Speech2Transcript.utils import export_to_json
                export_to_json(diarization_results, json_output_path)
                
                result["outputs"]["diarization"] = {
                    "json": f"/outputs/{basename}_diarization.json",
                    "num_speakers": diarization_results["speaker"].nunique(),
                    "duration": float(diarization_results["end"].max())
                }
                
                # Run transcription if requested
                if params['transcribe']:
                    logger.info("Starting transcription...")
                    
                    transcription_pipeline = init_transcription_pipeline()
                    
                    transcription_results = transcription_pipeline.process_diarization(
                        diarization_results,
                        filepath,
                        min_segment_length=0.5,
                        temp_dir=os.path.join(app.config['OUTPUT_FOLDER'], "temp")
                    )
                    
                    # Update the diarization results with transcriptions
                    diarization_results = transcription_results
                    
                    # Export updated results with transcription
                    export_to_json(diarization_results, json_output_path)
                    
                    result["outputs"]["transcription"] = {
                        "json": f"/outputs/{basename}_diarization.json"
                    }
                
                # Run summarization if requested
                if params['summarize'] and params['transcribe']:
                    logger.info("Starting summarization with Gemini...")
                    
                    summarizer = init_gemini_summarizer()
                    
                    summary_output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{basename}_gemini_summary.json")
                    
                    summary_result = summarizer.process_transcript(
                        transcript_path=json_output_path,
                        output_path=summary_output_path,
                        text_column="transcription",
                        speaker_column="speaker"
                    )
                    
                    # Extract key information for response
                    result["outputs"]["summary"] = {
                        "json": f"/outputs/{basename}_gemini_summary.json",
                        "text": f"/outputs/{basename}_gemini_summary_summary.txt",
                        "patient_name": summary_result["extracted_info"].get("patient_name", "Unknown"),
                        "provider_name": summary_result["extracted_info"].get("provider_name", "Unknown"),
                    }
                    
                    if "conditions" in summary_result["extracted_info"]:
                        result["outputs"]["summary"]["conditions"] = summary_result["extracted_info"]["conditions"]
                    
                    if "medications" in summary_result["extracted_info"]:
                        result["outputs"]["summary"]["medications"] = summary_result["extracted_info"]["medications"]
            
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5512))
    app.run(host='0.0.0.0', port=port, debug=True)
