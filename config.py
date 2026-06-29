"""
config.py — Speech Smoothing System v2
"""
import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    # Flask
    SECRET_KEY            = os.getenv("SECRET_KEY", "speech-system-2025")
    DEBUG                 = False
    UPLOAD_FOLDER         = "uploads"
    OUTPUT_FOLDER         = "outputs"
    MAX_CONTENT_LENGTH    = 100 * 1024 * 1024

    # Allowed file types
    ALLOWED_AUDIO_EXTENSIONS = {"mp3","wav","m4a","ogg","flac","aac","wma","opus"}
    ALLOWED_VIDEO_EXTENSIONS = {"mp4","mkv","webm","avi","mov","3gp"}

    # Ollama (free local AI)
    OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

    # Whisper transcription
    OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
    LOCAL_WHISPER_MODEL = os.getenv("LOCAL_WHISPER_MODEL", "small")

    # Audio processing
    TARGET_SAMPLE_RATE = 16000   # Whisper native rate
    MAX_AUDIO_DURATION = 300

    # Trained model path
    MODEL_PATH = os.getenv("MODEL_PATH", "models/severity_model.pkl")

    # UASpeech dataset path
    UASPEECH_PATH = os.getenv("UASPEECH_PATH", "")

    # Severity thresholds
    SEVERITY_MILD_MIN     = 65
    SEVERITY_MODERATE_MIN = 38
