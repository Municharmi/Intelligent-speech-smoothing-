"""
api/routes.py
Flask API routes. Uses Excel transcriptions as ground truth
when available, falls back to Whisper for unknown files.
"""
import os, uuid, json, traceback
from flask import Blueprint, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename

from core.audio_processor import AudioProcessor
from core.feature_extractor import FeatureExtractor
from core.severity_classifier import SeverityClassifier
from core.speech_interpreter import SpeechInterpreter
from utils.validators import validate_file, validate_text_input
from utils.response import success_response, error_response

api_bp = Blueprint("api", __name__)
_uploaded_files    = {}
_transcript_lookup = {}

def _cfg(config, key, default):
    if isinstance(config, dict): return config.get(key, default)
    return getattr(config, key, default)

def _load_lookup():
    global _transcript_lookup
    if not _transcript_lookup:
        p = "data/transcript_lookup.json"
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                _transcript_lookup = json.load(f)
            print(f"[API] Loaded {len(_transcript_lookup)} transcript entries from lookup")
    return _transcript_lookup

def _lookup(filename: str) -> str:
    lu    = _load_lookup()
    fname = filename.lower().strip()
    if fname in lu: return lu[fname]
    for ext in [".m4a.mp4",".mp4",".m4a",".mp3",".wav",".ogg"]:
        if fname.endswith(ext):
            stem = fname[:-len(ext)]
            if stem in lu: return lu[stem]
    return ""

def _save_upload(file):
    filename    = secure_filename(file.filename)
    ext         = filename.rsplit(".",1)[-1].lower() if "." in filename else "mp4"
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    upload_dir  = _cfg(current_app.config, "UPLOAD_FOLDER", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    save_path   = os.path.join(upload_dir, unique_name)
    file.save(save_path)
    return save_path, unique_name


# ── POST /api/analyze/audio ────────────────────────────────────
@api_bp.route("/analyze/audio", methods=["POST"])
def analyze_audio():
    if "file" not in request.files:
        return error_response("No file provided.", 400)

    file              = request.files["file"]
    age_group         = request.form.get("age_group",      "adult")
    known_severity    = request.form.get("known_severity", "auto")
    manual_transcript = request.form.get("transcript",     "").strip()

    valid, msg = validate_file(file, current_app.config)
    if not valid:
        return error_response(msg, 400)

    file_path = None
    try:
        original_filename = file.filename
        file_path, unique_name = _save_upload(file)
        _uploaded_files[unique_name] = file_path

        # ── Transcript priority ──────────────────────────────
        # 1. User manually corrected it in the UI
        # 2. Excel ground truth lookup
        # 3. Whisper auto-transcription
        if manual_transcript:
            raw_transcript     = manual_transcript
            transcript_source  = "manual"
        else:
            excel = _lookup(original_filename)
            if excel:
                raw_transcript    = excel
                transcript_source = "excel"
            else:
                processor         = AudioProcessor(current_app.config)
                raw_transcript    = processor.transcribe(file_path)
                transcript_source = "whisper"

        # ── Feature extraction ───────────────────────────────
        processor  = AudioProcessor(current_app.config)
        audio_data = processor.load_and_preprocess(file_path)
        extractor  = FeatureExtractor(current_app.config)
        features   = extractor.extract(audio_data, transcript=raw_transcript)

        # ── Severity classification ──────────────────────────
        classifier      = SeverityClassifier(current_app.config)
        severity_result = classifier.classify(
            features=features, transcript=raw_transcript,
            known_severity=known_severity, age_group=age_group,
        )

        # ── AI interpretation ────────────────────────────────
        interpreter    = SpeechInterpreter(current_app.config)
        interpretation = interpreter.interpret(
            raw_text=raw_transcript, features=features,
            severity=severity_result, age_group=age_group,
        )

        return success_response({
            "file":               unique_name,
            "original_filename":  original_filename,
            "transcript":         raw_transcript,
            "transcript_source":  transcript_source,
            "features":           features.to_dict(),
            "severity":           severity_result,
            "interpretation":     interpretation,
        })

    except Exception as exc:
        traceback.print_exc()
        return error_response(f"Processing failed: {str(exc)}", 500)


# ── GET /api/audio/<filename> ──────────────────────────────────
@api_bp.route("/audio/<filename>", methods=["GET"])
def serve_audio(filename):
    safe = secure_filename(filename)
    path = _uploaded_files.get(safe)
    if not path or not os.path.exists(path):
        upload_dir = _cfg(current_app.config, "UPLOAD_FOLDER", "uploads")
        path = os.path.join(upload_dir, safe)
    if not os.path.exists(path):
        return error_response("Audio not found.", 404)
    return send_file(path, mimetype="audio/mpeg")


# ── POST /api/analyze/text ─────────────────────────────────────
@api_bp.route("/analyze/text", methods=["POST"])
def analyze_text():
    body           = request.get_json(silent=True) or {}
    text           = body.get("text","").strip()
    age_group      = body.get("age_group","adult")
    known_severity = body.get("known_severity","auto")

    valid, msg = validate_text_input(text)
    if not valid:
        return error_response(msg, 400)

    try:
        classifier      = SeverityClassifier(current_app.config)
        severity_result = classifier.classify_from_text(
            transcript=text, known_severity=known_severity, age_group=age_group)

        interpreter    = SpeechInterpreter(current_app.config)
        interpretation = interpreter.interpret(
            raw_text=text, features=None,
            severity=severity_result, age_group=age_group)

        return success_response({
            "transcript":     text, "features": None,
            "severity":       severity_result,
            "interpretation": interpretation,
        })
    except Exception as exc:
        traceback.print_exc()
        return error_response(f"Analysis failed: {str(exc)}", 500)


# ── GET /api/health ────────────────────────────────────────────
@api_bp.route("/health", methods=["GET"])
def health():
    return success_response({"status":"ok","service":"Speech Smoothing System v2"})
