"""
core/audio_processor.py
Loads audio/video files, preprocesses them, and transcribes speech.
Works with all formats: .m4a.mp4, .wav, .mp3, .ogg, etc.
"""
import os, io, logging, tempfile
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

def _cfg(config, key, default):
    if isinstance(config, dict): return config.get(key, default)
    return getattr(config, key, default)

@dataclass
class AudioData:
    samples: np.ndarray
    sample_rate: int
    duration_seconds: float
    original_format: str
    file_size_bytes: int

    @property
    def is_valid(self):
        return self.samples is not None and len(self.samples) > 0

class AudioProcessor:
    def __init__(self, config):
        self.config       = config
        self.target_sr    = _cfg(config, "TARGET_SAMPLE_RATE", 16000)
        self.max_duration = _cfg(config, "MAX_AUDIO_DURATION", 300)

    # ── Load & preprocess ──────────────────────────────────────
    def load_and_preprocess(self, file_path: str) -> AudioData:
        import librosa
        from pydub import AudioSegment

        ext       = os.path.splitext(file_path)[1].lower().lstrip(".")
        file_size = os.path.getsize(file_path)

        try:
            segment = AudioSegment.from_file(file_path)
        except Exception as e:
            raise RuntimeError(
                f"Cannot decode audio file. Make sure ffmpeg is installed.\n"
                f"Run: winget install --id Gyan.FFmpeg -e\nError: {e}"
            )

        if len(segment) / 1000 > self.max_duration:
            raise ValueError(f"Audio too long ({len(segment)/1000:.0f}s). Max: {self.max_duration}s")

        buf = io.BytesIO()
        segment.export(buf, format="wav")
        buf.seek(0)

        samples, sr = librosa.load(buf, sr=self.target_sr, mono=True)
        max_amp = np.max(np.abs(samples))
        if max_amp > 0:
            samples = samples / max_amp
        samples, _ = librosa.effects.trim(samples, top_db=20)

        return AudioData(
            samples=samples,
            sample_rate=sr,
            duration_seconds=len(samples)/sr,
            original_format=ext,
            file_size_bytes=file_size,
        )

    # ── Transcription ──────────────────────────────────────────
    def transcribe(self, file_path: str) -> str:
        api_key = _cfg(self.config, "OPENAI_API_KEY", "")
        if api_key:
            return self._cloud_transcribe(file_path, api_key)
        return self._local_transcribe(file_path)

    def _preprocess_for_whisper(self, file_path: str) -> str:
        """Convert to 16kHz mono WAV and normalize for best Whisper accuracy."""
        from pydub import AudioSegment
        from pydub.effects import normalize
        try:
            seg = AudioSegment.from_file(file_path)
            seg = seg.set_channels(1).set_frame_rate(16000)
            seg = normalize(seg)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            seg.export(tmp.name, format="wav")
            return tmp.name
        except Exception:
            return file_path

    def _local_transcribe(self, file_path: str) -> str:
        try:
            import whisper
        except ImportError:
            raise RuntimeError("Run: pip install openai-whisper")

        enhanced = self._preprocess_for_whisper(file_path)
        try:
            model_name = _cfg(self.config, "LOCAL_WHISPER_MODEL", "small")
            model  = whisper.load_model(model_name)
            result = model.transcribe(
                enhanced, language="en", fp16=False,
                temperature=0.0, beam_size=5,
                condition_on_previous_text=False,
                no_speech_threshold=0.4,
            )
            return result["text"].strip()
        finally:
            if enhanced != file_path:
                try: os.remove(enhanced)
                except: pass

    def _cloud_transcribe(self, file_path: str, api_key: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        with open(file_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1", file=f,
                language="en", response_format="text"
            )
        return resp.strip() if isinstance(resp, str) else resp.text.strip()
