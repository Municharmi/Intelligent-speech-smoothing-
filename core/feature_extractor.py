"""
core/feature_extractor.py
Extracts acoustic + linguistic features from audio.
These features are used by the severity classifier.
"""
import logging
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def _cfg(config, key, default):
    if isinstance(config, dict): return config.get(key, default)
    return getattr(config, key, default)

@dataclass
class AudioFeatures:
    # Prosodic
    pitch_mean_hz:          float = 0.0
    pitch_std_hz:           float = 0.0
    pitch_range_hz:         float = 0.0
    voiced_ratio:           float = 0.0
    speech_rate_syl_per_sec:float = 0.0
    pause_ratio:            float = 0.0
    pause_count:            int   = 0
    mean_pause_duration_s:  float = 0.0
    npvi:                   float = 0.0

    # Spectral
    spectral_centroid_mean: float = 0.0
    spectral_centroid_std:  float = 0.0
    spectral_flatness_mean: float = 0.0
    zcr_mean:               float = 0.0
    zcr_std:                float = 0.0

    # Energy
    rms_mean:               float = 0.0
    rms_std:                float = 0.0
    energy_dynamic_range:   float = 0.0

    # Voice quality
    jitter_percent:         float = 0.0
    shimmer_percent:        float = 0.0

    # MFCC
    mfcc_mean:              List[float] = field(default_factory=list)
    mfcc_std:               List[float] = field(default_factory=list)

    # Linguistic
    utterance_length_words: int   = 0
    type_token_ratio:       float = 0.0
    mean_word_length:       float = 0.0
    verb_ratio:             float = 0.0
    function_word_ratio:    float = 0.0
    oov_rate:               float = 0.0

    # Meta
    duration_seconds:       float = 0.0
    sample_rate:            int   = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
            elif isinstance(v, list):
                d[k] = [round(x, 4) if isinstance(x, float) else x for x in v]
        return d

FUNCTION_WORDS = {
    "the","a","an","is","are","was","were","be","have","has","do","does",
    "will","would","could","should","to","of","in","for","on","with","at",
    "and","or","but","i","you","he","she","it","we","they","my","your",
    "his","her","its","me","him","us","them","this","that","please"
}

class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.sr     = _cfg(config, "TARGET_SAMPLE_RATE", 16000)

    def extract(self, audio_data, transcript: str = "") -> AudioFeatures:
        from core.audio_processor import AudioData
        f  = AudioFeatures(
            duration_seconds=round(audio_data.duration_seconds, 3),
            sample_rate=audio_data.sample_rate,
        )
        y  = audio_data.samples
        sr = audio_data.sample_rate

        self._mfcc(y, sr, f)
        self._prosodic(y, sr, f)
        self._spectral(y, sr, f)
        self._energy(y, sr, f)
        self._voice_quality(y, sr, f)
        if transcript.strip():
            self._linguistic(transcript, f)

        logger.info(
            "Features: rate=%.1f syl/s  pauses=%.0f%%  voiced=%.0f%%  pitch=%.0fHz",
            f.speech_rate_syl_per_sec, f.pause_ratio*100,
            f.voiced_ratio*100, f.pitch_mean_hz
        )
        return f

    def _mfcc(self, y, sr, f):
        import librosa
        mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        f.mfcc_mean = mfcc.mean(axis=1).tolist()
        f.mfcc_std  = mfcc.std(axis=1).tolist()

    def _prosodic(self, y, sr, f):
        import librosa
        from scipy.signal import find_peaks

        # Pitch
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=50, fmax=400, sr=sr,
            frame_length=2048, hop_length=512
        )
        voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
        if len(voiced_f0) > 0:
            f.pitch_mean_hz  = float(np.mean(voiced_f0))
            f.pitch_std_hz   = float(np.std(voiced_f0))
            f.pitch_range_hz = float(np.max(voiced_f0) - np.min(voiced_f0))
        f.voiced_ratio = float(np.sum(voiced_flag) / max(len(voiced_flag), 1))

        # Pauses via RMS
        hop    = 512
        rms    = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
        thresh = np.mean(rms) * 0.15
        fdur   = hop / sr
        silent = rms < thresh

        pauses, in_p, p_start = [], False, 0
        for i, s in enumerate(silent):
            if s and not in_p:  in_p=True;  p_start=i
            elif not s and in_p:
                in_p=False
                d = (i - p_start) * fdur
                if d > 0.15: pauses.append(d)
        if in_p:
            d = (len(silent)-p_start)*fdur
            if d > 0.15: pauses.append(d)

        f.pause_count          = len(pauses)
        f.mean_pause_duration_s= float(np.mean(pauses)) if pauses else 0.0
        total_pause            = sum(pauses)
        f.pause_ratio          = float(total_pause / max(f.duration_seconds, 0.1))

        # Speech rate via energy peaks
        speech_dur = max(f.duration_seconds - total_pause, 0.1)
        env        = np.convolve(rms, np.ones(5)/5, mode="same")
        peaks, _   = find_peaks(env, distance=int(0.12/fdur), prominence=0.01)
        f.speech_rate_syl_per_sec = round(len(peaks) / speech_dur, 3)

        # nPVI
        if len(voiced_f0) > 2:
            intervals = np.diff(voiced_f0)
            sums = np.abs(intervals[:-1]) + np.abs(intervals[1:])
            ratios = np.where(sums>0, 200*np.abs(intervals[:-1]-intervals[1:])/sums, 0)
            f.npvi = float(np.mean(ratios))

    def _spectral(self, y, sr, f):
        import librosa
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        zcr      = librosa.feature.zero_crossing_rate(y)[0]
        f.spectral_centroid_mean = float(np.mean(centroid))
        f.spectral_centroid_std  = float(np.std(centroid))
        f.spectral_flatness_mean = float(np.mean(flatness))
        f.zcr_mean               = float(np.mean(zcr))
        f.zcr_std                = float(np.std(zcr))

    def _energy(self, y, sr, f):
        import librosa
        rms = librosa.feature.rms(y=y)[0]
        f.rms_mean = float(np.mean(rms))
        f.rms_std  = float(np.std(rms))
        rms_db     = librosa.amplitude_to_db(rms + 1e-9)
        f.energy_dynamic_range = float(np.max(rms_db) - np.min(rms_db))

    def _voice_quality(self, y, sr, f):
        import librosa
        hop = 512
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=50, fmax=400, sr=sr,
            frame_length=2048, hop_length=hop
        )
        voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
        if len(voiced_f0) < 4:
            return
        periods = (sr / voiced_f0).astype(int)

        # Jitter
        diffs = np.abs(np.diff(periods.astype(float)))
        f.jitter_percent = round(float(min(100*np.mean(diffs)/max(np.mean(periods),1), 30)), 3)

        # Shimmer
        cursor, amps = 0, []
        for p in periods:
            end = cursor + p
            if end > len(y): break
            frame = y[cursor:end]
            amps.append(np.sqrt(np.mean(frame**2)) + 1e-9)
            cursor = end
        if len(amps) > 2:
            amp_arr  = np.array(amps)
            amp_diff = np.abs(np.diff(amp_arr))
            f.shimmer_percent = round(float(min(100*np.mean(amp_diff)/np.mean(amp_arr), 50)), 3)

    def _linguistic(self, transcript: str, f: AudioFeatures):
        words = transcript.lower().strip().split()
        if not words:
            return
        f.utterance_length_words = len(words)
        f.mean_word_length       = round(np.mean([len(w) for w in words]), 3)
        f.type_token_ratio       = round(len(set(words)) / len(words), 4)
        f.function_word_ratio    = round(sum(1 for w in words if w in FUNCTION_WORDS) / len(words), 4)

        VERBS = {"is","are","was","were","am","want","need","like","go","come",
                 "help","give","eat","drink","play","feel","know","do","have","has"}
        f.verb_ratio = round(sum(1 for w in words if w in VERBS) / len(words), 4)

        oov = [w for w in words if not w.isalpha() or (len(w)==1 and w not in {"a","i"})]
        f.oov_rate = round(len(oov) / len(words), 4)
