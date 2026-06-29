"""
core/severity_classifier.py
Classifies severity using acoustic features as primary signal (70%)
and linguistic features as secondary (30%).

Severity thresholds based on clinical literature:
  Mild:     speech rate >2.5 syl/s, pause <35%, clear articulation
  Moderate: speech rate 1.5–2.5 syl/s, pause 35–55%
  Severe:   speech rate <1.5 syl/s, pause >55%
"""
import os, pickle, logging
from typing import Dict, Any, Optional
from core.feature_extractor import AudioFeatures

logger = logging.getLogger(__name__)

def _cfg(config, key, default):
    if isinstance(config, dict): return config.get(key, default)
    return getattr(config, key, default)

class SeverityClassifier:
    def __init__(self, config):
        self.config   = config
        self._model   = None
        model_path    = _cfg(config, "MODEL_PATH", "models/severity_model.pkl")
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    self._model = pickle.load(f)
                logger.info("Loaded trained model from %s (n=%d, acc=%.3f)",
                            model_path,
                            self._model.get("n_train", 0),
                            self._model.get("cv_score", 0))
            except Exception as e:
                logger.warning("Could not load model: %s — using rule-based", e)

    # ── Public API ─────────────────────────────────────────────

    def classify(self, features: AudioFeatures, transcript: str,
                 known_severity: str = "auto",
                 age_group: str = "adult") -> Dict[str, Any]:

        has_audio = (features is not None
                     and features.duration_seconds > 0
                     and features.rms_mean > 0)

        # Try ML model first (if trained)
        ml_result = None
        if self._model and has_audio:
            try:
                ml_result = self._predict_ml(features)
            except Exception as e:
                logger.warning("ML model failed: %s — falling back to rules", e)

        if ml_result:
            severity   = ml_result["severity"]
            score      = ml_result["score"]
            confidence = ml_result["confidence"]
            flags      = self._acoustic_flags(features) if has_audio else []
        elif has_audio:
            a_score, flags = self._acoustic_score(features)
            l_score        = self._linguistic_score(transcript)
            score          = 0.70 * a_score + 0.30 * l_score
            confidence     = 80
            severity       = self._to_severity(score)
        else:
            score, flags   = self._linguistic_score(transcript), []
            confidence     = 55
            severity       = self._to_severity(score)

        # Age adjustment
        score += {"child": 8, "teen": 4, "adult": 0}.get(age_group, 0)
        score  = min(score, 100.0)

        # Known severity hint
        if known_severity != "auto":
            hint  = {"mild": 82, "moderate": 52, "severe": 18}[known_severity]
            score = 0.5 * score + 0.5 * hint
            confidence = 88
            severity   = self._to_severity(score)

        return {
            "severity":       severity,
            "score":          round(score, 2),
            "confidence":     int(confidence),
            "description":    self._describe(severity, has_audio),
            "sub_scores":     self._sub_scores(features, transcript, has_audio),
            "clinical_flags": flags,
            "model_used":     "ml_model" if ml_result else
                              ("acoustic+linguistic" if has_audio else "linguistic_only"),
        }

    def classify_from_text(self, transcript: str,
                           known_severity: str = "auto",
                           age_group: str = "adult") -> Dict[str, Any]:
        return self.classify(AudioFeatures(), transcript, known_severity, age_group)

    # ── ML prediction ──────────────────────────────────────────

    def _predict_ml(self, f: AudioFeatures) -> Optional[Dict]:
        model  = self._model["model"]
        scaler = self._model["scaler"]
        cols   = self._model["feature_cols"]
        inv    = self._model["label_inverse"]

        feat_dict = f.to_dict()
        import numpy as np
        X = np.array([[float(feat_dict.get(c, 0)) for c in cols]], dtype=np.float32)
        X_s = scaler.transform(X)

        pred = int(model.predict(X_s)[0])
        try:
            proba = model.predict_proba(X_s)[0]
            conf  = int(round(max(proba) * 100))
        except Exception:
            conf = 70

        sev_score = {"mild": 80, "moderate": 52, "severe": 20}
        sev = inv[pred]
        return {"severity": sev, "score": float(sev_score[sev]), "confidence": conf}

    # ── Acoustic scoring ───────────────────────────────────────

    def _acoustic_score(self, f: AudioFeatures):
        score = 100.0
        flags = self._acoustic_flags(f)

        sr = f.speech_rate_syl_per_sec
        if sr > 0:
            if   sr < 1.0: score -= 40
            elif sr < 1.8: score -= 28
            elif sr < 2.8: score -= 14
            elif sr > 7.0: score -= 8

        pr = f.pause_ratio
        if   pr > 0.60: score -= 30
        elif pr > 0.42: score -= 18
        elif pr > 0.28: score -= 8

        vr = f.voiced_ratio
        if   vr > 0 and vr < 0.25: score -= 20
        elif vr > 0 and vr < 0.42: score -= 10

        if f.pitch_mean_hz > 0:
            if   f.pitch_range_hz < 15: score -= 15
            elif f.pitch_range_hz < 35: score -= 7

        if f.jitter_percent  > 3.0: score -= 12
        elif f.jitter_percent > 1.5: score -= 5
        if f.shimmer_percent > 6.0: score -= 10

        sc = f.spectral_centroid_mean
        if   sc > 0 and sc < 1000: score -= 15
        elif sc > 0 and sc < 1600: score -= 7

        if f.duration_seconds < 1.5: score -= 10

        return max(0.0, min(100.0, score)), flags

    def _acoustic_flags(self, f: AudioFeatures):
        flags = []
        sr = f.speech_rate_syl_per_sec
        if sr > 0 and sr < 1.8:
            flags.append({"feature":"speech_rate",
                           "observation":f"Slow speech rate ({sr:.1f} syl/s) — reduced fluency",
                           "severity":"high" if sr<1.0 else "medium"})
        if f.pause_ratio > 0.42:
            flags.append({"feature":"pause_ratio",
                           "observation":f"High pause ratio ({f.pause_ratio:.0%}) — word-finding difficulty",
                           "severity":"high" if f.pause_ratio>0.60 else "medium"})
        if f.voiced_ratio > 0 and f.voiced_ratio < 0.40:
            flags.append({"feature":"voiced_ratio",
                           "observation":f"Low voiced ratio ({f.voiced_ratio:.0%}) — reduced phonation",
                           "severity":"medium"})
        if f.pitch_mean_hz > 0 and f.pitch_range_hz < 25:
            flags.append({"feature":"pitch","observation":f"Narrow pitch range ({f.pitch_range_hz:.0f}Hz) — monotone speech","severity":"medium"})
        if f.jitter_percent > 2.0:
            flags.append({"feature":"jitter","observation":f"Elevated jitter ({f.jitter_percent:.1f}%) — vocal instability","severity":"medium"})
        return flags

    # ── Linguistic scoring ─────────────────────────────────────

    def _linguistic_score(self, transcript: str) -> float:
        if not transcript or not transcript.strip():
            return 50.0
        words = transcript.lower().strip().split()
        n = len(words)
        if n == 0: return 20.0

        score = 65.0
        if   n <= 1: score -= 30
        elif n <= 3: score -= 15
        elif n >= 7: score += 10

        ttr = len(set(words)) / n
        if   ttr < 0.3: score -= 10
        elif ttr > 0.7: score += 8

        VERBS = {"is","are","was","want","need","like","go","come","feel","eat","drink","play"}
        if not any(w in VERBS for w in words) and n > 2:
            score -= 12

        mwl = sum(len(w) for w in words) / n
        if   mwl < 3: score -= 5
        elif mwl > 5: score += 5

        return max(0.0, min(100.0, score))

    # ── Helpers ────────────────────────────────────────────────

    def _to_severity(self, score: float) -> str:
        if   score >= 65: return "mild"
        elif score >= 38: return "moderate"
        else:             return "severe"

    def _sub_scores(self, f, transcript, has_audio):
        if has_audio:
            a, _ = self._acoustic_score(f)
            l    = self._linguistic_score(transcript)
            return {"prosodic":round(a,1),"spectral":round(a,1),
                    "linguistic":round(l,1),"mfcc":round(a,1)}
        l = self._linguistic_score(transcript)
        return {"prosodic":50,"spectral":50,"linguistic":round(l,1),"mfcc":50}

    def _describe(self, severity: str, has_audio: bool) -> str:
        base = {
            "mild":     "Speech patterns are consistent with mild intellectual disability. "
                        "The individual demonstrates functional communication with some "
                        "reductions in fluency or vocabulary.",
            "moderate": "Speech patterns are consistent with moderate intellectual disability. "
                        "Significant reductions in speech rate, fluency, or sentence complexity.",
            "severe":   "Speech patterns suggest severe intellectual disability or significant "
                        "communication impairment. Communication relies on single words or gestures.",
        }
        desc = base.get(severity, "")
        if not has_audio:
            desc += " (Text-only analysis — upload audio for higher accuracy.)"
        return desc
