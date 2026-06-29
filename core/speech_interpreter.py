"""
core/speech_interpreter.py
Uses local Ollama (free) to interpret speech and generate clinical output.
"""
import json, logging, urllib.request, urllib.error
from typing import Optional, Dict, Any
from core.feature_extractor import AudioFeatures

logger = logging.getLogger(__name__)

def _cfg(config, key, default):
    if isinstance(config, dict): return config.get(key, default)
    return getattr(config, key, default)

class SpeechInterpreter:
    def __init__(self, config):
        self.config      = config
        self.ollama_url  = _cfg(config, "OLLAMA_URL",   "http://localhost:11434")
        self.model       = _cfg(config, "OLLAMA_MODEL", "llama3")

    def interpret(self, raw_text: str, features: Optional[AudioFeatures],
                  severity: Dict[str, Any], age_group: str = "adult") -> Dict[str, Any]:
        self._check_ollama()
        prompt   = self._build_prompt(raw_text, features, severity, age_group)
        response = self._call_ollama(prompt)
        return self._parse(response, raw_text)

    def _check_ollama(self):
        try:
            req = urllib.request.Request(f"{self.ollama_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception:
            raise RuntimeError(
                f"Ollama is not running. Fix:\n"
                f"  1. Open a new terminal\n"
                f"  2. Run: ollama serve\n"
                f"  3. Run: ollama pull {self.model}"
            )

    def _call_ollama(self, prompt: str) -> str:
        url     = f"{self.ollama_url}/api/generate"
        payload = json.dumps({
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.1, "num_predict": 1000},
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())["response"]
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama call failed: {e}")

    def _build_prompt(self, raw_text, features, severity, age_group) -> str:
        sev  = severity.get("severity","unknown").upper()
        desc = severity.get("description","")
        flags= severity.get("clinical_flags",[])
        flag_text = "\n".join(f"- {f['observation']}" for f in flags[:4]) or "None"

        feat_text = ""
        if features and features.speech_rate_syl_per_sec > 0:
            feat_text = (
                f"Speech rate: {features.speech_rate_syl_per_sec:.1f} syl/s  |  "
                f"Pause ratio: {features.pause_ratio:.0%}  |  "
                f"Voiced ratio: {features.voiced_ratio:.0%}  |  "
                f"Pitch range: {features.pitch_range_hz:.0f}Hz"
            )

        return f"""You are a clinical speech-language pathologist AI specialising in intellectual disabilities.

Severity: {sev}
Description: {desc}
Acoustic features: {feat_text or 'text-only input'}
Clinical flags:
{flag_text}
Age group: {age_group}
Raw speech: "{raw_text}"

Respond ONLY with a JSON object — no markdown, no explanation, just JSON:
{{
  "interpreted_meaning": "Clear grammatical sentence of what the person intended to say.",
  "emotional_tone": "happy|frustrated|neutral|confused|urgent|distressed|excited",
  "topic_category": "basic needs|social interaction|emotional expression|information seeking|protest/refusal|greeting|other",
  "speech_patterns": [
    {{"pattern": "Pattern name", "description": "Clinical description", "severity_indicator": true}}
  ],
  "metrics": {{
    "vocabulary_complexity": 50,
    "articulation_score": 50,
    "communication_intent_clarity": 60
  }},
  "communication_suggestions": [
    {{"suggestion": "Actionable suggestion for caregivers/therapists", "priority": "high|medium|low"}}
  ],
  "next_steps": "2-3 sentences on evaluations, interventions, and support."
}}
Output ONLY the JSON. Nothing before or after."""

    def _parse(self, raw: str, fallback: str) -> Dict[str, Any]:
        text = raw.strip()
        if "```" in text:
            for part in text.split("```"):
                part = part.strip().lstrip("json").strip()
                if part.startswith("{"): text = part; break
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            text = text[s:e+1]
        try:
            r = json.loads(text)
            r.setdefault("interpreted_meaning", fallback)
            r.setdefault("emotional_tone", "neutral")
            r.setdefault("topic_category", "other")
            r.setdefault("speech_patterns", [])
            r.setdefault("metrics", {"vocabulary_complexity":50,"articulation_score":50,"communication_intent_clarity":50})
            r.setdefault("communication_suggestions", [])
            r.setdefault("next_steps", "Consult a qualified speech-language pathologist for full evaluation.")
            return r
        except Exception:
            logger.error("JSON parse failed. Raw: %s", raw[:300])
            return {
                "interpreted_meaning": fallback,
                "emotional_tone": "neutral", "topic_category": "other",
                "speech_patterns": [{"pattern":"Parse error","description":"Try a larger Ollama model: ollama pull llama3","severity_indicator":False}],
                "metrics": {"vocabulary_complexity":0,"articulation_score":0,"communication_intent_clarity":0},
                "communication_suggestions": [{"suggestion":"Ensure Ollama llama3 model is installed and running.","priority":"high"}],
                "next_steps": "Restart Flask and ensure ollama serve is running.",
            }
