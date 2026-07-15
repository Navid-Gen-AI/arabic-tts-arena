"""
Audar TTS — Arabic-first expressive TTS by AudarAI.

Serves Audar-TTS-V1 (Pro tier) via the AudarAI speech API. The smaller family
members, Audar-TTS-V1-Flash and Audar-TTS-V1-Turbo, are available as open
weights: https://huggingface.co/audarai

The Modal secret `audar-tts-credentials` only needs to provide AUDAR_API_KEY.
Each request picks one of the configured voices at random.
"""

import modal
import os
import random
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

BASE_URL = "https://prod.audarai.com/apiv2"
TTS_MODEL = "audar-tts-v1-pro"
VOICES = ["Rashid", "Saeed"]

# The arena serves audio as-is (no loudness normalization), and output level
# varies a few dB between catalog voices. Normalize to the typical loudness of
# current arena entries (measured median ≈ -18 dBFS RMS across models) so
# blind A/B votes compare voices rather than playback levels.
TARGET_RMS_DBFS = -18.0
PEAK_CAP = 0.95

# ---------------------------------------------------------------------------
# 1. Image — lightweight, no GPU needed for API calls
# ---------------------------------------------------------------------------
audar_tts_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "requests",
        "numpy",
        "soundfile",
    )
    .add_local_python_source(*LOCAL_MODULES)
)


# ---------------------------------------------------------------------------
# 2. Register the model
# ---------------------------------------------------------------------------
@register_model
@app.cls(
    image=audar_tts_image,
    # Leaderboard latency is timed around the whole remote call, so container
    # cold-starts count against us. Keep the (cheap, CPU-only) container warm
    # a bit longer — matches the precedent set for fish_speech_s2 upstream.
    scaledown_window=120,
    secrets=[modal.Secret.from_name("audar-tts-credentials")],
)
class AudarTTSModel(BaseTTSModel):
    """Audar-TTS-V1 — Arabic-first expressive TTS served via the AudarAI API."""

    # ── Required class attributes ──────────────────────────────────────────
    model_id = "audar_tts_v1"
    display_name = "Audar-TTS-V1-Pro"
    model_url = "https://huggingface.co/audarai"
    gpu = ""
    open_weight = False

    # ── Lifecycle ──────────────────────────────────────────────────────────
    @modal.enter()
    def load_model(self):
        """Read the API key from the Modal secret (env overrides optional)."""
        self.api_key = os.environ["AUDAR_API_KEY"]
        self.base_url = os.environ.get("AUDAR_BASE_URL", BASE_URL).rstrip("/")
        self.tts_model = os.environ.get("AUDAR_MODEL", TTS_MODEL)
        voices = os.environ.get("AUDAR_VOICE", ",".join(VOICES))
        self.voices = [v.strip() for v in voices.split(",") if v.strip()]
        # Model selection is routed via the `provider` query parameter (the
        # `model` field in the JSON body is ignored by the gateway).
        self.endpoint = f"{self.base_url}/v1/speech/audio/speech?provider={self.tts_model}"
        print(
            f"✅ {self.display_name} ready "
            f"(model={self.tts_model}, voices={self.voices})"
        )

    # ── Core method ────────────────────────────────────────────────────────
    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Generate Arabic speech from text via the AudarAI API."""
        import io
        import requests
        import numpy as np
        import soundfile as sf

        try:
            response = requests.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "audio/wav",
                },
                json={
                    "text": text,
                    "voice": random.choice(self.voices),
                    "model": self.tts_model,
                    "response_format": "wav",
                    "speed": 1.0,
                },
                timeout=25,  # under the 30 s arena cap, leaves a safety margin
            )
            response.raise_for_status()

            # The API returns WAV bytes; decode to numpy + sample_rate so the
            # BaseTTSModel helper can re-emit normalized WAV.
            wav_array, sample_rate = sf.read(io.BytesIO(response.content))
            if not isinstance(wav_array, np.ndarray):
                wav_array = np.array(wav_array, dtype=np.float32)
            # Downmix to mono if the API ever returns stereo
            if wav_array.ndim > 1:
                wav_array = wav_array.mean(axis=1)

            # Guard: the API returns HTTP 200 with near-silent audio for
            # unknown voice names — fail loudly instead of serving silence.
            wav_array = wav_array.astype(np.float64)
            rms = float(np.sqrt(np.mean(wav_array**2)))
            if rms < 1e-4:  # ≈ -80 dBFS
                return self.error_response("API returned empty/silent audio")

            # Loudness normalization (see TARGET_RMS_DBFS note above).
            wav_array = wav_array * (10 ** (TARGET_RMS_DBFS / 20.0) / rms)
            peak = float(np.max(np.abs(wav_array)))
            if peak > PEAK_CAP:
                wav_array = wav_array * (PEAK_CAP / peak)

            audio_base64 = self.audio_to_base64(wav_array, int(sample_rate))
            return self.success_response(audio_base64, int(sample_rate))

        except Exception as e:
            return self.error_response(e)
