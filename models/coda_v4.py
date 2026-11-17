"""
Coda-V4 — proprietary Arabic TTS by Coda, served over an HTTP API.

Text handling (normalisation, tashkeel) and synthesis both happen server-side,
so this client just posts plain Arabic text and receives 24 kHz mono WAV back,
spoken by the "coda" voice.

The Modal secret `coda-v4-credentials` provides:
    CODA_V4_API_KEY   — Coda-V4 gateway key issued to the arena (required)
    CODA_V4_BASE_URL  — optional, defaults to BASE_URL below

Its own secret, not Coda-V2's `coda-tts-credentials`: the Coda-V4 gateway issues
the arena its own key, and a missing one must fail loudly rather than fall back
to the Coda-V2 key.
"""

import io
import os

import modal

from app import LOCAL_MODULES
from models import BaseTTSModel, arena_cls, register_model

BASE_URL = "https://coda.eracoda.sa/coda-v4"
PUBLIC_MODEL = "Coda-V4"
# The gateway's default voice is "nouf"; the arena entry is the "coda" voice,
# so it is always sent explicitly.
VOICE = "coda"

# The arena serves audio as-is, and output level varies a few dB between
# models. Normalize to the loudness of the current field (~-18 dBFS RMS) so
# blind A/B votes compare voices rather than playback levels — same treatment
# audar_tts_v1 applies.
TARGET_RMS_DBFS = -18.0
PEAK_CAP = 0.95

# ---------------------------------------------------------------------------
# 1. Image — lightweight, no GPU needed for API calls
# ---------------------------------------------------------------------------
coda_v4_image = (
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
@arena_cls(
    image=coda_v4_image,
    # Leaderboard latency is timed around the whole remote call, so container
    # cold starts count against us. Keep the (cheap, CPU-only) container warm
    # a little longer — matches audar_tts_v1 / fish_speech_s2.
    scaledown_window=120,
    retries=0,
    secrets=[modal.Secret.from_name("coda-v4-credentials")],
)
class CodaV4Model(BaseTTSModel):
    """Coda-V4 — proprietary Arabic TTS, served via the Coda API."""

    # ── Required class attributes ──────────────────────────────────────────
    model_id = "coda_v4"
    display_name = "Coda-V4"
    model_url = "https://coda.eracoda.sa"
    gpu = ""            # API-based — the GPU is on Coda's side
    open_weight = False

    # ── Lifecycle ──────────────────────────────────────────────────────────
    @modal.enter()
    def load_model(self):
        self.api_key = os.environ["CODA_V4_API_KEY"]
        self.base_url = (os.environ.get("CODA_V4_BASE_URL") or BASE_URL).rstrip("/")
        self.endpoint = f"{self.base_url}/v1/audio/speech"
        print(f"✅ {self.display_name} ready (endpoint: {self.base_url}, voice: {VOICE})")

    # ── Core method ────────────────────────────────────────────────────────
    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Generate Arabic speech via the Coda gateway."""
        import numpy as np
        import requests
        import soundfile as sf

        text = text.strip()
        if not text:
            raise ValueError("Input text is empty")

        response = requests.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "audio/wav",
            },
            json={
                "model": PUBLIC_MODEL,
                "input": text,
                "voice": VOICE,
                "response_format": "wav",
            },
            timeout=25,  # under the 30 s arena cap, leaves a safety margin
        )
        response.raise_for_status()

        wav_array, sample_rate = sf.read(io.BytesIO(response.content))
        if not isinstance(wav_array, np.ndarray):
            wav_array = np.array(wav_array, dtype=np.float32)
        if wav_array.ndim > 1:
            wav_array = wav_array.mean(axis=1)

        wav_array = wav_array.astype(np.float64)
        rms = float(np.sqrt(np.mean(wav_array**2)))
        if rms < 1e-4:  # ≈ -80 dBFS — fail loudly rather than serve silence
            raise RuntimeError("Gateway returned empty/silent audio")

        # Loudness normalization (see TARGET_RMS_DBFS above).
        wav_array = wav_array * (10 ** (TARGET_RMS_DBFS / 20.0) / rms)
        peak = float(np.max(np.abs(wav_array)))
        if peak > PEAK_CAP:
            wav_array = wav_array * (PEAK_CAP / peak)

        return self.success_response(
            self.audio_to_base64(wav_array, int(sample_rate)), int(sample_rate)
        )
