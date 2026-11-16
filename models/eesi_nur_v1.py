"""
Nur-V1 — Arabic text-to-speech by EESI Labs Inc., served over the EESI API.

The client posts plain Arabic text and receives WAV back in Nur's own voice.

The Modal secret `eesi-tts-credentials` provides:
    EESI_API_KEY   — API key issued to the arena
    EESI_BASE_URL  — optional, defaults to BASE_URL below
"""

import io
import os

import modal

from app import LOCAL_MODULES
from models import BaseTTSModel, arena_cls, register_model

BASE_URL = "https://api.eesi.ai"
PUBLIC_MODEL = "nur-v1"
VOICE = "ev_a68b5c50"  # Nur

# Output level differs between models; normalize to the field's loudness
# (~-18 dBFS RMS) so votes compare voices, not playback levels — the same
# treatment coda_v2 and audar_tts_v1 apply.
TARGET_RMS_DBFS = -18.0
PEAK_CAP = 0.95

eesi_nur_v1_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "requests",
        "numpy",
        "soundfile",
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@arena_cls(
    image=eesi_nur_v1_image,
    # Container cold starts count against latency; keep the CPU-only
    # container warm a little longer, as coda_v2 and audar_tts_v1 do.
    scaledown_window=120,
    retries=0,
    secrets=[modal.Secret.from_name("eesi-tts-credentials")],
)
class EesiNurV1Model(BaseTTSModel):
    """Nur-V1 — Arabic text-to-speech by EESI Labs Inc., via the EESI API."""

    model_id = "eesi_nur_v1"
    display_name = "Nur-V1 (EESI Labs Inc.)"
    model_url = "https://eesi.ai"
    gpu = ""  # API-based
    open_weight = False

    @modal.enter()
    def load_model(self):
        self.api_key = os.environ["EESI_API_KEY"]
        base_url = os.environ.get("EESI_BASE_URL", BASE_URL).rstrip("/")
        self.endpoint = f"{base_url}/v1/audio/speech"
        print(f"✅ {self.display_name} ready (endpoint: {base_url})")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Generate Arabic speech via the EESI API."""
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
            },
            json={
                "model": PUBLIC_MODEL,
                "voice": VOICE,
                "input": text,
                "language": "ar",
                "response_format": "wav",
            },
            timeout=25,  # under the 30 s arena cap
        )
        response.raise_for_status()

        wav_array, sample_rate = sf.read(io.BytesIO(response.content), dtype="float64")
        if wav_array.ndim > 1:
            wav_array = wav_array.mean(axis=1)

        rms = float(np.sqrt(np.mean(wav_array**2)))
        if rms < 1e-4:  # ≈ -80 dBFS — fail loudly rather than serve silence
            raise RuntimeError("EESI API returned empty/silent audio")

        wav_array = wav_array * (10 ** (TARGET_RMS_DBFS / 20.0) / rms)
        peak = float(np.max(np.abs(wav_array)))
        if peak > PEAK_CAP:
            wav_array = wav_array * (PEAK_CAP / peak)

        return self.success_response(
            self.audio_to_base64(wav_array, int(sample_rate)), int(sample_rate)
        )
