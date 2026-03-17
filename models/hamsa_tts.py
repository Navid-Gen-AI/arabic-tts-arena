import modal
import os
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

# ---------------------------------------------------------------------------
# 1. Image — lightweight, no GPU needed for API calls
# ---------------------------------------------------------------------------
hamsa_tts_image = (
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
    image=hamsa_tts_image,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("hamsa-tts-credentials")],
)
class HamsaTTSModel(BaseTTSModel):
    """
    Hamsa TTS — API-based Arabic text-to-speech.

    Uses the Hamsa realtime TTS API.
    https://tryhamsa.com
    """

    # ── Required class attributes ──────────────────────────────────────────
    model_id = "hamsa_tts"
    display_name = "Hamsa TTS"
    model_url = "https://tryhamsa.com"
    gpu = ""  # API-based, no GPU needed

    # ── Lifecycle ──────────────────────────────────────────────────────────
    @modal.enter()
    def load_model(self):
        """Read API credentials from environment variables."""
        self.api_key = os.environ["API_KEY"]
        self.api_token = os.environ["API_TOKEN"]
        self.api_url = "https://api.tryhamsa.com/v1/realtime/tts"
        self.speakers_ids = [
            "43f88a30-1a02-4f0b-8f75-e1b82ff724ef",
            "eee27c16-0381-44ff-a78c-c912f53d8545",
            "9d257806-7034-4dff-a189-c838e74af17f",
            "7dc8dde2-3e3d-4ee7-b92f-d5eec6be0d72",

        ]
        print(f"✅ {self.display_name} ready (endpoint: {self.api_url})")

    # ── Core method ────────────────────────────────────────────────────────
    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Generate Arabic speech from text via the Hamsa TTS API."""
        import requests
        import numpy as np
        import soundfile as sf
        import io
        import random

        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Token {self.api_token}",
                    "Content-Type": "application/json",
                    "Cookie": f"api_key={self.api_key}",
                },
                json={
                    "text": text,
                    "speaker": random.choice(self.speakers_ids),
                    "dialect": "msa",
                    "mulaw": False,
                },
                timeout=30,
            )
            response.raise_for_status()

            # Parse raw audio bytes from the response
            audio_bytes = response.content
            wav_array, sample_rate = sf.read(io.BytesIO(audio_bytes))

            if not isinstance(wav_array, np.ndarray):
                wav_array = np.array(wav_array, dtype=np.float32)

            audio_base64 = self.audio_to_base64(wav_array, sample_rate)
            return self.success_response(audio_base64, sample_rate)

        except Exception as e:
            return self.error_response(e)
