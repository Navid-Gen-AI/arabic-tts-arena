"""Hamsa TTS — API-based Arabic text-to-speech.

No local weights or GPU: synthesize() POSTs to the Hamsa realtime TTS API
(MSA dialect) and decodes the returned audio with soundfile. Each request
randomly picks one of four speaker IDs.

https://tryhamsa.com
"""

import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

API_URL = "https://api.tryhamsa.com/v1/realtime/tts"
SPEAKER_IDS = [
    "43f88a30-1a02-4f0b-8f75-e1b82ff724ef",
    "eee27c16-0381-44ff-a78c-c912f53d8545",
    "9d257806-7034-4dff-a189-c838e74af17f",
    "7dc8dde2-3e3d-4ee7-b92f-d5eec6be0d72",
]

hamsa_tts_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "requests",
        "numpy",
        "soundfile",
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=hamsa_tts_image,
    scaledown_window=60,
    secrets=[modal.Secret.from_name("hamsa-tts-credentials")],
)
class HamsaTTSModel(BaseTTSModel):
    """Hamsa TTS — API-based Arabic text-to-speech (https://tryhamsa.com)."""

    model_id = "hamsa_tts"
    display_name = "Hamsa TTS"
    model_url = "https://tryhamsa.com"
    gpu = ""
    open_weight = False

    @modal.enter()
    def load_model(self):
        import os

        self.api_key = os.environ["API_KEY"]
        self.api_token = os.environ["API_TOKEN"]
        print(f"✅ {self.display_name} ready (endpoint: {API_URL})")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        import requests
        import numpy as np
        import soundfile as sf
        import io
        import random

        try:
            response = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Token {self.api_token}",
                    "Content-Type": "application/json",
                    # The API expects the key as a cookie, not a header.
                    "Cookie": f"api_key={self.api_key}",
                },
                json={
                    "text": text,
                    "speaker": random.choice(SPEAKER_IDS),
                    "dialect": "msa",
                    "mulaw": False,
                },
                timeout=30,
            )
            response.raise_for_status()

            wav_array, sample_rate = sf.read(io.BytesIO(response.content))

            if not isinstance(wav_array, np.ndarray):
                wav_array = np.array(wav_array, dtype=np.float32)

            audio_base64 = self.audio_to_base64(wav_array, sample_rate)
            return self.success_response(audio_base64, sample_rate)

        except Exception as e:
            return self.error_response(e)
