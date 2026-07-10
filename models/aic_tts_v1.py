"""AIC TTS v1 — API-based Arabic TTS from Egypt's Applied Innovation Center.

No local weights or GPU: synthesize() POSTs to the AIC arena inference
endpoint (Triton-style infer payload) and decodes int16 PCM at 22.05 kHz.
Each request randomly picks the "sara" or "adam" voice.

https://www.aic.gov.eg/
"""

import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

API_URL = "https://arena.aic.gov.eg/v2/models/tts_pipeline/infer"
SPEAKERS = ["sara", "adam"]
TEMPO = {"adam": 1.1, "sara": 1.0}
SAMPLE_RATE = 22050

aic_tts_image = (
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
    image=aic_tts_image,
    scaledown_window=60,
    secrets=[modal.Secret.from_name("aic-tts-credentials")],
)
class AICTTSModel(BaseTTSModel):
    """AIC TTS — API-based Arabic text-to-speech."""

    model_id = "aic_tts"
    display_name = "AIC TTS v1"
    model_url = "https://www.aic.gov.eg/"
    gpu = ""
    open_weight = False

    @modal.enter()
    def load_model(self):
        import os

        self.api_token = os.environ["API_TOKEN"]
        print(f"✅ {self.display_name} ready (endpoint: {API_URL})")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        import requests
        import numpy as np
        import random
        import json

        speaker = random.choice(SPEAKERS)

        try:
            import time
            start = time.perf_counter()

            # The endpoint expects its request JSON embedded as a raw string;
            # `text` is interpolated unescaped, matching the vendor's example.
            payload_data = (
                f'{{"text": "{text}", "speaker": "{speaker}", "type": "msa", '
                f'"tempo": {TEMPO[speaker]}, "token": "{self.api_token}"}}'
            )

            payload = json.dumps(
                {
                    "inputs": [
                        {
                            "name": "TEXT",
                            "shape": [1],
                            "datatype": "BYTES",
                            "data": [payload_data],
                        }
                    ],
                    "outputs": [{"name": "AUDIO"}],
                }
            )

            headers = {"Content-Type": "application/json"}

            # New session per call for thread safety and clean connection handling.
            with requests.Session() as session:
                response = session.post(API_URL, headers=headers, data=payload)
            print(f"API response status: {response.status_code}")
            print(f"API response body (first 500 chars): {response.text[:500]}")
            res_data = response.json()

            audio_samples = res_data["outputs"][0]["data"]

            print(f"Extracted {len(audio_samples)} audio samples. ✅")
            audio_samples = np.array(audio_samples, dtype=np.int16)

            audio_base64 = self.audio_to_base64(audio_samples, SAMPLE_RATE)

            return self.success_response(
                audio_base64, SAMPLE_RATE,
                inference_seconds=time.perf_counter() - start,
            )

        except Exception as e:
            return self.error_response(e)
