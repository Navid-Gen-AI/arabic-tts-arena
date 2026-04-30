import modal
import os
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

# ---------------------------------------------------------------------------
# 1. Image — lightweight, no GPU needed for API calls
# ---------------------------------------------------------------------------
aic_tts_image = (
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
    image=aic_tts_image,
    scaledown_window=60,
    secrets=[modal.Secret.from_name("aic-tts-credentials")],
)
class AICTTSModel(BaseTTSModel):
    """
    AIC TTS — API-based Arabic text-to-speech.
    """

    # ── Required class attributes ──────────────────────────────────────────
    model_id = "aic_tts"
    display_name = "AIC TTS v1"
    model_url = "https://www.aic.gov.eg/"
    gpu = ""
    open_weight = False

    # ── Lifecycle ──────────────────────────────────────────────────────────
    @modal.enter()
    def load_model(self):
        """Read API credentials from environment variables."""
        self.api_token = os.environ["API_TOKEN"]
        self.api_url = "https://arena.aic.gov.eg/v2/models/tts_pipeline/infer"
        self.speakers = [
            "sara",
            "adam",
        ]
        self.adam_tempo = 1.1
        self.sara_tempo = 1.0

        print(f"✅ {self.display_name} ready (endpoint: {self.api_url})")

    # ── Core method ────────────────────────────────────────────────────────
    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Generate Arabic speech from text via the AIC TTS API."""
        import requests
        import numpy as np
        import random
        import json

        speaker = random.choice(self.speakers)

        try:
            payload_data = (
                '{"text": "'
                + text
                + '", "speaker": "'
                + speaker
                + '", "type": "msa", "tempo": '
                + str(self.adam_tempo if speaker == "adam" else self.sara_tempo)
                + ', "token": "'
                + self.api_token
                + '"}'
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

            # Use a new session each time to ensure thread safety and proper connection handling
            with requests.Session() as session:
                response = session.post(self.api_url, headers=headers, data=payload)
            print(f"API response status: {response.status_code}")
            print(f"API response body (first 500 chars): {response.text[:500]}")
            res_data = response.json()

            # Access the first element of the 'outputs' list, then the 'data' key
            audio_samples = res_data["outputs"][0]["data"]

            print(f"Extracted {len(audio_samples)} audio samples. ✅")
            audio_samples = np.array(audio_samples, dtype=np.int16)

            audio_base64 = self.audio_to_base64(audio_samples, 22050)

            return self.success_response(audio_base64, 22050)

        except Exception as e:
            return self.error_response(e)
