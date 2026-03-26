
import modal
import os
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES
import random

silma_tts_large_v1_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "requests",
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=silma_tts_large_v1_image,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("silma-tts-cloud-api")],
)

class SilmaLargeTTSModel(BaseTTSModel):
    """
    SILMA TTS — API-based Arabic text-to-speech.

    """

    # ── Required class attributes ──────────────────────────────────────────
    model_id = "silma_tts_v1_large"            # unique, lowercase, underscores
    display_name = "SILMA TTS v1 Large"             # shown in the arena UI
    model_url = "https://silma.ai/arabic-tts-models"            # link to your model/product page
    gpu = ""                                         # empty for API-based models (no GPU)

    # ── Lifecycle ──────────────────────────────────────────────────────────
    @modal.enter()
    def load_model(self):
        """
        Called once when the container starts.

        Read your API credentials from environment variables.
        Modal injects them automatically from the secret you specified above.
        """
        self.api_key = os.environ["API_KEY"]
        self.api_url = os.environ["API_URL"]
        print(f"✅ {self.display_name} ready (endpoint: {self.api_url})")

    # ── Core method ────────────────────────────────────────────────────────
    @modal.method()
    def synthesize(self, text: str) -> dict:
        """
        Generate Arabic speech from text by calling your API.

        Must return:
            self.success_response(audio_base64, sample_rate)  — on success
            self.error_response(error)                        — on failure
        """
        import http.client
        import json

        voice_id = random.choice(["Sulaiman","Salma"])
        speed = random.choice([1.1,1.2,1.3])


        try:
            # ── Call your API ──────────────────────────────────────────────

            conn = http.client.HTTPSConnection(f"{self.api_url}")

            payload = {
                        "model_id": "silma-tts-pro-msa-large",
                        "text": text,
                        "reference_audio_id": voice_id,
                        "nfe_steps": 16,
                        "speaking_speed": speed,
                    }

            headers = {
                'Accept': "application/json",
                'Content-Type': "application/json",
                'apiKey': f"{self.api_key}"
            }

            conn.request("POST", "/tts/generate", json.dumps(payload), headers)

            res = conn.getresponse()

            if res.status >= 400:
                raise Exception(f"HTTP Error {res.status}: {res.reason}")
        
            data = res.read()

            response_dict = json.loads(data.decode("utf-8"))

            ## get the generated audio wav as base64 from the response object
            base64_string = response_dict.get('audio_base64_encoded')


            if base64_string:
                return self.success_response(base64_string, 24000)
            else:
                raise self.error_response("Missing 'audio_base64_encoded' in API response.")

        except Exception as e:
            return self.error_response(e)
