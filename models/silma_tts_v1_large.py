"""
Example: How to add a closed-source / API-based TTS model to the arena.

This file is a template for companies that want to add their proprietary
TTS model via an API. Copy this file, rename it, and adapt it.

=== Setup ===

1. Copy this file:     cp example_api_model.py  your_company_tts.py
2. Edit the class below with your API details.
3. Open a PR with your model file.
4. After the PR is merged, DM the maintainer of the repo your API
   credentials. They will run ONE command to store them securely:

       modal secret create example-company-tts API_KEY=sk-xxx API_URL=https://...

   Your keys are stored in Modal's encrypted vault — never in git.

=== That's it. ===
"""

import modal
import os
from models import BaseTTSModel, register_model
from app import app

# ---------------------------------------------------------------------------
# 1. Define your image — lightweight, no GPU needed for API calls
# ---------------------------------------------------------------------------
silma_tts_pro_msa_large_api_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "requests",
    )
)


# ---------------------------------------------------------------------------
# 2. Register your model
# ---------------------------------------------------------------------------

@register_model
@app.cls(
    image=silma_tts_pro_msa_large_api_image,
    # No GPU needed — we're just calling an API
    scaledown_window=300,
    # ⬇️ This is your Modal secret name — must match what the maintainer creates
    secrets=[modal.Secret.from_name("silma-tts-cloud-api")],
)

class SILMATTS_APIModel(BaseTTSModel):
    """
    SILMA TTS — API-based Arabic text-to-speech.

    """

    # ── Required class attributes ──────────────────────────────────────────
    model_id = "silma_tts_pro_msa_large"                         # unique, lowercase, underscores
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

        try:
            # ── Call your API ──────────────────────────────────────────────

            conn = http.client.HTTPSConnection(f"{self.api_url}")

            payload = {
                        "model_id": "silma-tts-pro-msa-large",
                        "text": text,
                        "reference_audio_id": "Sulaiman",
                        "nfe_steps": 16
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
