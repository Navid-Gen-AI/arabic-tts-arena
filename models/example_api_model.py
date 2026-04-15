"""
Example: How to add a closed-source / API-based TTS model to the arena.

This file is a template for companies that want to add their proprietary
TTS model via an API. Copy this file, rename it, and adapt it.

=== Setup ===

1. Copy this file:     cp example_api_model.py  your_company_tts.py
2. Edit the class below with your API details.
3. Open a PR with your model file.
4. After the PR is merged, DM the maintainer of the repo your API credentials.
    > modal secret create example-company-tts API_KEY=<your_api_key> API_URL=<your_api_url>
   Your keys are stored in an encrypted vault (never in git).

=== That's it. ===
"""

import modal
import os
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

# ---------------------------------------------------------------------------
# 1. Define your image — lightweight, no GPU needed for API calls
# ---------------------------------------------------------------------------
example_api_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "requests",
        "numpy",
        "soundfile",
    )
    .add_local_python_source(*LOCAL_MODULES)
)

# ---------------------------------------------------------------------------
# 2. Register your model
# ---------------------------------------------------------------------------
@register_model
@app.cls(
    image=example_api_image,
    # No GPU needed — we're just calling an API
    scaledown_window=60,
    # ⬇️ This is your Modal secret name — must match what the maintainer creates
    secrets=[modal.Secret.from_name("example-company-tts")],
)
class ExampleAPIModel(BaseTTSModel):
    """
    Example Company TTS — API-based Arabic text-to-speech.

    Replace this docstring with a short description of your model.
    """

    # ── Required class attributes ──────────────────────────────────────────
    model_id = "example_api"                         # unique, lowercase, underscores
    display_name = "Example Company TTS"             # shown in the arena UI
    model_url = "https://example.com/tts"            # link to your model/product page
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
        import requests
        import numpy as np
        import io

        try:
            # ── Call your API ──────────────────────────────────────────────
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "language": "ar",
                    # Add any other parameters your API expects:
                    # "voice": "default",
                    # "speed": 1.0,
                },
                timeout=30,
            )
            response.raise_for_status()

            # ── Parse the response ─────────────────────────────────────────
            #
            # Adapt this section to match your API's response format.
            # Common patterns:
            #
            #   A) API returns raw audio bytes (WAV/MP3):
            #      audio_bytes = response.content
            #
            #   B) API returns JSON with base64 audio:
            #      audio_base64 = response.json()["audio"]
            #
            #   C) API returns JSON with a download URL:
            #      audio_url = response.json()["audio_url"]
            #      audio_bytes = requests.get(audio_url).content

            # --- Example: API returns raw WAV bytes ---
            import soundfile as sf

            audio_bytes = response.content
            wav_array, sample_rate = sf.read(io.BytesIO(audio_bytes))

            if not isinstance(wav_array, np.ndarray):
                wav_array = np.array(wav_array)

            audio_base64 = self.audio_to_base64(wav_array, sample_rate)
            return self.success_response(audio_base64, sample_rate)

        except Exception as e:
            return self.error_response(e)
