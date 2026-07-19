"""Grok Voice TTS 1.0 — xAI's expressive multilingual text-to-speech, served
via the OpenRouter audio API (no local weights or GPU).

Model: https://openrouter.ai/x-ai/grok-voice-tts-1.0
Docs:  https://docs.x.ai/developers/model-capabilities/audio/text-to-speech
"""

import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

API_URL = "https://openrouter.ai/api/v1/audio/speech"
MODEL_SLUG = "x-ai/grok-voice-tts-1.0"
VOICE = "sal"  # smooth/balanced — the standard narration register

XAI_OPTIONS = {"language": "auto", "text_normalization": True}

grok_voice_tts_image = (
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
    image=grok_voice_tts_image,
    scaledown_window=60,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class GrokVoiceTTSModel(BaseTTSModel):
    """Grok Voice TTS 1.0 — xAI TTS via OpenRouter."""

    model_id = "grok_voice_tts"
    display_name = "Grok Voice TTS 1.0"
    model_url = "https://openrouter.ai/x-ai/grok-voice-tts-1.0"
    gpu = ""
    open_weight = False

    @modal.enter()
    def load_model(self):
        import os

        self.api_key = os.environ["OPENROUTER_API_KEY"]
        print(f"✅ {self.display_name} ready (model: {MODEL_SLUG}, voice: {VOICE})")

    def _request(self, text: str, with_options: bool):
        import requests

        payload = {
            "model": MODEL_SLUG,
            "input": text,
            "voice": VOICE,
            "response_format": "pcm",  # lossless; rate arrives in Content-Type
        }
        if with_options:
            payload["provider"] = {"options": {"xai": XAI_OPTIONS}}
        return requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=60,
        )

    @modal.method()
    def synthesize(self, text: str) -> dict:
        try:
            import re
            import time
            import numpy as np

            text = (text or "").strip()
            if not text:
                return self.error_response("Input text is empty")

            start = time.perf_counter()

            response = self._request(text, with_options=True)
            if response.status_code != 200:
                print(f"⚠️ options request failed ({response.status_code}: "
                      f"{response.text[:200]}), retrying without provider options")
                response = self._request(text, with_options=False)
            if response.status_code != 200:
                return self.error_response(
                    f"OpenRouter HTTP {response.status_code}: {response.text[:300]}"
                )

            # PCM response: 16-bit mono, sample rate in the Content-Type header
            # (e.g. "audio/pcm;rate=24000;channels=1").
            rate_match = re.search(r"rate=(\d+)", response.headers.get("content-type", ""))
            sample_rate = int(rate_match.group(1)) if rate_match else 24000
            wav = np.frombuffer(response.content, dtype=np.int16).astype(np.float32) / 32768.0

            if wav.size < 100:
                return self.error_response(f"Audio too short: {wav.size} samples")

            return self.success_response(
                self.audio_to_base64(wav, sample_rate), sample_rate,
                inference_seconds=time.perf_counter() - start,
            )

        except Exception as e:
            import traceback
            return self.error_response(f"{e}\n{traceback.format_exc()}")
