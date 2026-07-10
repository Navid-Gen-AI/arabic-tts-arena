"""SILMA TTS v2 — API-based Arabic text-to-speech (SILMA cloud streaming API).

The endpoint streams raw float32 PCM at 24 kHz; chunks are reassembled on
4-byte sample boundaries before concatenation. Voice, speed, and creativity
are sampled per request.

https://silma.ai/arabic-tts-models
"""

import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

API_URL = "https://api.silma.ai/tts/v2/stream"
API_MODEL_ID = "silma-tts-v2-msa"
VOICE_IDS = ["sarah", "salma", "saja", "sultan", "salim"]
SPEED_CHOICES = [0.4]
CREATIVITY_CHOICES = [0.3, 0.4]

silma_tts_v2_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("requests", "numpy", "soundfile")
    .add_local_python_source(*LOCAL_MODULES)
)

with silma_tts_v2_image.imports():
    import requests
    import numpy as np


@register_model
@app.cls(
    image=silma_tts_v2_image,
    scaledown_window=60,
    secrets=[modal.Secret.from_name("silma-tts-cloud-api")],
)
class SilmaV2TTSModel(BaseTTSModel):
    """SILMA TTS v2 — API-based Arabic text-to-speech."""

    model_id = "silma_tts_v2"
    display_name = "SILMA TTS v2"
    model_url = "https://silma.ai/arabic-tts-models"
    gpu = ""
    open_weight = False
    sample_rate = 24000

    def stream_waveform(self, text, voice_id, creativity, speed):
        import time

        payload = {
            "model_id": API_MODEL_ID,
            "text": text,
            "creativity": creativity,
            "speed": speed,
            "voice_id": voice_id,
        }

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "apiKey": f"{self.api_key}",
        }

        start_time = time.perf_counter()
        carry_over = b""
        first_byte_received = False

        try:
            with self.persistent_session.post(
                API_URL, json=payload, stream=True, headers=headers
            ) as r:
                if r.status_code == 200:
                    for chunk in r.iter_content(chunk_size=None):
                        if chunk:
                            if not first_byte_received:
                                ttft = (time.perf_counter() - start_time) * 1000
                                print(f"⏱️ TTFT: {ttft:.2f} ms")
                                first_byte_received = True

                            # Network chunks can split a float32 mid-sample:
                            # keep only whole 4-byte samples and carry the
                            # remainder into the next chunk.
                            current_data = carry_over + chunk
                            cut_off = (len(current_data) // 4) * 4
                            valid_bytes = current_data[:cut_off]
                            carry_over = current_data[cut_off:]

                            if valid_bytes:
                                yield np.frombuffer(valid_bytes, dtype=np.float32)

                else:
                    print(f"Server rejected the request with status: {r.status_code}")

                    try:
                        error_data = r.json()

                        raise Exception(
                            f"HTTP Error {r.status_code}: {error_data.get('detail')}"
                        )

                    except requests.exceptions.JSONDecodeError:
                        print(f"Error message: '{r.text}'")
                        raise Exception(f"HTTP Error {r.status_code}: {r.text}")

        except Exception as e:
            print(f"Streaming Error: {e}")
            raise Exception(f"Streaming Error {r.status_code}: {r.text}")

    @modal.enter()
    def load_model(self):
        import os

        self.api_key = os.environ["API_KEY"]
        self.persistent_session = requests.Session()  # reuse TCP connections
        print(f"✅ {self.display_name} ready (endpoint: {API_URL})")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        import random

        voice_id = random.choice(VOICE_IDS)
        speed = random.choice(SPEED_CHOICES)
        creativity = random.choice(CREATIVITY_CHOICES)

        try:
            import time
            start = time.perf_counter()

            all_audio_chunks = []
            print("Starting Streaming ...")
            for audio_chunk in self.stream_waveform(text, voice_id, creativity, speed):
                if audio_chunk is not None:
                    all_audio_chunks.append(audio_chunk)

            if len(all_audio_chunks) > 0:
                full_waveform = np.concatenate(all_audio_chunks)

                audio_base64 = self.audio_to_base64(full_waveform, self.sample_rate)

                if audio_base64:
                    return self.success_response(
                        audio_base64, self.sample_rate,
                        inference_seconds=time.perf_counter() - start,
                    )
            else:
                # `raise <dict>` is a TypeError; the except below turns it into
                # the actual error response. Kept as-is from the original.
                raise self.error_response("Empty audio chunks from API response.")

        except Exception as e:
            print(f"Generation error: {e}")
            return self.error_response(e)
