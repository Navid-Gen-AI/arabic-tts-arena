import modal
import os
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES
import random
import time


silma_tts_v2_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("requests", "numpy")
    .add_local_python_source(*LOCAL_MODULES)
)

with silma_tts_v2_image.imports():
    import requests
    import numpy as np


@register_model
@app.cls(
    image=silma_tts_v2_image,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("silma-tts-cloud-api")],
)
class SilmaV2TTSModel(BaseTTSModel):
    """
    SILMA TTS v2 — API-based Arabic text-to-speech.

    """

    # ── Required class attributes ──────────────────────────────────────────
    model_id = "silma_tts_v2"  # unique, lowercase, underscores
    display_name = "SILMA TTS v2"  # shown in the arena UI
    model_url = "https://silma.ai/arabic-tts-models"  # link to your model/product page
    gpu = ""  # empty for API-based models (no GPU)
    open_weight = False
    sample_rate = 24000

    def stream_waveform(self, text, voice_id, creativity, speed):
        api_url = self.api_url

        payload = {
            "model_id": "silma-tts-v2-msa",
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
                api_url, json=payload, stream=True, headers=headers
            ) as r:
                if r.status_code == 200:
                    # Iterate over raw bytes as they arrive from the network
                    for chunk in r.iter_content(chunk_size=None):
                        if chunk:
                            if not first_byte_received:
                                ttft = (time.perf_counter() - start_time) * 1000
                                print(f"⏱️ TTFT: {ttft:.2f} ms")
                                first_byte_received = True

                            # Combine carry-over from previous chunk with new data
                            current_data = carry_over + chunk

                            # Calculate how many full 4-byte floats we have
                            num_floats = len(current_data) // 4
                            cut_off = num_floats * 4

                            # Separate valid bytes from the new remainder
                            valid_bytes = current_data[:cut_off]
                            carry_over = current_data[cut_off:]

                            if valid_bytes:
                                # Convert to waveform and yield back to the caller
                                waveform = np.frombuffer(valid_bytes, dtype=np.float32)

                                yield waveform

                ##status code != 200
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

    # ── Lifecycle ──────────────────────────────────────────────────────────
    @modal.enter()
    def load_model(self):
        """
        Called once when the container starts.

        Read your API credentials from environment variables.
        Modal injects them automatically from the secret you specified above.
        """
        self.api_key = os.environ["API_KEY"]
        self.api_url = (
            "https://api.silma.ai/tts/v2/stream"  # hardcoded endpoint for streaming TTS
        )
        self.persistent_session = (
            requests.Session()
        )  # reuse TCP connections for efficiency
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

        voice_id = random.choice(["sarah", "salma", "saja", "sultan", "salim"])
        speed = random.choice([0.4, 0.5, 0.6])
        creativity = random.choice([0.2, 0.3, 0.4, 0.5])

        try:
            # ── Call your API ──────────────────────────────────────────────

            all_audio_chunks = []
            print("Starting Streaming ...")
            for audio_chunk in self.stream_waveform(text, voice_id, creativity, speed):
                if audio_chunk is not None:
                    all_audio_chunks.append(audio_chunk)

            if len(all_audio_chunks) > 0:
                full_waveform = np.concatenate(all_audio_chunks)

                audio_base64 = self.audio_to_base64(full_waveform, self.sample_rate)

                if audio_base64:
                    return self.success_response(audio_base64, self.sample_rate)
            else:
                raise self.error_response("Empty audio chunks from API response.")

        except Exception as e:
            print(f"Generation error: {e}")
            return self.error_response(e)
