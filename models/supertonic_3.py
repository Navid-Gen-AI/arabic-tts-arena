"""Supertonic 3 — lightweight ONNX multilingual TTS with Arabic support.

Runs on CPU via the `supertonic` pip package; uses the built-in "M4" male
voice style.

Model: https://huggingface.co/Supertone/supertonic-3
"""

import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

VOICE_NAME = "M4"


supertonic_3_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(
        "supertonic",
        "numpy",
        "soundfile",
        "huggingface_hub",
    )
    # ONNX assets + M4 voice style. Must be a `python3 -c` shell command, not
    # .run_function() (which imports this module in a bare build container
    # where the local app/models sources don't exist yet). Command string
    # matches the originally-deployed image byte-for-byte so Modal's cache
    # reuses it.
    .run_commands(
        "python3 -c \""
        "from supertonic import TTS; "
        "tts = TTS(auto_download=True); "
        f"tts.get_voice_style(voice_name='{VOICE_NAME}')"
        "\"",
        secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=supertonic_3_image,
    scaledown_window=60,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    timeout=300,
)
class Supertonic3Model(BaseTTSModel):
    """Supertonic 3 -- lightweight ONNX multilingual TTS with Arabic support."""

    model_id = "supertonic_3"
    display_name = "Supertonic 3"
    model_url = "https://huggingface.co/Supertone/supertonic-3"
    gpu = "CPU"

    @modal.enter()
    def load_model(self):
        from supertonic import TTS

        self.tts = TTS(auto_download=True)
        self.voice_style = self.tts.get_voice_style(voice_name=VOICE_NAME)
        runtime = getattr(self.tts, "tts", None)
        self.sample_rate = int(
            getattr(self.tts, "sample_rate", None)
            or getattr(runtime, "sample_rate", 24000)
        )
        print(f"Supertonic 3 loaded (voice={VOICE_NAME}, lang=ar, sr={self.sample_rate})")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        try:
            import numpy as np

            text = text.strip()
            if not text:
                return self.error_response("Input text is empty")

            wav, _duration = self.tts.synthesize(
                text,
                voice_style=self.voice_style,
                lang="ar",
            )
            wav = np.asarray(wav, dtype=np.float32)
            duration = np.asarray(_duration).reshape(-1)
            if wav.ndim > 1:
                wav = wav[0]
            # Trim to the model-reported duration — raw output is padded.
            if duration.size:
                expected_samples = int(self.sample_rate * float(duration[0]))
                wav = wav[:expected_samples]

            if wav.size < 100:
                return self.error_response(f"Audio too short: {wav.size} samples")

            audio_base64 = self.audio_to_base64(wav, self.sample_rate)
            return self.success_response(audio_base64, self.sample_rate)

        except Exception as e:
            import traceback
            return self.error_response(f"{e}\n{traceback.format_exc()}")
