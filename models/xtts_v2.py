"""XTTS v2 — Coqui's multilingual TTS with Arabic support and voice cloning.

Uses the `TTS` pip package (coqui-tts fork) for simple inference; clones a
fixed Arabic reference voice (~6 s clip is enough for XTTS cloning).

Model: https://huggingface.co/tts-hub/XTTS-v2
"""

import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# Arabic reference audio — reuse the same Spark TTS / Arabic-F5-TTS reference.
REF_AUDIO_REPO = "IbrahimSalah/Arabic-TTS-Spark"
REF_AUDIO_DIR = "/root/xtts-ref"
REF_AUDIO_PATH = f"{REF_AUDIO_DIR}/reference.wav"

SAMPLE_RATE = 24000  # XTTS v2 native output rate


xtts_v2_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "libsndfile1", "espeak-ng")
    .uv_pip_install(
        "torch>=2.0.0,<2.9",
        "torchaudio>=2.0.0,<2.9",
        "numpy",
        "soundfile",
        "huggingface_hub",
        "coqui-tts",
        "transformers<=4.46.2",  # coqui-tts requires <=4.46.2; 5.x removed isin_mps_friendly
    )
    .env({"COQUI_TOS_AGREED": "1"})  # auto-accept Coqui TOS for non-interactive build
    # Model + Arabic reference clip. Must be `python3 -c` shell commands, not
    # .run_function() (which imports this module in a bare build container
    # where the local app/models sources don't exist yet). Command strings
    # match the originally-deployed image byte-for-byte so Modal's cache
    # reuses it.
    .run_commands(
        "python3 -c \""
        "from TTS.api import TTS; "
        f"TTS('{XTTS_MODEL_NAME}')"
        "\"",
        "python3 -c \""
        "from huggingface_hub import hf_hub_download; "
        "hf_hub_download("
        f"    repo_id='{REF_AUDIO_REPO}',"
        "    repo_type='space',"
        "    filename='reference.wav',"
        f"    local_dir='{REF_AUDIO_DIR}'"
        ")\"",
        secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=xtts_v2_image,
    gpu="T4",
    scaledown_window=120,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class XTTSv2Model(BaseTTSModel):
    """XTTS v2 — Coqui multilingual TTS with Arabic voice cloning."""

    model_id = "xtts_v2"
    display_name = "XTTS v2"
    model_url = "https://huggingface.co/tts-hub/XTTS-v2"
    gpu = "T4"

    @modal.enter()
    def load_model(self):
        from TTS.api import TTS

        self.tts = TTS(XTTS_MODEL_NAME).to("cuda")
        self.sample_rate = SAMPLE_RATE
        self._ref_audio = REF_AUDIO_PATH

        print(f"✅ XTTS v2 loaded on CUDA (sr={self.sample_rate})")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        try:
            import numpy as np
            import os

            if not os.path.exists(self._ref_audio):
                return self.error_response(
                    f"Reference audio not found: {self._ref_audio}"
                )

            text = text.strip()
            if not text:
                return self.error_response("Input text is empty")

            print(f"[xtts_v2] text: {text[:80]}")
            import time
            start = time.perf_counter()

            wav = self.tts.tts(
                text=text,
                speaker_wav=self._ref_audio,
                language="ar",
            )

            if not isinstance(wav, np.ndarray):
                wav = np.array(wav, dtype=np.float32)
            wav = wav.astype(np.float32, copy=False)

            if wav.ndim > 1:
                wav = wav.reshape(-1)

            if wav.size < 100:
                return self.error_response(f"Audio too short: {wav.size} samples")

            return self.success_response(
                self.audio_to_base64(wav, self.sample_rate), self.sample_rate,
                inference_seconds=time.perf_counter() - start,
            )

        except Exception as e:
            import traceback
            return self.error_response(f"{e}\n{traceback.format_exc()}")
