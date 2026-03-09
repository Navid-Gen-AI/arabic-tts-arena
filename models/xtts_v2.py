import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

# XTTS v2 — Coqui's multilingual TTS with Arabic support and voice cloning.
# Uses the `TTS` pip package (coqui-tts fork) for simple inference.
# Ref: https://huggingface.co/tts-hub/XTTS-v2

# Arabic reference audio — reuse the same Spark TTS / Arabic-F5-TTS reference
_REF_AUDIO_REPO = "IbrahimSalah/Arabic-TTS-Spark"

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
    # Pre-download the XTTS v2 model and Arabic reference audio
    .env({"COQUI_TOS_AGREED": "1"})  # Auto-accept Coqui TOS for non-interactive build
    .run_commands(
        "python3 -c \""
        "from TTS.api import TTS; "
        "TTS('tts_models/multilingual/multi-dataset/xtts_v2')"
        "\"",
        # Download Arabic reference audio
        "python3 -c \""
        "from huggingface_hub import hf_hub_download; "
        "hf_hub_download("
        "    repo_id='IbrahimSalah/Arabic-TTS-Spark',"
        "    repo_type='space',"
        "    filename='reference.wav',"
        "    local_dir='/root/xtts-ref'"
        ")\"",
        secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=xtts_v2_image,
    gpu="T4",
    scaledown_window=300,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class XTTSv2Model(BaseTTSModel):
    """XTTS v2 — Coqui multilingual TTS with Arabic voice cloning.

    Supports 17 languages including Arabic. Requires only ~6 seconds of
    reference audio for voice cloning.

    Source: https://huggingface.co/tts-hub/XTTS-v2
    """

    model_id = "xtts_v2"
    display_name = "XTTS v2"
    model_url = "https://huggingface.co/tts-hub/XTTS-v2"

    @modal.enter()
    def load_model(self):
        """Load XTTS v2 model."""
        from TTS.api import TTS

        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
        self.sample_rate = 24000  # XTTS v2 native output rate
        self._ref_audio = "/root/xtts-ref/reference.wav"

        print(f"✅ XTTS v2 loaded on CUDA (sr={self.sample_rate})")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Synthesize Arabic text to speech with voice cloning."""
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

            # Generate speech using the TTS API
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
                return self.error_response(
                    f"Audio too short: {wav.size} samples"
                )

            print(f"[xtts_v2] audio: len={wav.size}, sr={self.sample_rate}")

            audio_base64 = self.audio_to_base64(wav, self.sample_rate)
            return self.success_response(audio_base64, self.sample_rate)

        except Exception as e:
            import traceback
            return self.error_response(f"{e}\n{traceback.format_exc()}")
