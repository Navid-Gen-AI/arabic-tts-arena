"""
VoxCPM2 — Tokenizer-Free Multilingual TTS with Voice Design
=============================================================

OpenBMB's VoxCPM2: a 2B-parameter, tokenizer-free diffusion autoregressive
TTS model supporting 30 languages (including Arabic) with 48 kHz output.

Key features:
    - Native Arabic support — no language tag needed
    - Voice Design — generates a natural Arabic voice from a text description
      alone, no reference audio required
    - 48 kHz studio-quality output via AudioVAE V2 built-in super-resolution
    - ~8 GB VRAM, RTF ~0.3 on RTX 4090

Model:   https://huggingface.co/openbmb/VoxCPM2
GitHub:  https://github.com/OpenBMB/VoxCPM
Paper:   https://arxiv.org/abs/2509.24650
License: Apache-2.0
"""

import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

# ---------------------------------------------------------------------------
# Modal image
# ---------------------------------------------------------------------------

voxcpm2_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(
        "torch>=2.5.0",
        "torchaudio>=2.5.0",
        "numpy",
        "soundfile",
        "huggingface_hub",
        "voxcpm",
    )
    # Pre-download VoxCPM2 weights so they're baked into the image
    .run_commands(
        "python3 -c \""
        "from voxcpm import VoxCPM; "
        "model = VoxCPM.from_pretrained("
        "    'openbmb/VoxCPM2',"
        "    load_denoiser=False,"
        "    optimize=False,"
        "); "
        "print(f'VoxCPM2 downloaded, sr={model.tts_model.sample_rate}')"
        "\"",
        secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
        gpu="A10G",
    )
    .add_local_python_source(*LOCAL_MODULES)
)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

@register_model
@app.cls(
    image=voxcpm2_image,
    gpu="A10G",
    scaledown_window=60,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class VoxCPM2Model(BaseTTSModel):
    """VoxCPM2 — OpenBMB tokenizer-free multilingual TTS (2B params).

    Uses Voice Design mode to synthesize Arabic text with a natural Arabic
    voice, avoiding the need for a reference audio clip.

    Source: https://huggingface.co/openbmb/VoxCPM2
    """

    model_id = "voxcpm2"
    display_name = "VoxCPM 2"
    model_url = "https://huggingface.co/openbmb/VoxCPM2"
    gpu = "A10G"

    @modal.enter()
    def load_model(self):
        """Load VoxCPM2 when the container starts."""
        from voxcpm import VoxCPM

        self.model = VoxCPM.from_pretrained(
            "openbmb/VoxCPM2",
            load_denoiser=False,
            optimize=False,
        )
        self.sample_rate = self.model.tts_model.sample_rate  # 48000

        print(f"✅ VoxCPM2 loaded on CUDA (sr={self.sample_rate})")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Synthesize Arabic text to speech using Voice Design mode."""
        try:
            import numpy as np

            text = text.strip()
            if not text:
                return self.error_response("Input text is empty")

            # Prepend the Arabic voice design description to the user's text
            print(f"[voxcpm2] text: {text[:80]}")

            wav = self.model.generate(
                text=text,
                cfg_value=2.0,
                inference_timesteps=10,
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

            print(
                f"[voxcpm2] audio: len={wav.size}, sr={self.sample_rate}, "
                f"min={wav.min():.4f}, max={wav.max():.4f}"
            )

            audio_base64 = self.audio_to_base64(wav, self.sample_rate)
            return self.success_response(audio_base64, self.sample_rate)

        except Exception as e:
            import traceback
            return self.error_response(f"{e}\n{traceback.format_exc()}")
