"""VoxCPM2 — OpenBMB's 2B-param tokenizer-free diffusion-AR multilingual TTS.

Arabic is supported natively (no language tag needed) and output is 48 kHz
via the built-in AudioVAE V2 super-resolution. Synthesizes from text alone,
no reference audio required.

Model:  https://huggingface.co/openbmb/VoxCPM2
GitHub: https://github.com/OpenBMB/VoxCPM
Paper:  https://arxiv.org/abs/2509.24650
"""

import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

MODEL_REPO = "openbmb/VoxCPM2"


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
    # Weights baked in via a throwaway from_pretrained. Must be a `python3 -c`
    # shell command, not .run_function() (which imports this module in a bare
    # build container where the local app/models sources don't exist yet).
    # Command string matches the originally-deployed image byte-for-byte so
    # Modal's cache reuses it (a rebuild needs an A10G builder).
    # gpu: VoxCPM.from_pretrained initializes on CUDA even with optimize=False.
    .run_commands(
        "python3 -c \""
        "from voxcpm import VoxCPM; "
        "model = VoxCPM.from_pretrained("
        f"    '{MODEL_REPO}',"
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


@register_model
@app.cls(
    image=voxcpm2_image,
    gpu="A10G",
    scaledown_window=60,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class VoxCPM2Model(BaseTTSModel):
    """VoxCPM2 — OpenBMB tokenizer-free multilingual TTS (2B params)."""

    model_id = "voxcpm2"
    display_name = "VoxCPM 2"
    model_url = "https://huggingface.co/openbmb/VoxCPM2"
    gpu = "A10G"

    @modal.enter()
    def load_model(self):
        from voxcpm import VoxCPM

        self.model = VoxCPM.from_pretrained(
            MODEL_REPO,
            load_denoiser=False,
            optimize=False,
        )
        self.sample_rate = self.model.tts_model.sample_rate  # 48000

        print(f"✅ VoxCPM2 loaded on CUDA (sr={self.sample_rate})")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        try:
            import numpy as np

            text = text.strip()
            if not text:
                return self.error_response("Input text is empty")

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
                return self.error_response(f"Audio too short: {wav.size} samples")

            return self.success_response(
                self.audio_to_base64(wav, self.sample_rate), self.sample_rate,
            )

        except Exception as e:
            import traceback
            return self.error_response(f"{e}\n{traceback.format_exc()}")
