"""
OmniVoice — Omnilingual Zero-Shot TTS with Voice Design
=========================================================

Uses the k2-fsa/OmniVoice model (Diffusion Language Model architecture)
with 600+ language support.  Generates speech using **Voice Design** mode
(instruct parameter) instead of voice cloning from reference audio.

Key features:
    1. **Voice Design mode** — generates a natural Arabic voice from a text
       description (instruct), avoiding mismatched reference audio issues.
    2. **Fast inference** — RTF as low as 0.025 (40x real-time).

Model:  https://huggingface.co/k2-fsa/OmniVoice
Paper:  https://arxiv.org/abs/2604.00688
"""

import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

# Voice design instruct — OmniVoice generates a natural Arabic voice from
# a text description rather than cloning from a reference audio.
_VOICE_INSTRUCT = "female"

# ---------------------------------------------------------------------------
# Modal image
# ---------------------------------------------------------------------------

omnivoice_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy",
        "soundfile",
        "huggingface_hub",
        "omnivoice",
        "requests",
    )
    # Pre-download OmniVoice model weights so they're baked into the image
    .run_commands(
        "python3 -c \""
        "from omnivoice import OmniVoice; "
        "import torch; "
        "model = OmniVoice.from_pretrained("
        "    'k2-fsa/OmniVoice',"
        "    device_map='cpu',"
        "    dtype=torch.float32,"
        "); "
        "print(f'OmniVoice downloaded, sampling_rate={model.sampling_rate}')"
        "\"",
        secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    )
    .add_local_python_source(*LOCAL_MODULES)
)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

@register_model
@app.cls(
    image=omnivoice_image,
    gpu="A10G",
    scaledown_window=300,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class OmniVoiceModel(BaseTTSModel):
    """OmniVoice — omnilingual zero-shot TTS with Voice Design mode.

    Uses the k2-fsa/OmniVoice model (600+ languages, diffusion LM
    architecture) to synthesise Arabic speech via Voice Design —
    describing the desired voice with the instruct parameter instead
    of cloning from reference audio.

    Model: https://huggingface.co/k2-fsa/OmniVoice
    """

    model_id = "omnivoice"
    display_name = "OmniVoice"
    model_url = "https://huggingface.co/k2-fsa/OmniVoice"
    gpu = "A10G"

    @modal.enter()
    def load_model(self):
        """Load the OmniVoice TTS model."""
        import torch
        from omnivoice import OmniVoice

        self.model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map="cuda",
            dtype=torch.bfloat16,
        )
        self.sample_rate = self.model.sampling_rate  # 24000 Hz

        print(f"✅ OmniVoice loaded on CUDA (sr={self.sample_rate}, "
              f"mode=voice-design)")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Synthesise Arabic text using Voice Design mode.

        Args:
            text: Arabic text to synthesise.
        """
        try:
            from omnivoice import OmniVoiceGenerationConfig

            print(f"[omnivoice] Voice Design mode, "
                  f"instruct='{_VOICE_INSTRUCT}', "
                  f"text={text[:60]}…")

            gen_config = OmniVoiceGenerationConfig(
                num_step=32,
                guidance_scale=2.0,
                denoise=True,
                postprocess_output=True,
            )

            audios = self.model.generate(
                text=text,
                instruct=_VOICE_INSTRUCT,
                generation_config=gen_config,
            )

            # audios is a list of tensors with shape (1, T) at 24 kHz
            wav = audios[0]
            wav_np = wav.squeeze().cpu().numpy()

            audio_base64 = self.audio_to_base64(wav_np, self.sample_rate)

            result = self.success_response(audio_base64, self.sample_rate)
            result["voice_mode"] = "design"
            return result

        except Exception as e:
            import traceback
            return self.error_response(f"{e}\n{traceback.format_exc()}")
