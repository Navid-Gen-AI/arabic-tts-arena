"""OmniVoice — k2-fsa omnilingual zero-shot TTS (diffusion LM, 600+ languages).

Synthesizes via Voice Design: the voice is described with a text `instruct`
instead of cloned from reference audio, avoiding mismatched-reference issues.

Model: https://huggingface.co/k2-fsa/OmniVoice
Paper: https://arxiv.org/abs/2604.00688
"""

import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

MODEL_REPO = "k2-fsa/OmniVoice"
# Voice Design description — OmniVoice generates a voice matching this text.
VOICE_INSTRUCT = "female"


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
    # Weights baked in via a throwaway from_pretrained. Must be a `python3 -c`
    # shell command, not .run_function() (which imports this module in a bare
    # build container where the local app/models sources don't exist yet).
    # Command string matches the originally-deployed image byte-for-byte so
    # Modal's cache reuses it.
    .run_commands(
        "python3 -c \""
        "from omnivoice import OmniVoice; "
        "import torch; "
        "model = OmniVoice.from_pretrained("
        f"    '{MODEL_REPO}',"
        "    device_map='cpu',"
        "    dtype=torch.float32,"
        "); "
        "print(f'OmniVoice downloaded, sampling_rate={model.sampling_rate}')"
        "\"",
        secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=omnivoice_image,
    gpu="A10G",
    scaledown_window=60,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class OmniVoiceModel(BaseTTSModel):
    """OmniVoice — omnilingual zero-shot TTS with Voice Design mode."""

    model_id = "omnivoice"
    display_name = "OmniVoice"
    model_url = "https://huggingface.co/k2-fsa/OmniVoice"
    gpu = "A10G"

    @modal.enter()
    def load_model(self):
        import torch
        from omnivoice import OmniVoice

        self.model = OmniVoice.from_pretrained(
            MODEL_REPO,
            device_map="cuda",
            dtype=torch.bfloat16,
        )
        self.sample_rate = self.model.sampling_rate  # 24000 Hz

        print(f"✅ OmniVoice loaded on CUDA (sr={self.sample_rate}, mode=voice-design)")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        try:
            from omnivoice import OmniVoiceGenerationConfig

            print(f"[omnivoice] instruct='{VOICE_INSTRUCT}', text={text[:60]}…")

            gen_config = OmniVoiceGenerationConfig(
                num_step=32,
                guidance_scale=2.0,
                denoise=True,
                postprocess_output=True,
            )
            audios = self.model.generate(
                text=text,
                instruct=VOICE_INSTRUCT,
                generation_config=gen_config,
            )

            # generate() returns a list of (1, T) tensors at 24 kHz.
            wav_np = audios[0].squeeze().cpu().numpy()

            result = self.success_response(
                self.audio_to_base64(wav_np, self.sample_rate), self.sample_rate,
            )
            result["voice_mode"] = "design"
            return result

        except Exception as e:
            import traceback
            return self.error_response(f"{e}\n{traceback.format_exc()}")
