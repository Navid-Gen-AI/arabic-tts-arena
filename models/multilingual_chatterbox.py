import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

# The multilingual Chatterbox model is NOT available via `chatterbox-tts` pip package.
# It lives in a custom `src/chatterbox` module in the HF Space. We clone it at image
# build time so we can import `chatterbox.mtl_tts.ChatterboxMultilingualTTS`.
CHATTERBOX_MTL_REPO = "https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS"

# Default Arabic voice prompt used by the official Space
AR_VOICE_PROMPT_URL = (
    "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac"
)

chatterbox_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "libsndfile1", "git")
    .uv_pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy==1.26.0",
        "soundfile",
        "librosa==0.10.0",
        "resampy==0.4.3",
        "s3tokenizer",
        "transformers==4.46.3",
        "diffusers==0.29.0",
        "omegaconf==2.3.0",
        "resemble-perth==1.0.1",
        "silero-vad==5.1.2",
        "conformer==0.3.2",
        "safetensors",
        "huggingface_hub",
    )
    # Clone the HF Space repo and make src/ importable
    .run_commands(
        f"git clone {CHATTERBOX_MTL_REPO} /opt/chatterbox-mtl",
        "cp -r /opt/chatterbox-mtl/src/chatterbox /usr/local/lib/python3.12/site-packages/chatterbox",
    )
    # Pre-download model weights and Arabic voice prompt so they're baked into the image
    .run_commands(
        "python3 -c \""
        "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; "
        "ChatterboxMultilingualTTS.from_pretrained(device='cuda')"
        "\"",
        secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
        gpu="T4",
    )
    .run_commands(
        f"python3 -c \""
        f"import urllib.request; "
        f"urllib.request.urlretrieve('{AR_VOICE_PROMPT_URL}', '/root/ar_voice_prompt.flac')"
        f"\"",
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=chatterbox_image,
    gpu="T4",
    scaledown_window=300,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class ChatterboxModel(BaseTTSModel):
    """ResembleAI Chatterbox Multilingual — TTS with Arabic support."""

    model_id = "chatterbox"
    display_name = "Multilingual Chatterbox"
    model_url = "https://huggingface.co/ResembleAI/chatterbox"
    gpu = "T4"

    @modal.enter()
    def load_model(self):
        """Load the Chatterbox Multilingual TTS model when container starts."""
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self.model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
        self.sample_rate = self.model.sr

        # Use the pre-downloaded Arabic voice prompt (baked into the image)
        self._ar_prompt_path = "/root/ar_voice_prompt.flac"

        print(f"✅ Chatterbox Multilingual loaded on CUDA (sr={self.sample_rate})")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Synthesize Arabic text to speech."""
        try:
            wav = self.model.generate(
                text,
                language_id="ar",
                audio_prompt_path=self._ar_prompt_path,
                exaggeration=0.5,
                temperature=0.8,
                cfg_weight=0.5,
            )

            # wav is a torch tensor, convert to numpy
            wav_np = wav.squeeze().cpu().numpy()
            audio_base64 = self.audio_to_base64(wav_np, self.sample_rate)

            return self.success_response(audio_base64, self.sample_rate)
        except Exception as e:
            return self.error_response(e)
