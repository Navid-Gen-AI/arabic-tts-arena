import modal
from typing import Optional
from models import BaseTTSModel, register_model
from app import app

chatterbox_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "libsndfile1", "espeak-ng")
    .uv_pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.26.0",
        "soundfile",
        "chatterbox-tts",
    )
)


@register_model
@app.cls(
    image=chatterbox_image,
    gpu="T4",
    scaledown_window=300,
    secrets=[modal.Secret.from_name("huggingface")],
)
class ChatterboxModel(BaseTTSModel):
    """ResembleAI Chatterbox — multilingual TTS with Arabic support."""

    model_id = "chatterbox"
    display_name = "Chatterbox"
    model_url = "https://huggingface.co/ResembleAI/chatterbox"

    @modal.enter()
    def load_model(self):
        """Load the Chatterbox multilingual model when container starts."""
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self.model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
        self.sample_rate = self.model.sr
        print(f"✅ Chatterbox multilingual loaded on CUDA (sr={self.sample_rate})")

    @modal.method()
    def synthesize(self, text: str, speaker_wav: Optional[str] = None) -> dict:
        """Synthesize Arabic text to speech."""
        try:
            wav = self.model.generate(
                text,
                lang="ar",
                exaggeration=0.5,
                cfg=0.5,
            )

            # wav is a torch tensor, convert to numpy
            wav_np = wav.squeeze().cpu().numpy()
            audio_base64 = self.audio_to_base64(wav_np, self.sample_rate)

            return self.success_response(audio_base64, self.sample_rate)
        except Exception as e:
            return self.error_response(e)
