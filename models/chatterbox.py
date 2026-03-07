import modal
from typing import Optional
from models import BaseTTSModel, register_model
from app import app, base_gpu_image

chatterbox_image = base_gpu_image.uv_pip_install("chatterbox-tts")


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
                language_id="ar",
                exaggeration=0.5,
                cfg=0.5,
            )

            # wav is a torch tensor, convert to numpy
            wav_np = wav.squeeze().cpu().numpy()
            audio_base64 = self.audio_to_base64(wav_np, self.sample_rate)

            return self.success_response(audio_base64, self.sample_rate)
        except Exception as e:
            return self.error_response(e)
