import modal
from typing import Optional
from models import BaseTTSModel, register_model
from app import app

# Coqui TTS requires Python < 3.12, so XTTS-v2 gets its own base image
xtts_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.11",
    )
    .apt_install("ffmpeg", "libsndfile1", "espeak-ng")
    .uv_pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy",
        "soundfile",
        "TTS>=0.22.0",
    )
)


@register_model
@app.cls(
    image=xtts_image,
    gpu="T4",
    scaledown_window=300,
    secrets=[modal.Secret.from_name("huggingface")],
)
class XTTSv2Model(BaseTTSModel):
    """Coqui XTTS-v2 with Arabic support."""
    
    model_id = "xtts_v2"
    display_name = "XTTS-v2"
    model_url = "https://huggingface.co/coqui/XTTS-v2"
    
    @modal.enter()
    def load_model(self):
        """Load XTTS-v2 model when container starts."""
        from TTS.api import TTS
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        self.device = device
        self.default_speaker_wav = None
        print(f"✅ XTTS-v2 loaded on {device}")
    
    @modal.method()
    def synthesize(self, text: str, speaker_wav: Optional[str] = None) -> dict:
        """Synthesize Arabic text to speech."""
        wav = self.tts.tts(
            text=text,
            language="ar",
            speaker_wav=speaker_wav or self.default_speaker_wav,
        )
        
        sample_rate = 24000
        audio_base64 = self.audio_to_base64(wav, sample_rate)
        
        return self.success_response(audio_base64, sample_rate)
