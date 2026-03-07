import modal
from typing import Optional
from models import BaseTTSModel, register_model
from app import app

spark_tts_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "libsndfile1", "git")
    .uv_pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.36.0",
        "numpy",
        "soundfile",
        "huggingface_hub",
    )
    # Clone SparkTTS repo for inference code and install its dependencies
    .run_commands(
        "git clone https://github.com/SparkAudio/Spark-TTS.git /root/spark-tts && "
        "cd /root/spark-tts && "
        "pip install --upgrade pip && "
        "pip install -r requirements.txt || true"
    )
    # Download the Arabic fine-tuned checkpoint
    .run_commands(
        "python -m huggingface_hub.commands.huggingface_cli download IbrahimSalah/Arabic-TTS-Spark "
        "--local-dir /root/checkpoints/arabic-spark-tts",
        secrets=[modal.Secret.from_name("huggingface")],
    )
)


@register_model
@app.cls(
    image=spark_tts_image,
    gpu="T4",
    scaledown_window=300,
    secrets=[modal.Secret.from_name("huggingface")],
)
class SparkTTSModel(BaseTTSModel):
    """Arabic-TTS-Spark — SparkTTS fine-tuned for Arabic speech synthesis."""

    model_id = "spark_tts"
    display_name = "Spark TTS (Arabic)"
    model_url = "https://huggingface.co/IbrahimSalah/Arabic-TTS-Spark"

    @modal.enter()
    def load_model(self):
        """Load Arabic Spark-TTS when container starts."""
        import sys
        sys.path.insert(0, "/root/spark-tts")

        from cli.SparkTTS import SparkTTS

        self.model = SparkTTS(
            model_dir="/root/checkpoints/arabic-spark-tts",
            device="cuda",
        )
        self.sample_rate = 16000  # SparkTTS default
        print(f"✅ Arabic Spark-TTS loaded on CUDA (sr={self.sample_rate})")

    @modal.method()
    def synthesize(self, text: str, speaker_wav: Optional[str] = None) -> dict:
        """Synthesize Arabic text to speech."""
        try:
            import torch

            with torch.no_grad():
                wav = self.model.inference(
                    text=text,
                    prompt_speech_path=speaker_wav,
                )

            # wav is a torch tensor
            if isinstance(wav, torch.Tensor):
                wav_np = wav.squeeze().cpu().numpy()
            else:
                import numpy as np
                wav_np = wav if isinstance(wav, np.ndarray) else np.array(wav)

            audio_base64 = self.audio_to_base64(wav_np, self.sample_rate)
            return self.success_response(audio_base64, self.sample_rate)
        except Exception as e:
            return self.error_response(e)
