import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

habibi_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "libsndfile1", "git")
    .uv_pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy",
        "soundfile",
        "huggingface_hub",
        "f5-tts",
    )
    # Pre-download the Habibi-TTS checkpoint + Vocos vocoder
    .run_commands(
        "python3 -c \""
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('SWivid/Habibi-TTS', local_dir='/root/checkpoints/habibi-tts'); "
        "snapshot_download('charactr/vocos-mel-24khz', local_dir='/root/checkpoints/vocos-mel-24khz')"
        "\"",
        # List what we got so we can debug file layout
        "ls -la /root/checkpoints/habibi-tts/",
        secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=habibi_image,
    gpu="T4",
    scaledown_window=300,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class HabibiTTSModel(BaseTTSModel):
    """Habibi-TTS — F5-TTS fine-tuned for Arabic speech synthesis."""

    model_id = "habibi_tts"
    display_name = "Habibi TTS"
    model_url = "https://github.com/SWivid/Habibi-TTS"

    @modal.enter()
    def load_model(self):
        """Load Habibi-TTS (F5-TTS Arabic) when container starts."""
        import os
        from f5_tts.api import F5TTS

        ckpt_dir = "/root/checkpoints/habibi-tts"

        # Habibi-TTS may ship vocab as vocab.txt or vocab.json — detect it
        vocab_file = ""
        for name in ("vocab.txt", "vocab.json"):
            path = os.path.join(ckpt_dir, name)
            if os.path.exists(path):
                vocab_file = path
                break

        print(f"  ckpt dir contents: {os.listdir(ckpt_dir)}")
        print(f"  vocab file: {vocab_file or '(none — using char tokenizer)'}")

        self.tts = F5TTS(
            ckpt_file=os.path.join(ckpt_dir, "model_last.safetensors"),
            vocab_file=vocab_file if vocab_file else "",
            tokenizer="custom" if vocab_file else "char",
            device="cuda",
        )
        self.sample_rate = 24000  # F5-TTS default
        print(f"✅ Habibi-TTS loaded on CUDA (sr={self.sample_rate})")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Synthesize Arabic text to speech."""
        try:
            wav, sr, _ = self.tts.infer(
                ref_file="",   # zero-shot — no reference audio needed
                ref_text="",   # no reference text
                gen_text=text,
            )

            # wav is a numpy array from F5-TTS
            audio_base64 = self.audio_to_base64(wav, sr)
            return self.success_response(audio_base64, sr)
        except Exception as e:
            return self.error_response(e)
