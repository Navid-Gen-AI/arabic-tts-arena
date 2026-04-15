import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

# Reference text for the Unified model (MSA — Modern Standard Arabic).
# Source: official Habibi-TTS assets (ElevenLabs voice library, ID JjTirzdD7T3GMLkwdd3a).
_REF_TEXT = (
    "كان اللعيب حاضرًا في العديد من الأنشطة والفعاليات المرتبطة بكأس العالم، "
    "مما سمح للجماهير بالتفاعل معه والتقاط الصور التذكارية."
)

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
        "huggingface_hub[hf_xet]",
        "f5-tts",
        "cached-path",
    )
    # Pre-download the Unified checkpoint + vocab + MSA reference audio
    .run_commands(
        "python3 -c \""
        "from cached_path import cached_path; "
        "print('ckpt:', cached_path('hf://SWivid/Habibi-TTS/Unified/model_200000.safetensors')); "
        "print('vocab:', cached_path('hf://SWivid/Habibi-TTS/Unified/vocab.txt'))"
        "\"",
        # Download MSA reference audio from the official Habibi-TTS HF Space
        "python3 -c \""
        "from huggingface_hub import hf_hub_download; "
        "p = hf_hub_download(repo_id='chenxie95/Habibi-TTS', repo_type='space', "
        "filename='assets/MSA.mp3', local_dir='/root'); "
        "print('ref audio downloaded:', p)"
        "\"",
        secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=habibi_image,
    gpu="T4",
    scaledown_window=120,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class HabibiTTSModel(BaseTTSModel):
    """Habibi-TTS — F5-TTS fine-tuned for Arabic speech synthesis (Unified model)."""

    model_id = "habibi_tts"
    display_name = "Habibi TTS"
    model_url = "https://github.com/SWivid/Habibi-TTS"
    gpu = "T4"

    @modal.enter()
    def load_model(self):
        """Load Habibi-TTS Unified model when container starts."""
        from cached_path import cached_path
        from f5_tts.infer.utils_infer import load_model, load_vocoder
        from f5_tts.model import DiT

        # Habibi-TTS Unified model config — identical to F5TTS_v1_Base architecture
        self._model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
        )

        # Resolve checkpoint & vocab paths (cached from image build)
        ckpt_file = str(
            cached_path("hf://SWivid/Habibi-TTS/Unified/model_200000.safetensors")
        )
        vocab_file = str(
            cached_path("hf://SWivid/Habibi-TTS/Unified/vocab.txt")
        )

        print(f"  ckpt file : {ckpt_file}")
        print(f"  vocab file: {vocab_file}")

        # Load the model using F5-TTS low-level API (same as official Habibi-TTS code)
        self.ema_model = load_model(
            DiT,
            self._model_cfg,
            ckpt_file,
            mel_spec_type="vocos",
            vocab_file=vocab_file,
            device="cuda",
        )

        # Load the Vocos vocoder (downloaded automatically by F5-TTS)
        self.vocoder = load_vocoder(vocoder_name="vocos", device="cuda")

        self.sample_rate = 24000
        self._ref_audio = "/root/assets/MSA.mp3"
        self._ref_text = _REF_TEXT

        print(f"✅ Habibi-TTS Unified loaded on CUDA (sr={self.sample_rate})")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Synthesize Arabic text to speech using the Unified model."""
        try:
            from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

            # Preprocess the reference audio (resampling, trimming, ASR if needed)
            ref_audio, ref_text = preprocess_ref_audio_text(
                self._ref_audio, self._ref_text, show_info=print
            )

            # Run inference — matches official Habibi-TTS infer_process() usage
            wav, sr, _ = infer_process(
                ref_audio,
                ref_text,
                text,
                self.ema_model,
                self.vocoder,
                mel_spec_type="vocos",
                speed=1.0,
                device="cuda",
            )

            audio_base64 = self.audio_to_base64(wav, sr)
            return self.success_response(audio_base64, sr)
        except Exception as e:
            return self.error_response(e)
