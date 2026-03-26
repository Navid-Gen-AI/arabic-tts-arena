import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

oute_tts_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "libsndfile1", "clang", "git", "cmake", "libomp-dev")
    .uv_pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "torchcodec",
        "numpy",
        "soundfile",
        "huggingface_hub",
        "outetts>=0.4.0",
    )
    # Pre-download the model weights + DAC codec + reference audio
    .run_commands(
        "python3 -c \""
        "import outetts; "
        "outetts.Interface(config=outetts.ModelConfig.auto_config("
        "    model=outetts.Models.VERSION_1_0_SIZE_1B,"
        "    backend=outetts.Backend.HF,"
        "))"
        "\"",
        # Download Arabic reference audio
        "python3 -c \""
        "from huggingface_hub import hf_hub_download; "
        "hf_hub_download("
        "    repo_id='IbrahimSalah/Arabic-TTS-Spark',"
        "    repo_type='space',"
        "    filename='reference.wav',"
        "    local_dir='/root/oute-ref'"
        ")\"",
        secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=oute_tts_image,
    gpu="A10G",
    scaledown_window=300,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class OuteTTSModel(BaseTTSModel):
    """OuteTTS 1.0 (1B) — Llama-based multilingual TTS with strong Arabic support.

    Arabic is a "High Training Data" language. Uses DAC audio encoder for
    high-fidelity output. Supports one-shot voice cloning.

    Source: https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B
    """

    model_id = "oute_tts"
    display_name = "OuteTTS 1.0"
    model_url = "https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B"
    gpu = "A10G"

    @modal.enter()
    def load_model(self):
        """Load OuteTTS interface and create Arabic speaker profile."""
        import outetts

        self.interface = outetts.Interface(
            config=outetts.ModelConfig.auto_config(
                model=outetts.Models.VERSION_1_0_SIZE_1B,
                backend=outetts.Backend.HF,
            )
        )

        # Create an Arabic speaker profile from reference audio
        self._ref_audio = "/root/oute-ref/reference.wav"
        self.speaker = self.interface.create_speaker(self._ref_audio)

        # OuteTTS uses DAC codec — sample rate determined by the codec
        # ibm-research/DAC.speech.v1.0 outputs at 16kHz
        self.sample_rate = 16000

        print(f"✅ OuteTTS 1.0 loaded on CUDA (sr={self.sample_rate})")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Synthesize Arabic text to speech."""
        try:
            import outetts
            import numpy as np

            text = text.strip()
            if not text:
                return self.error_response("Input text is empty")

            print(f"[oute_tts] text: {text[:80]}")

            output = self.interface.generate(
                config=outetts.GenerationConfig(
                    text=text,
                    generation_type=outetts.GenerationType.CHUNKED,
                    speaker=self.speaker,
                    sampler_config=outetts.SamplerConfig(
                        temperature=0.4,
                        top_k=40,
                        top_p=0.9,
                        min_p=0.05,
                        repetition_penalty=1.1,
                    ),
                )
            )

            # Get audio as numpy array
            audio = output.audio
            if hasattr(audio, 'cpu'):
                audio = audio.cpu().numpy()
            if not isinstance(audio, np.ndarray):
                audio = np.asarray(audio, dtype=np.float32)
            audio = audio.astype(np.float32, copy=False)

            if audio.ndim > 1:
                audio = audio.reshape(-1)

            # Get actual sample rate from output if available
            sr = getattr(output, 'sr', None) or getattr(output, 'sample_rate', None) or self.sample_rate

            if audio.size < 100:
                return self.error_response(
                    f"Audio too short: {audio.size} samples"
                )

            print(f"[oute_tts] audio: len={audio.size}, sr={sr}")

            audio_base64 = self.audio_to_base64(audio, sr)
            return self.success_response(audio_base64, sr)

        except Exception as e:
            import traceback
            return self.error_response(f"{e}\n{traceback.format_exc()}")
