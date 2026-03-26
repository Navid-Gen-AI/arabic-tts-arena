import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES


silma_tts_v1_small_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "libsndfile1", "git", "build-essential", "clang")
    .uv_pip_install(
        "silma-tts",
        "cached-path"
    )
    # Pre-download model weights and reference audio so they're baked into the image
    .run_commands(
        "python3 -c \""
        "from silma_tts.api import SilmaTTS; "
        "SilmaTTS()"
        "\"",
    )
    .run_commands(
        "python3 -c \""
        "from cached_path import cached_path; "
        "cached_path('https://github.com/SILMA-AI/silma-tts/raw/refs/heads/main/src/silma_tts/infer/ref_audio_samples/ar.ref.24k.wav')"
        "\"",
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=silma_tts_v1_small_image,
    gpu="T4",
    scaledown_window=300,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class SilmaSmallTTSModel(BaseTTSModel):
    """
    SILMA TTS v1 is a high-performance, 150M-parameter bilingual (Arabic/English) TTS model
    """

    model_id = "silma_tts_v1_small"
    display_name = "SILMA TTS v1 Small"
    model_url = "https://huggingface.co/silma-ai/silma-tts"
    gpu = "T4"

    @modal.enter()
    def load_model(self):
        """Load SILMA TTS using pip package"""
       
        from silma_tts.api import SilmaTTS
        from cached_path import cached_path

        print("Loading SILMA TTS...")

        ## Load models/weights
        self.model = SilmaTTS()

        url = "https://github.com/SILMA-AI/silma-tts/raw/refs/heads/main/src/silma_tts/infer/ref_audio_samples/ar.ref.24k.wav"

        self.sample_rate = 24000
        self._ref_audio_path = cached_path(url)
        self._ref_text = "ويدقق النظر في القرآن الكريم وسائر الكتب السماوية ويتبع مسالك الرسل العظام عليهم الصلاة والسلام."


        print(f"✅ SILMA TTS Loaded (ref_audio_path={self._ref_audio_path})")


    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Synthesize Arabic text."""

        try:
            
            wav, sr, spec = self.model.infer(
                ref_file=self._ref_audio_path,
                ref_text=self._ref_text,
                gen_text=text,
                file_wave=None,
                seed=None,
                speed=1
            )
            
            audio_base64 = self.audio_to_base64(wav, sr)
            
            return self.success_response(audio_base64, sr)

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return self.error_response(f"{e}\n{traceback.format_exc()}")