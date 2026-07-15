"""SILMA TTS v1 Small — 150M-parameter bilingual (Arabic/English) TTS.

Uses the `silma-tts` pip package; inference clones a fixed Arabic reference
clip shipped in the upstream repo.

Model: https://huggingface.co/silma-ai/silma-tts
"""

import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

# Fixed Arabic reference clip from the upstream repo, plus its transcript.
REF_AUDIO_URL = (
    "https://github.com/SILMA-AI/silma-tts/raw/refs/heads/main"
    "/src/silma_tts/infer/ref_audio_samples/ar.ref.24k.wav"
)
REF_TEXT = "ويدقق النظر في القرآن الكريم وسائر الكتب السماوية ويتبع مسالك الرسل العظام عليهم الصلاة والسلام."

SAMPLE_RATE = 24000


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
    # Weights + Arabic reference clip. Must be `python3 -c` shell commands, not
    # .run_function() (which imports this module in a bare build container
    # where the local app/models sources don't exist yet). Kept as two separate
    # layers — byte-identical to the originally-deployed image so Modal reuses
    # the cached build (a fresh SilmaTTS() warm-load can OOM the builder).
    .run_commands(
        "python3 -c \""
        "from silma_tts.api import SilmaTTS; "
        "SilmaTTS()"
        "\"",
    )
    .run_commands(
        "python3 -c \""
        "from cached_path import cached_path; "
        f"cached_path('{REF_AUDIO_URL}')"
        "\"",
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=silma_tts_v1_small_image,
    gpu="T4",
    scaledown_window=120,
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
        from silma_tts.api import SilmaTTS
        from cached_path import cached_path

        self.model = SilmaTTS()
        self.sample_rate = SAMPLE_RATE
        self._ref_audio_path = cached_path(REF_AUDIO_URL)
        self._ref_text = REF_TEXT

        print(f"✅ SILMA TTS Loaded (ref_audio_path={self._ref_audio_path})")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        try:
            wav, sr, spec = self.model.infer(
                ref_file=self._ref_audio_path,
                ref_text=self._ref_text,
                gen_text=text,
                file_wave=None,
                seed=None,
                speed=1
            )

            return self.success_response(self.audio_to_base64(wav, sr), sr)

        except Exception as e:
            import traceback
            return self.error_response(f"{e}\n{traceback.format_exc()}")
