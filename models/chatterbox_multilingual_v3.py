"""
Chatterbox Multilingual V3 — ResembleAI multilingual TTS with Arabic support
============================================================================

Resemble AI's latest general-purpose multilingual TTS (0.5B). V3 improves
speaker similarity, reduces hallucinations, and produces more natural speech
than the V2 checkpoint, while keeping native Arabic (``ar``) support.

The V3 checkpoint is opt-in via ``t3_model="v3"``. The last tagged PyPI
release (v0.1.2, Jun 2025) predates V3, so we install from the official
GitHub source to guarantee the V3 weights and loader are available.

Model:   https://huggingface.co/ResembleAI/chatterbox
GitHub:  https://github.com/resemble-ai/chatterbox
License: MIT
"""

import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

# Official resemble-ai/chatterbox repo. Its src/ ships the opt-in V3
# multilingual loader (t3_model="v3"); the last tagged PyPI release (v0.1.2,
# Jun 2025) predates V3, so we install from GitHub master.
#
# We `pip install .` (not copy-src) so the package both (a) installs its OWN
# declared dependencies — torch==2.6.0, transformers==5.2.0, pykakasi,
# spacy-pkuseg, pyloudnorm, resemble-perth, ... which the V2-era pinned set does
# NOT satisfy — and (b) registers the `chatterbox-tts` distribution metadata
# that chatterbox/__init__.py reads via importlib.metadata.version(). The PyPI
# torch wheel bundles its own CUDA libs, so it runs on this image as-is.
CHATTERBOX_REPO = "https://github.com/resemble-ai/chatterbox.git"

# Official Arabic (female) voice prompt used by the Chatterbox Multilingual
# Space. The reference language matches the synthesis language (Arabic), so
# the default cfg_weight avoids any cross-lingual accent leakage.
AR_VOICE_PROMPT_URL = (
    "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac"
)

chatterbox_v3_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "libsndfile1", "git")
    # Install Chatterbox (incl. the opt-in V3 multilingual loader) from source,
    # letting it resolve its own pinned dependencies and register its metadata.
    .run_commands(
        f"git clone --depth 1 {CHATTERBOX_REPO} /opt/chatterbox",
        "cd /opt/chatterbox && python3 -m pip install .",
    )
    # Pre-download the V3 multilingual weights so they're baked into the image
    .run_commands(
        "python3 -c \""
        "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; "
        "ChatterboxMultilingualTTS.from_pretrained(device='cuda', t3_model='v3')"
        "\"",
        secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
        gpu="T4",
    )
    # Pre-download the Arabic voice prompt (baked into the image)
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
    image=chatterbox_v3_image,
    gpu="T4",
    scaledown_window=120,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class ChatterboxMultilingualV3Model(BaseTTSModel):
    """ResembleAI Chatterbox Multilingual V3 — TTS with Arabic support."""

    model_id = "chatterbox_multilingual_v3"
    display_name = "Chatterbox Multilingual V3"
    model_url = "https://huggingface.co/ResembleAI/chatterbox"
    gpu = "T4"

    @modal.enter()
    def load_model(self):
        """Load the Chatterbox Multilingual V3 model when the container starts."""
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self.model = ChatterboxMultilingualTTS.from_pretrained(
            device="cuda",
            t3_model="v3",
        )
        self.sample_rate = self.model.sr

        # Use the pre-downloaded Arabic voice prompt (baked into the image)
        self._ar_prompt_path = "/root/ar_voice_prompt.flac"

        print(f"✅ Chatterbox Multilingual V3 loaded on CUDA (sr={self.sample_rate})")

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
