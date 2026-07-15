"""Nabra (نبرة) 82M — Kokoro-82M fine-tuned for Modern Standard Arabic.

Pipeline: normalize -> diacritize (camel-tools) -> phonemize (espeak-ng) -> Kokoro.
The demo Space's `arabic_g2p.py` front-end is vendored at build time (pinned) so
the text→phoneme mapping stays byte-for-byte identical to training.

Model: https://huggingface.co/oddadmix/Nabra-82M-v0.1
Demo:  https://huggingface.co/spaces/oddadmix/Nabra-82M-Demo
"""

import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

MODEL_REPO = "oddadmix/Nabra-82M-v0.1"
MODEL_REVISION = "adf6abf35c46db5f2b08803b5067a12c759b1ee1"
MODEL_DIR = "/root/nabra"

SPACE_REPO = "oddadmix/Nabra-82M-Demo"
SPACE_REVISION = "5a3654226f47844c00b4b800061b6d2ce97ba190"
FRONTEND_DIR = "/root/nabra_src"

# Kokoro fork with config=/model= override support used by the demo — pinned.
KOKORO_FORK = "kokoro @ git+https://github.com/Oddadmix/kokoro/@df50e07df746aec0bd7a6f237752d7109ace0b3d"

SAMPLE_RATE = 24000


nabra_82m_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("espeak-ng", "ffmpeg", "libsndfile1", "git")
    .uv_pip_install(
        # CPU-only torch wheel — 82M params synthesize fine on CPU and it
        # keeps the CUDA runtime (~2.5 GB) out of the image.
        "torch==2.7.0",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .uv_pip_install(
        "numpy",
        "soundfile",
        "huggingface_hub[hf_xet]",
        KOKORO_FORK,
        "misaki[en]>=0.9.4",
        "phonemizer-fork>=3.3.2",
        "camel-tools",
    )
    # camel-tools' MSA diacritizer data isn't in the pip package — bake it in.
    .run_commands("camel_data -i disambig-mle-calima-msa-r13")
    # Weights, voicepack, and the Space's Arabic G2P module — all pinned.
    # Must be `python3 -c` shell commands, not .run_function() (which imports
    # this module in a bare build container where the local app/models sources
    # don't exist yet).
    .run_commands(
        "python3 -c \"from huggingface_hub import hf_hub_download; "
        f"[hf_hub_download('{MODEL_REPO}', f, revision='{MODEL_REVISION}', local_dir='{MODEL_DIR}') "
        "for f in ('config.json', 'kokoro_arabic.pth', 'af_msa.pt')]\"",
        "python3 -c \"from huggingface_hub import hf_hub_download; "
        f"hf_hub_download('{SPACE_REPO}', 'arabic_g2p.py', repo_type='space', "
        f"revision='{SPACE_REVISION}', local_dir='{FRONTEND_DIR}')\"",
        secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=nabra_82m_image,
    scaledown_window=60,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    timeout=300,
)
class Nabra82MModel(BaseTTSModel):
    """Nabra (نبرة) 82M — Kokoro-82M fine-tuned for Modern Standard Arabic."""

    model_id = "nabra_82m"
    display_name = "Nabra 82M"
    model_url = "https://huggingface.co/oddadmix/Nabra-82M-v0.1"
    gpu = "CPU"

    @modal.enter()
    def load_model(self):
        import sys
        import torch

        sys.path.insert(0, FRONTEND_DIR)
        from arabic_g2p import ArabicG2P, EXTRA_SYMBOLS, clean_phonemes
        from kokoro import KModel, KPipeline
        from kokoro import pipeline as kpipeline_mod

        # config=/model= keep KModel off the hexgrad base-model fallback;
        # disable_complex uses the conv-based STFT path, which is robust on
        # every backend (the complex path is CUDA-flaky anyway).
        kmodel = KModel(
            repo_id=MODEL_REPO,
            config=f"{MODEL_DIR}/config.json",
            model=f"{MODEL_DIR}/kokoro_arabic.pth",
            disable_complex=True,
        ).eval()
        # ʕ (ع) and ħ (ح) were trained on Kokoro's free vocab gap slots 7/8.
        kmodel.vocab.update(EXTRA_SYMBOLS)

        # Route KPipeline's G2P through espeak-ng Arabic, with the same
        # phoneme cleanup used at training time.
        kpipeline_mod.LANG_CODES.setdefault("ar", "ar")
        self.pipeline = KPipeline(lang_code="ar", repo_id=MODEL_REPO, model=kmodel)
        orig_g2p = self.pipeline.g2p

        def arabic_g2p(text):
            phonemes, extra = orig_g2p(text)
            return clean_phonemes(phonemes), extra

        self.pipeline.g2p = arabic_g2p

        self.g2p = ArabicG2P(diacritize=True)
        # KPipeline.load_voice requires a CPU FloatTensor.
        self.voice = torch.load(
            f"{MODEL_DIR}/af_msa.pt", map_location="cpu", weights_only=True
        )

        print(f"✅ Nabra 82M loaded (sr={SAMPLE_RATE})")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        try:
            import numpy as np
            import torch
            from arabic_g2p import normalize_text

            text = (text or "").strip()
            if not text:
                return self.error_response("Input text is empty")

            # diacritize() is a no-op if the text already carries tashkeel.
            normalized, _ = normalize_text(text)
            diacritized = self.g2p.diacritize(normalized)
            print(f"[nabra_82m] tashkeel: {diacritized[:80]}")

            with torch.inference_mode():
                chunks = [
                    np.asarray(audio.detach().cpu() if torch.is_tensor(audio) else audio)
                    for _, _, audio in self.pipeline(diacritized, voice=self.voice)
                ]
            if not chunks:
                return self.error_response("Nabra produced no audio")

            wav = np.concatenate(chunks).astype(np.float32)
            if wav.size < 100:
                return self.error_response(f"Audio too short: {wav.size} samples")

            return self.success_response(
                self.audio_to_base64(wav, SAMPLE_RATE), SAMPLE_RATE,
            )

        except Exception as e:
            import traceback
            return self.error_response(f"{e}\n{traceback.format_exc()}")
