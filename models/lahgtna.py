import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES
import random

# Dialect-specific reference audio & prompts bundled in the HF Space
# Source: https://huggingface.co/spaces/oddadmix/lahgtna-chatterbox-demo
DIALECT_CONFIG = {
    "eg": {
        "label": "Egyptian",
        "audio_filename": "fdddf1f2a317e626a0db94d6d7d92e772039121f120787c80be2250c09999ec1.wav",
        "text": "إزيك يا صاحبي؟ أنا رايح الشغل دلوقتي وهكلمك بعدين. متقلقش عليا أنا عارف الطريق كويس.",
    },
    "sa": {
        "label": "Saudi",
        "audio_filename": "saudi-ref.wav",
        "text": "كيف الحال يا أخوي؟ أنا بروح الشغل الحين وأتصل فيك بعدين. لا تشيل هم أنا أعرف الطريق زين.",
    },
    "mo": {
        "label": "Moroccan",
        "audio_filename": "mor-ref.wav",
        "text": "لاباس عليك يا صاحبي؟ أنا غادي للخدمة دابا ونتصل بيك من بعد. ماتقلقش علي أنا كنعرف الطريق مزيان.",
    },
    "iq": {
        "label": "Iraqi",
        "audio_filename": "iraqi-ref.wav",
        "text": "شلونك يا صديقي؟ أنا رايح للشغل هسه وأتصل بيك بعدين. لا تشيل هم أنا أعرف الطريق زين.",
    },
}

ALL_DIALECTS = list(DIALECT_CONFIG.keys())

LAHGTNA_REPO = "https://huggingface.co/spaces/oddadmix/lahgtna-chatterbox-demo"

lahgtna_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "libsndfile1", "git")
    .uv_pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy==1.26.0",
        "soundfile",
        "librosa==0.10.0",
        "resampy==0.4.3",
        "s3tokenizer",
        "transformers==4.46.3",
        "diffusers==0.29.0",
        "omegaconf==2.3.0",
        "resemble-perth==1.0.1",
        "silero-vad==5.1.2",
        "conformer==0.3.2",
        "safetensors",
        "huggingface_hub",
    )
    # Download chatterbox source + reference WAVs from the HF Space.
    # NOTE: plain `git clone` only fetches LFS pointer files for WAVs,
    # so we use huggingface_hub to download them properly.
    .env({"LAHGTNA_CACHE_BUST": "2026-03-10"})  # bump to force re-download
    .run_commands(
        # 1. Download the chatterbox Python module (src/chatterbox/*)
        "python3 -c \""
        "from huggingface_hub import snapshot_download; "
        "snapshot_download("
        "    repo_id='oddadmix/lahgtna-chatterbox-demo',"
        "    repo_type='space',"
        "    allow_patterns=['src/chatterbox/**'],"
        "    local_dir='/opt/lahgtna-chatterbox'"
        ")\"",
        # Copy the chatterbox module into site-packages
        "cp -r /opt/lahgtna-chatterbox/src/chatterbox "
        "/usr/local/lib/python3.12/site-packages/chatterbox",
        # 2. Download all reference WAV files
        "python3 -c \""
        "from huggingface_hub import hf_hub_download; "
        "wavs = ["
        "    'fdddf1f2a317e626a0db94d6d7d92e772039121f120787c80be2250c09999ec1.wav',"
        "    'saudi-ref.wav',"
        "    'mor-ref.wav',"
        "    'iraqi-ref.wav',"
        "]; "
        "[hf_hub_download("
        "    repo_id='oddadmix/lahgtna-chatterbox-demo',"
        "    repo_type='space',"
        "    filename=w,"
        "    local_dir='/opt/lahgtna-chatterbox'"
        ") for w in wavs]; "
        "print('All Lahgtna reference WAVs downloaded')"
        "\"",
        # 3. Pre-download the fine-tuned Arabic model weights at build time
        #    so from_pretrained() finds them in cache and uses the correct weights.
        "python3 -c \""
        "from huggingface_hub import snapshot_download; "
        "snapshot_download("
        "    repo_id='oddadmix/lahgtna-chatterbox-v0',"
        "    repo_type='model',"
        "    revision='main',"
        "    allow_patterns=["
        "        've.pt',"
        "        't3_mtl23ls_v2.safetensors',"
        "        's3gen.pt',"
        "        'grapheme_mtl_merged_expanded_v1.json',"
        "        'conds.pt',"
        "        'Cangjie5_TC.json',"
        "    ],"
        "    force_download=True,"
        "); "
        "print('Lahgtna model weights pre-downloaded')"
        "\"",
        secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=lahgtna_image,
    gpu="T4",
    scaledown_window=300,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class LahgtnaChatterboxModel(BaseTTSModel):
    """Lahgtna Chatterbox — Arabic dialect TTS (Egyptian, Saudi, Moroccan, Iraqi).

    Fine-tuned Chatterbox Multilingual for Arabic dialects.
    Source: https://huggingface.co/oddadmix/lahgtna-chatterbox-v0
    """

    model_id = "lahgtna"
    display_name = "Lahgtna"
    model_url = "https://huggingface.co/oddadmix/lahgtna-chatterbox-v0"
    gpu = "T4"

    @modal.enter()
    def load_model(self):
        """Load the Lahgtna Chatterbox model and verify reference audio."""
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self.model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
        if hasattr(self.model, "to"):
            self.model.to("cuda")
        self.sample_rate = self.model.sr

        # Verify all dialect reference audio files exist
        self._ref_dir = "/opt/lahgtna-chatterbox"
        for dialect_id, cfg in DIALECT_CONFIG.items():
            path = f"{self._ref_dir}/{cfg['audio_filename']}"
            import os
            if not os.path.exists(path):
                print(f"⚠️ Missing reference audio for {dialect_id}: {path}")

        print(f"✅ Lahgtna Chatterbox loaded on CUDA (sr={self.sample_rate})")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Synthesize Arabic text in a randomly selected dialect.

        Args:
            text: Arabic text to synthesize.
        """
        try:
            dialect = random.choice(ALL_DIALECTS)
            cfg = DIALECT_CONFIG[dialect]
            ref_audio = f"{self._ref_dir}/{cfg['audio_filename']}"

            print(f"[lahgtna] dialect={dialect} ({cfg['label']}), text={text[:60]}…")

            # Set seed for reproducible generation (matches demo default)
            import torch
            import numpy as np
            seed = 42
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            np.random.seed(seed)

            wav = self.model.generate(
                text,
                language_id=dialect,
                audio_prompt_path=ref_audio,
                exaggeration=0.5,
                temperature=0.8,
                cfg_weight=0.5,
                repetition_penalty=1.75,
            )

            wav_np = wav.squeeze().cpu().numpy()
            audio_base64 = self.audio_to_base64(wav_np, self.sample_rate)

            result = self.success_response(audio_base64, self.sample_rate)
            result["dialect"] = dialect
            return result

        except Exception as e:
            import traceback
            return self.error_response(f"{e}\n{traceback.format_exc()}")
