"""
Lahgatna 2.0 — Arabic Dialect-Aware TTS with Auto-Detection
=============================================================

Faithfully implements the official inference pipeline from:
    https://github.com/Oddadmix/lahgtna-chatterbox

Key improvements over Lahgtna v1:
    1. **Dialect Router** — uses the oddadmix/dialect-router-v0.1 classifier
       to auto-detect the Arabic dialect of the input text (Egyptian, Saudi,
       Moroccan, Iraqi, Sudanese, Tunisian, Lebanese, Syrian, Libyan,
       Palestinian, or MSA) rather than random sampling.
    2. **MSA (ar) fallback** — supports Modern Standard Arabic as an 11th
       dialect, matching the official config.
    3. **Accurate generation defaults** — repetition_penalty=2.0, min_p=0.05,
       top_p=1.0 as in the official `generate()` method.
    4. **Fine-tuned T3 weight overlay** — follows the official TTSEngine
       pattern: load base model, then overlay the fine-tuned T3 weights.

The dialect-to-Chatterbox-language mapping uses repurposed ISO 639-1 codes
(e.g. Egyptian → "ms", Saudi → "sv"). This is intentional — the Chatterbox
backbone was originally trained on multilingual data and the Arabic
fine-tuning overrides those language slots.

Source: https://huggingface.co/oddadmix/lahgtna-chatterbox-v1
Router: https://huggingface.co/oddadmix/dialect-router-v0.1
"""

import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

# ---------------------------------------------------------------------------
# Dialect → Chatterbox language code + reference audio mapping
# ---------------------------------------------------------------------------
# Mirrors config.LANGUAGE_CODES from the official repo exactly.
# "code" = the Chatterbox backbone's internal language identifier.
# "ref"  = filename of the reference audio bundled from the HF Space.
# "label" = human-readable dialect name.

LANGUAGE_CODES = {
    "eg": {"code": "ms", "ref": "fdddf1f2a317e626a0db94d6d7d92e772039121f120787c80be2250c09999ec1.wav", "label": "Egyptian"},
    "sa": {"code": "sv", "ref": "saudi-ref.wav", "label": "Saudi"},
    "mo": {"code": "pl", "ref": "mor-ref.wav", "label": "Moroccan"},
    "iq": {"code": "no", "ref": "iraqi-ref.wav", "label": "Iraqi"},
    "sd": {"code": "pt", "ref": "fdddf1f2a317e626a0db94d6d7d92e772039121f120787c80be2250c09999ec1.wav", "label": "Sudanese"},
    "tn": {"code": "da", "ref": "tun-ref.wav", "label": "Tunisian"},
    "lb": {"code": "nl", "ref": "leb-ref.wav", "label": "Lebanese"},
    "sy": {"code": "ko", "ref": "syrian-ref.wav", "label": "Syrian"},
    "ly": {"code": "sw", "ref": "lib-ref.wav", "label": "Libyan"},
    "ps": {"code": "he", "ref": "pal-ref.wav", "label": "Palestinian"},
    "ar": {"code": "ar", "ref": "fdddf1f2a317e626a0db94d6d7d92e772039121f120787c80be2250c09999ec1.wav", "label": "MSA"},
}

ALL_DIALECTS = list(LANGUAGE_CODES.keys())

# Official generation defaults from config.DEFAULT_GENERATION_KWARGS
DEFAULT_GENERATION_KWARGS = {
    "exaggeration": 0.5,
    "temperature": 0.8,
    "cfg_weight": 0.5,
}

# ---------------------------------------------------------------------------
# Model / Router repos
# ---------------------------------------------------------------------------
MODEL_REPO_ID = "oddadmix/lahgtna-chatterbox-v1"
ROUTER_MODEL_ID = "oddadmix/dialect-router-v0.1"

SNAPSHOT_PATTERNS = [
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
]

# ---------------------------------------------------------------------------
# Modal image
# ---------------------------------------------------------------------------

lahgtna_v2_image = (
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
    .env({"LAHGTNA_V2_CACHE_BUST": "2025-07-18-v2"})
    .run_commands(
        # 1. Download the chatterbox Python module (src/chatterbox/*)
        #    from the official Lahgtna fork — includes dialect-specific
        #    text normalisers (Egyptian, Saudi, Moroccan, etc.)
        "python3 -c \""
        "from huggingface_hub import snapshot_download; "
        "snapshot_download("
        "    repo_id='oddadmix/lahgtna-chatterbox-demo',"
        "    repo_type='space',"
        "    allow_patterns=['src/chatterbox/**'],"
        "    local_dir='/opt/lahgtna-chatterbox'"
        ")\"",
        # Copy into site-packages
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
        "    'leb-ref.wav',"
        "    'lib-ref.wav',"
        "    'syrian-ref.wav',"
        "    'tun-ref.wav',"
        "    'pal-ref.wav',"
        "]; "
        "[hf_hub_download("
        "    repo_id='oddadmix/lahgtna-chatterbox-demo',"
        "    repo_type='space',"
        "    filename=w,"
        "    local_dir='/opt/lahgtna-chatterbox'"
        ") for w in wavs]; "
        "print('All Lahgtna reference WAVs downloaded')"
        "\"",
        # 3. Pre-download the fine-tuned Arabic model weights
        "python3 -c \""
        "from huggingface_hub import snapshot_download; "
        "snapshot_download("
        "    repo_id='oddadmix/lahgtna-chatterbox-v1',"
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
        # 4. Pre-download the dialect-router classifier
        "python3 -c \""
        "from transformers import AutoModelForSequenceClassification, AutoTokenizer; "
        "AutoTokenizer.from_pretrained('oddadmix/dialect-router-v0.1'); "
        "AutoModelForSequenceClassification.from_pretrained('oddadmix/dialect-router-v0.1'); "
        "print('Dialect router model pre-downloaded')"
        "\"",
        secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    )
    .add_local_python_source(*LOCAL_MODULES)
)


# ---------------------------------------------------------------------------
# Dialect Router (mirrors official inference.py DialectRouter)
# ---------------------------------------------------------------------------

class DialectRouter:
    """
    Thin wrapper around the dialect-classification model.

    Maps Arabic text → one of the dialect codes defined in LANGUAGE_CODES.
    Falls back to "ar" (Modern Standard Arabic) when prediction fails.

    Source: https://github.com/Oddadmix/lahgtna-chatterbox/blob/main/src/inference.py
    """

    FALLBACK = "ar"

    def __init__(self, model_id: str = ROUTER_MODEL_ID, device: str = "cuda"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device).eval()

    def predict(self, text: str) -> str:
        """Return the predicted dialect code for *text*."""
        import torch

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            pred_id = int(torch.argmax(logits, dim=-1).item())
            label: str = self.model.config.id2label[pred_id]
            if label not in LANGUAGE_CODES:
                print(
                    f"⚠️ [dialect-router] Predicted label '{label}' not in "
                    f"LANGUAGE_CODES — falling back to '{self.FALLBACK}'."
                )
                return self.FALLBACK
            return label
        except Exception as e:
            print(f"⚠️ [dialect-router] Prediction failed ({e}); "
                  f"defaulting to '{self.FALLBACK}'.")
            return self.FALLBACK


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

@register_model
@app.cls(
    image=lahgtna_v2_image,
    gpu="T4",
    scaledown_window=120,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class LahgtnaV2Model(BaseTTSModel):
    """Lahgatna 2.0 — Arabic dialect-aware TTS with auto-detection.

    Uses the official dialect router to automatically detect the Arabic
    dialect of the input text, then synthesises speech with the matching
    reference audio and language code.

    Fine-tuned Chatterbox Multilingual for 10 Arabic dialects + MSA.
    Source: https://github.com/Oddadmix/lahgtna-chatterbox
    """

    model_id = "lahgtna_v2"
    display_name = "Lahgatna 2.0"
    model_url = "https://huggingface.co/oddadmix/lahgtna-chatterbox-v1"
    gpu = "T4"

    @modal.enter()
    def load_model(self):
        """Load the TTS model and dialect router.

        Follows the official TTSEngine pattern:
        1. Load base ChatterboxMultilingualTTS via from_pretrained()
        2. Overlay fine-tuned T3 weights from oddadmix/lahgtna-chatterbox-v1
        3. Load the dialect-router classifier
        """
        import os
        from pathlib import Path
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file as load_safetensors
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        # --- Load TTS model (official TTSEngine._load_model pattern) ---
        self.model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

        # Overlay the fine-tuned T3 weights
        ckpt_dir = Path(snapshot_download(
            repo_id=MODEL_REPO_ID,
            repo_type="model",
            revision="main",
            allow_patterns=SNAPSHOT_PATTERNS,
            token=os.getenv("HF_TOKEN"),
        ))
        t3_path = ckpt_dir / "t3_mtl23ls_v2.safetensors"
        print(f"Loading fine-tuned T3 weights from {t3_path} …")
        t3_state = load_safetensors(str(t3_path), device="cuda")
        self.model.t3.load_state_dict(t3_state)
        self.model.t3.to("cuda").eval()

        if hasattr(self.model, "to"):
            self.model.to("cuda")
        self.sample_rate = self.model.sr

        # --- Load dialect router ---
        self.router = DialectRouter(device="cuda")

        # Verify reference audio files exist
        self._ref_dir = "/opt/lahgtna-chatterbox"
        for dialect_id, cfg in LANGUAGE_CODES.items():
            path = f"{self._ref_dir}/{cfg['ref']}"
            if not os.path.exists(path):
                print(f"⚠️ Missing reference audio for {dialect_id}: {path}")

        print(f"✅ Lahgatna 2.0 loaded on CUDA (sr={self.sample_rate}, "
              f"dialects={len(LANGUAGE_CODES)}, router=dialect-router-v0.1)")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Synthesise Arabic text with automatic dialect detection.

        Pipeline (mirrors official inference.py run_pipeline):
            1. Run the dialect router to auto-detect dialect
            2. Look up the Chatterbox language code + reference audio
            3. Generate speech via ChatterboxMultilingualTTS.generate()

        Args:
            text: Arabic text to synthesise.
        """
        try:
            # --- Step 1: Auto-detect dialect ---
            language_id = self.router.predict(text)
            print(f"[lahgtna_v2] Auto-detected dialect: {language_id} "
                  f"({LANGUAGE_CODES[language_id]['label']})")

            # --- Step 2: Look up config ---
            lang_cfg = LANGUAGE_CODES[language_id]
            ref_audio = f"{self._ref_dir}/{lang_cfg['ref']}"
            language_code = lang_cfg["code"]

            print(f"[lahgtna_v2] dialect={language_id} ({lang_cfg['label']}), "
                  f"code={language_code}, text={text[:60]}…")

            # --- Step 3: Generate (official generate() signature) ---
            wav = self.model.generate(
                text,
                language_id=language_code,
                audio_prompt_path=ref_audio,
                exaggeration=DEFAULT_GENERATION_KWARGS["exaggeration"],
                temperature=DEFAULT_GENERATION_KWARGS["temperature"],
                cfg_weight=DEFAULT_GENERATION_KWARGS["cfg_weight"],
                repetition_penalty=2.0,
                min_p=0.05,
                top_p=1.0,
            )

            wav_np = wav.squeeze().cpu().numpy()
            audio_base64 = self.audio_to_base64(wav_np, self.sample_rate)

            result = self.success_response(audio_base64, self.sample_rate)
            result["dialect"] = language_id
            result["dialect_label"] = lang_cfg["label"]
            return result

        except Exception as e:
            import traceback
            return self.error_response(f"{e}\n{traceback.format_exc()}")
