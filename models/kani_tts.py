import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

# KaniTTS Arabic — dedicated Arabic model, 400M params, 22kHz output.
# Uses the lightweight `kani-tts` pip package which bundles the NeMo codec
# decoder internally, so we don't need the heavy `nemo_toolkit`.
# Ref: https://huggingface.co/nineninesix/kani-tts-400m-ar

# Tashkeel model for adding diacritics (same one used by Spark TTS)
_TASHKEEL_MODEL_ID = "Abdou/arabic-tashkeel-flan-t5-small"

kani_tts_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "libsndfile1", "g++")  # g++ needed by texterrors (nemo dep)
    .env({"CC": "gcc", "CXX": "g++"})  # Force g++ over clang++ for texterrors build
    .uv_pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy",
        "soundfile",
        "huggingface_hub",
        "kani-tts",
        "sentencepiece",
    )
    # kani-tts 1.0.1 needs TransformersKwargs (transformers>=4.53) but its
    # nemo-toolkit dep pins transformers<=4.52. Force-upgrade after install.
    .run_commands("pip install 'transformers>=4.53.0'")
    # Pre-download the Arabic model weights + tashkeel model
    .run_commands(
        "python3 -c \""
        "from kani_tts import KaniTTS; "
        "KaniTTS('nineninesix/kani-tts-400m-ar', suppress_logs=True)"
        "\"",
        # Pre-download tashkeel model
        "python3 -c \""
        "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; "
        "AutoTokenizer.from_pretrained('Abdou/arabic-tashkeel-flan-t5-small'); "
        "AutoModelForSeq2SeqLM.from_pretrained('Abdou/arabic-tashkeel-flan-t5-small')"
        "\"",
        secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=kani_tts_image,
    gpu="T4",
    scaledown_window=300,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class KaniTTSModel(BaseTTSModel):
    """KaniTTS Arabic — fast Arabic-specific TTS (400M params, 22kHz).

    Zero-shot model (no reference audio needed). Benefits greatly from
    diacritized (tashkeel) input text.

    Source: https://huggingface.co/nineninesix/kani-tts-400m-ar
    """

    model_id = "kani_tts"
    display_name = "KaniTTS Arabic"
    model_url = "https://huggingface.co/nineninesix/kani-tts-400m-ar"

    @modal.enter()
    def load_model(self):
        """Load the KaniTTS Arabic model and tashkeel model."""
        from kani_tts import KaniTTS
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        self.tts = KaniTTS(
            "nineninesix/kani-tts-400m-ar",
            max_new_tokens=1200,
            suppress_logs=True,
        )
        self.sample_rate = self.tts.sample_rate  # 22050

        # Load tashkeel model on CPU (tiny — ~75M params)
        self.tashkeel_tokenizer = AutoTokenizer.from_pretrained(_TASHKEEL_MODEL_ID)
        self.tashkeel_model = AutoModelForSeq2SeqLM.from_pretrained(
            _TASHKEEL_MODEL_ID,
        ).eval()  # stays on CPU

        print(f"✅ KaniTTS Arabic + tashkeel loaded (sr={self.sample_rate})")

    def _add_tashkeel(self, text: str, max_length: int = 256) -> str:
        """Add diacritics to plain Arabic text using the tashkeel model."""
        import torch

        inputs = self.tashkeel_tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        with torch.no_grad():
            outputs = self.tashkeel_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
            )
        return self.tashkeel_tokenizer.decode(outputs[0], skip_special_tokens=True)

    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Synthesize Arabic text to speech.

        Adds tashkeel (diacritics) for better pronunciation, then generates.
        """
        try:
            import numpy as np
            import re
            import unicodedata

            # Clean input text
            text = unicodedata.normalize("NFC", text).strip()
            if not text:
                return self.error_response("Input text is empty after cleaning")

            # Strip existing diacritics, then re-add via tashkeel model
            stripped = re.sub(
                r"[\u064B-\u065F\u0670]", "", text
            )
            diacritized = self._add_tashkeel(stripped)
            print(f"[kani_tts] original : {text[:80]}")
            print(f"[kani_tts] tashkeel : {diacritized[:80]}")

            # Generate audio (sampling params are per-call in kani-tts ≥1.0)
            audio, _ = self.tts(
                diacritized,
                temperature=0.6,
                top_p=0.95,
                repetition_penalty=1.1,
            )

            if not isinstance(audio, np.ndarray):
                audio = np.asarray(audio, dtype=np.float32)
            audio = audio.astype(np.float32, copy=False)

            if audio.ndim > 1:
                audio = audio.reshape(-1)

            if audio.size < 100:
                return self.error_response(
                    f"Audio too short: {audio.size} samples"
                )

            audio_base64 = self.audio_to_base64(audio, self.sample_rate)
            return self.success_response(audio_base64, self.sample_rate)

        except Exception as e:
            import traceback
            return self.error_response(f"{e}\n{traceback.format_exc()}")
