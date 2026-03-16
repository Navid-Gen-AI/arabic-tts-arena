import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES
import re
import unicodedata

# Reference transcript that matches the bundled reference.wav
_REF_TRANSCRIPT = "لَا يَمُرُّ يَوْمٌ إِلَّا وَأَسْتَقْبِلُ عِدَّةَ رَسَائِلَ، تَتَضَمَّنُ أَسْئِلَةً مُلِحَّةْ."

# Tashkeel model for adding diacritics to plain Arabic text
_TASHKEEL_MODEL_ID = "Abdou/arabic-tashkeel-flan-t5-small"

# Unicode ranges for Arabic diacritical marks (tashkeel)
_ARABIC_DIACRITICS = re.compile(
    "[\u0610-\u061A"   # Arabic signs
    "\u064B-\u065F"    # Arabic fathatan, dammatan, kasratan, fatha, damma, etc.
    "\u0670"           # Arabic letter superscript alef
    "\u06D6-\u06DC"    # Arabic small ligatures
    "\u06DF-\u06E4"    # Arabic small high signs
    "\u06E7-\u06E8"    # Arabic small high yeh/noon
    "\u06EA-\u06ED"    # Arabic small low signs
    "\uFE70-\uFE7F"   # Arabic presentation forms
    "]"
)


def strip_tashkeel(text: str) -> str:
    """Remove all Arabic diacritical marks from text."""
    return _ARABIC_DIACRITICS.sub("", text)


def clean_arabic_text(text: str) -> str:
    """Clean and normalize Arabic text before tashkeel.

    - Normalize unicode (NFC)
    - Strip existing diacritics so the model gets clean input
    - Collapse multiple spaces
    - Strip leading/trailing whitespace
    """
    # Normalize unicode
    text = unicodedata.normalize("NFC", text)
    # Strip any existing diacritics
    text = strip_tashkeel(text)
    # Replace various Arabic-specific whitespace / zero-width chars
    text = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF]", "", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


spark_tts_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.11",
    )
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(
        # Pin to versions matching the official Space
        "torch==2.1.0",
        "torchaudio==2.1.0",
        "transformers==4.46.2",
        "numpy==1.24.3",
        "soundfile==0.12.1",
        "einops==0.8.1",
        "einx==0.3.0",
        "accelerate==0.25.0",
        "huggingface_hub>=0.23.2",
        "sentencepiece",
        "soxr",
    )
    # Download the reference.wav from the official Space
    .run_commands(
        "python3 -c \""
        "from huggingface_hub import hf_hub_download; "
        "hf_hub_download("
        "    repo_id='IbrahimSalah/Arabic-TTS-Spark',"
        "    repo_type='space',"
        "    filename='reference.wav',"
        "    local_dir='/root/spark-ref'"
        ")\"",
        secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    )
    # Pre-download the tashkeel model so cold starts are fast
    .run_commands(
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
    image=spark_tts_image,
    gpu="T4",
    scaledown_window=300,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class SparkTTSModel(BaseTTSModel):
    """Arabic-TTS-Spark — SparkTTS fine-tuned for Arabic speech synthesis.

    Inference follows the official HF Space:
    https://huggingface.co/spaces/IbrahimSalah/Arabic-TTS-Spark
    """

    model_id = "spark_tts"
    display_name = "Arabic Spark TTS"
    model_url = "https://huggingface.co/IbrahimSalah/Arabic-TTS-Spark"
    gpu = "T4"

    @modal.enter()
    def load_model(self):
        """Load TTS model, processor, and tashkeel model."""
        from transformers import AutoModel, AutoProcessor
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_name = "IbrahimSalah/Arabic-TTS-Spark"

        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True,
        ).eval().to("cuda")

        # Link the model to the processor so it can call
        # tokenize_audio / detokenize_audio and sync sampling rates.
        # The official Space uses `processor.model = model` but
        # link_model() additionally validates the model interface and
        # synchronises processor.sampling_rate from the model config.
        self.processor.link_model(self.model)

        # Load tashkeel model on CPU (it's tiny — 75M params)
        self.tashkeel_tokenizer = AutoTokenizer.from_pretrained(_TASHKEEL_MODEL_ID)
        self.tashkeel_model = AutoModelForSeq2SeqLM.from_pretrained(
            _TASHKEEL_MODEL_ID,
        ).eval()  # stays on CPU

        self.ref_audio = "/root/spark-ref/reference.wav"
        self.ref_text = _REF_TRANSCRIPT
        self.sample_rate = 16000  # Model config specifies 16 kHz

        print(f"✅ Arabic Spark-TTS + tashkeel loaded on CUDA (sr={self.sample_rate})")

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
        """Synthesize Arabic text — cleans, adds tashkeel, then generates speech."""
        try:
            import torch
            import numpy as np
            import os

            # Verify reference audio exists
            if not os.path.exists(self.ref_audio):
                return self.error_response(
                    f"Reference audio not found: {self.ref_audio}"
                )

            # 1. Clean input text (normalize unicode, strip old diacritics, etc.)
            cleaned = clean_arabic_text(text)
            if not cleaned:
                return self.error_response("Input text is empty after cleaning")

            # 2. Add tashkeel (diacritics) — required by Spark TTS
            diacritized = self._add_tashkeel(cleaned)
            print(f"[spark_tts] original : {text[:80]}")
            print(f"[spark_tts] cleaned  : {cleaned[:80]}")
            print(f"[spark_tts] tashkeel : {diacritized[:80]}")

            # 3. Tokenize with reference audio + transcript (voice cloning)
            #    Official Space uses text.lower() — we use diacritized text
            inputs = self.processor(
                text=diacritized.lower(),
                prompt_speech_path=self.ref_audio,
                prompt_text=self.ref_text,
                return_tensors="pt",
            ).to("cuda")

            global_tokens_prompt = inputs.pop("global_token_ids_prompt", None)
            input_ids_len = inputs["input_ids"].shape[-1]

            print(f"[spark_tts] input_ids shape: {inputs['input_ids'].shape}")

            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=8000,
                    do_sample=True,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )

            print(f"[spark_tts] output_ids shape: {output_ids.shape}")

            # Decode to audio
            # CRITICAL: skip_special_tokens must be False so that bicodec
            # tokens (e.g. <|bicodec_semantic_1234|>) are preserved in the
            # decoded text.  The processor.decode() regex-parses them to
            # reconstruct the waveform.  With the default (True) the
            # tokenizer strips them → almost no semantic tokens → silence.
            output = self.processor.decode(
                generated_ids=output_ids,
                global_token_ids_prompt=global_tokens_prompt,
                input_ids_len=input_ids_len,
                skip_special_tokens=False,
            )

            print(f"[spark_tts] decode keys: {list(output.keys())}")

            audio = output["audio"]
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()

            sr = output.get("sampling_rate", self.sample_rate)
            self.sample_rate = sr

            if not isinstance(audio, np.ndarray):
                audio = np.asarray(audio, dtype=np.float32)
            audio = audio.astype(np.float32, copy=False)

            if audio.ndim > 1:
                audio = audio.reshape(-1)

            print(f"[spark_tts] raw audio: len={audio.size}, "
                  f"min={audio.min():.6f}, max={audio.max():.6f}, "
                  f"rms={np.sqrt(np.mean(audio**2)):.6f}")

            # Normalize audio to target RMS (official Space does this)
            target_rms = 0.1
            current_rms = np.sqrt(np.mean(audio ** 2))
            if current_rms > 1e-6:
                audio = audio * (target_rms / current_rms)

            if audio.size < 100:
                return self.error_response(
                    f"Audio too short: {audio.size} samples"
                )

            audio_base64 = self.audio_to_base64(audio, sr)
            return self.success_response(audio_base64, sr)

        except Exception as e:
            import traceback
            return self.error_response(f"{e}\n{traceback.format_exc()}")
