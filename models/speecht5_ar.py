import modal
import re
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

# MBZUAI SpeechT5 Arabic — SpeechT5 fine-tuned on ClArTTS (Classical Arabic).
# Uses pre-computed x-vector speaker embeddings instead of reference audio.
# IMPORTANT: Model was trained WITHOUT diacritics — strip tashkeel before inference.
# Ref: https://huggingface.co/MBZUAI/speecht5_tts_clartts_ar

_SPEAKER_EMBEDDING_IDX = 105  # Default speaker from the model card

# Unicode ranges for Arabic diacritical marks
_ARABIC_DIACRITICS = re.compile(
    r"[\u064B-\u065F\u0670]"
)

speecht5_ar_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy",
        "soundfile",
        "huggingface_hub",
        "transformers",
        "sentencepiece",
        "datasets[audio]",
    )
    # Pre-download all models and the speaker embeddings dataset
    .run_commands(
        "python3 -c \""
        "from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan; "
        "SpeechT5Processor.from_pretrained('MBZUAI/speecht5_tts_clartts_ar'); "
        "SpeechT5ForTextToSpeech.from_pretrained('MBZUAI/speecht5_tts_clartts_ar'); "
        "SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')"
        "\"",
        # Pre-download speaker embeddings
        "python3 -c \""
        "from datasets import load_dataset; "
        "load_dataset('herwoww/arabic_xvector_embeddings', split='validation')"
        "\"",
        secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=speecht5_ar_image,
    gpu="T4",
    scaledown_window=300,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class SpeechT5ArabicModel(BaseTTSModel):
    """MBZUAI SpeechT5 Arabic — SpeechT5 fine-tuned on Classical Arabic (ClArTTS).

    Lightweight model that uses pre-computed x-vector speaker embeddings
    instead of reference audio. Trained without diacritics.

    Source: https://huggingface.co/MBZUAI/speecht5_tts_clartts_ar
    """

    model_id = "speecht5_ar"
    display_name = "SpeechT5 Arabic"
    model_url = "https://huggingface.co/MBZUAI/speecht5_tts_clartts_ar"

    @modal.enter()
    def load_model(self):
        """Load SpeechT5 model, vocoder, and speaker embeddings."""
        import torch
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        from datasets import load_dataset

        model_name = "MBZUAI/speecht5_tts_clartts_ar"

        self.processor = SpeechT5Processor.from_pretrained(model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_name).to("cuda")
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan"
        ).to("cuda")

        # Load pre-computed speaker embeddings
        embeddings_dataset = load_dataset(
            "herwoww/arabic_xvector_embeddings", split="validation"
        )
        self.speaker_embedding = torch.tensor(
            embeddings_dataset[_SPEAKER_EMBEDDING_IDX]["speaker_embeddings"]
        ).unsqueeze(0).to("cuda")

        self.sample_rate = 16000  # SpeechT5 outputs at 16kHz

        print(f"✅ SpeechT5 Arabic loaded on CUDA (sr={self.sample_rate})")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Synthesize Arabic text — strips diacritics then generates speech."""
        try:
            import numpy as np
            import unicodedata

            # Clean and normalize
            text = unicodedata.normalize("NFC", text).strip()
            if not text:
                return self.error_response("Input text is empty")

            # Strip diacritics — model was trained without tashkeel
            text_clean = _ARABIC_DIACRITICS.sub("", text)
            print(f"[speecht5_ar] original: {text[:80]}")
            print(f"[speecht5_ar] cleaned : {text_clean[:80]}")

            inputs = self.processor(text=text_clean, return_tensors="pt").to("cuda")

            speech = self.model.generate_speech(
                inputs["input_ids"],
                self.speaker_embedding,
                vocoder=self.vocoder,
            )

            audio = speech.cpu().numpy()

            if not isinstance(audio, np.ndarray):
                audio = np.asarray(audio, dtype=np.float32)
            audio = audio.astype(np.float32, copy=False)

            if audio.ndim > 1:
                audio = audio.reshape(-1)

            if audio.size < 100:
                return self.error_response(
                    f"Audio too short: {audio.size} samples"
                )

            print(f"[speecht5_ar] audio: len={audio.size}, sr={self.sample_rate}")

            audio_base64 = self.audio_to_base64(audio, self.sample_rate)
            return self.success_response(audio_base64, self.sample_rate)

        except Exception as e:
            import traceback
            return self.error_response(f"{e}\n{traceback.format_exc()}")
