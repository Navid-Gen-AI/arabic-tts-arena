import modal
import re
import unicodedata
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

# Arabic-F5-TTS v2 — F5-TTS fine-tuned for Arabic (similar to Habibi TTS
# but uses a custom fork and different checkpoint/vocab).
# IMPORTANT: Diacritized text (تشكيل) is REQUIRED for this model.
# Ref: https://huggingface.co/spaces/IbrahimSalah/Arabic-F5-TTS

_REF_TRANSCRIPT = "لَا يَمُرُّ يَوْمٌ إِلَّا وَأَسْتَقْبِلُ عِدَّةَ رَسَائِلَ، تَتَضَمَّنُ أَسْئِلَةً مُلِحَّةْ."

# Tashkeel model for adding diacritics
_TASHKEEL_MODEL_ID = "Abdou/arabic-tashkeel-flan-t5-small"

# Unicode ranges for Arabic diacritical marks
_ARABIC_DIACRITICS = re.compile(
    r"[\u064B-\u065F\u0670]"
)

arabic_f5_tts_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "libsndfile1", "git")
    .uv_pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "torchcodec",
        "numpy",
        "soundfile",
        "huggingface_hub[hf_xet]",
        # Custom F5-TTS fork with Arabic support
        "f5-tts @ git+https://github.com/ibrahimabdelaal/F5-TTS-Arabic.git",
        "cached-path",
        "sentencepiece",
        "omegaconf",
        "transformers",
    )
    # Pre-download model files + reference audio + tashkeel model
    .run_commands(
        "python3 -c \""
        "from huggingface_hub import hf_hub_download; "
        "hf_hub_download(repo_id='IbrahimSalah/Arabic-F5-TTS-v2', filename='vocab.txt', local_dir='/root/arabic-f5'); "
        "hf_hub_download(repo_id='IbrahimSalah/Arabic-F5-TTS-v2', filename='model_547500_8_18.pt', local_dir='/root/arabic-f5'); "
        "hf_hub_download(repo_id='IbrahimSalah/Arabic-F5-TTS-v2', filename='F5TTS_Base_8_18.yaml', local_dir='/root/arabic-f5'); "
        "print('Arabic F5-TTS v2 files downloaded')"
        "\"",
        # Download reference audio from the Arabic-F5-TTS Space
        "python3 -c \""
        "from huggingface_hub import hf_hub_download; "
        "hf_hub_download("
        "    repo_id='IbrahimSalah/Arabic-F5-TTS',"
        "    repo_type='space',"
        "    filename='reference.wav',"
        "    local_dir='/root/arabic-f5-ref'"
        ")\"",
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
    image=arabic_f5_tts_image,
    gpu="T4",
    scaledown_window=120,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class ArabicF5TTSModel(BaseTTSModel):
    """Arabic-F5-TTS v2 — F5-TTS fine-tuned for Arabic speech synthesis.

    Requires diacritized (tashkeel) text input. Uses voice cloning with
    reference audio + transcript.

    Source: https://huggingface.co/IbrahimSalah/Arabic-F5-TTS-v2
    """

    model_id = "arabic_f5_tts"
    display_name = "Arabic F5-TTS"
    model_url = "https://huggingface.co/IbrahimSalah/Arabic-F5-TTS-v2"
    gpu = "T4"

    @modal.enter()
    def load_model(self):
        """Load Arabic F5-TTS v2 model, vocoder, and tashkeel model."""
        from f5_tts.infer.utils_infer import load_model, load_vocoder
        from f5_tts.model import DiT
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        # Load model config from YAML
        config_path = "/root/arabic-f5/F5TTS_Base_8_18.yaml"
        config = OmegaConf.load(config_path)

        # Extract full model arch config — matches how infer_cli.py uses
        # model_arc = model_cfg.model.arch directly
        from omegaconf import OmegaConf as _OC
        model_cfg = _OC.to_container(config.model.arch, resolve=True)

        ckpt_file = "/root/arabic-f5/model_547500_8_18.pt"
        vocab_file = "/root/arabic-f5/vocab.txt"

        print(f"  ckpt file : {ckpt_file}")
        print(f"  vocab file: {vocab_file}")
        print(f"  config    : {model_cfg}")

        self.ema_model = load_model(
            DiT,
            model_cfg,
            ckpt_file,
            mel_spec_type="vocos",
            vocab_file=vocab_file,
            device="cuda",
        )

        self.vocoder = load_vocoder(vocoder_name="vocos", device="cuda")

        # Load tashkeel model on CPU
        self.tashkeel_tokenizer = AutoTokenizer.from_pretrained(_TASHKEEL_MODEL_ID)
        self.tashkeel_model = AutoModelForSeq2SeqLM.from_pretrained(
            _TASHKEEL_MODEL_ID,
        ).eval()

        self.sample_rate = 24000
        self._ref_audio = "/root/arabic-f5-ref/reference.wav"
        self._ref_text = _REF_TRANSCRIPT

        print(f"✅ Arabic F5-TTS v2 + tashkeel loaded on CUDA (sr={self.sample_rate})")

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
            from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

            # Clean and normalize
            text = unicodedata.normalize("NFC", text).strip()
            if not text:
                return self.error_response("Input text is empty after cleaning")

            # Strip existing diacritics, then re-add via tashkeel model
            stripped = _ARABIC_DIACRITICS.sub("", text)
            diacritized = self._add_tashkeel(stripped)
            print(f"[arabic_f5_tts] original : {text[:80]}")
            print(f"[arabic_f5_tts] tashkeel : {diacritized[:80]}")

            # Preprocess reference audio
            ref_audio, ref_text = preprocess_ref_audio_text(
                self._ref_audio, self._ref_text, show_info=print
            )

            # Run inference
            wav, sr, _ = infer_process(
                ref_audio,
                ref_text,
                diacritized,
                self.ema_model,
                self.vocoder,
                mel_spec_type="vocos",
                speed=1.0,
                nfe_step=32,
                cfg_strength=1.8,
                device="cuda",
            )

            audio_base64 = self.audio_to_base64(wav, sr)
            return self.success_response(audio_base64, sr)

        except Exception as e:
            import traceback
            return self.error_response(f"{e}\n{traceback.format_exc()}")
