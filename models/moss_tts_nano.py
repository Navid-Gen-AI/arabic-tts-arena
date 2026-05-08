import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

MODEL_NAME = "OpenMOSS-Team/MOSS-TTS-Nano-100M"
AUDIO_TOKENIZER_NAME = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
PROMPT_AUDIO_PATH = "/root/audio-assets/msa-ar.wav"

moss_tts_nano_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "libsndfile1", "git")
    .uv_pip_install(
        "torch==2.7.0",
        "torchaudio==2.7.0",
        "transformers==4.57.1",
        "numpy>=1.24",
        "soundfile",
        "sentencepiece>=0.1.99",
        "accelerate",
        "huggingface_hub",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/OpenMOSS/MOSS-TTS-Nano.git /opt/MOSS-TTS-Nano",
        "cd /opt/MOSS-TTS-Nano && pip install --no-deps -e .",
    )
    .run_commands(
        "python3 -c \""
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download('{MODEL_NAME}'); "
        f"snapshot_download('{AUDIO_TOKENIZER_NAME}')"
        "\"",
        secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    )
    .add_local_file("audio-assets/msa-ar.wav", PROMPT_AUDIO_PATH)
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=moss_tts_nano_image,
    gpu="T4",
    scaledown_window=60,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    timeout=600,
)
class MossTTSNanoModel(BaseTTSModel):
    """MOSS-TTS Nano -- tiny multilingual TTS with Arabic support."""

    model_id = "moss_tts_nano"
    display_name = "MOSS-TTS Nano"
    model_url = "https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M"
    gpu = "T4"

    @staticmethod
    def _waveform_to_soundfile_array(wav):
        """Match upstream MOSS-TTS-Nano channel handling for WAV writing."""
        import numpy as np

        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 1:
            return wav
        if wav.ndim != 2:
            raise ValueError(f"Unsupported waveform shape: {wav.shape}")
        if wav.shape[0] <= 8 and wav.shape[0] < wav.shape[1]:
            wav = wav.T
        return wav

    @modal.enter()
    def load_model(self):
        """Load MOSS-TTS-Nano and use the bundled Arabic prompt voice."""
        import sys
        import torch
        from transformers import AutoModelForCausalLM

        if "/opt/MOSS-TTS-Nano" not in sys.path:
            sys.path.insert(0, "/opt/MOSS-TTS-Nano")

        dtype = torch.bfloat16
        if not torch.cuda.is_available():
            dtype = torch.float32
        elif not torch.cuda.is_bf16_supported():
            dtype = torch.float16

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )
        self.model.to(device=self.device, dtype=dtype)
        if hasattr(self.model, "_set_attention_implementation"):
            self.model._set_attention_implementation("sdpa")
        self.model.eval()
        self.sample_rate = 48000

        print(
            f"MOSS-TTS Nano loaded on {self.device} "
            f"(prompt={PROMPT_AUDIO_PATH}, sr={self.sample_rate})"
        )

    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Synthesize Arabic speech using the bundled Arabic reference audio."""
        try:
            import numpy as np
            import torch

            text = text.strip()
            if not text:
                return self.error_response("Input text is empty")

            import os
            import uuid

            output_audio_path = f"/tmp/moss_tts_nano_{uuid.uuid4().hex}.wav"

            with torch.inference_mode():
                result = self.model.inference(
                    text=text,
                    output_audio_path=output_audio_path,
                    mode="voice_clone",
                    prompt_audio_path=PROMPT_AUDIO_PATH,
                    text_tokenizer_path=MODEL_NAME,
                    audio_tokenizer_type="moss-audio-tokenizer-nano",
                    audio_tokenizer_pretrained_name_or_path=AUDIO_TOKENIZER_NAME,
                    device=self.device,
                    max_new_frames=375,
                    voice_clone_max_text_tokens=75,
                    do_sample=True,
                    use_kv_cache=True,
                    text_temperature=1.0,
                    text_top_p=1.0,
                    text_top_k=50,
                    audio_temperature=0.8,
                    audio_top_p=0.95,
                    audio_top_k=25,
                    audio_repetition_penalty=1.2,
                )

            wav = result.get("waveform_numpy")
            if wav is None:
                wav = result.get("waveform")
            if wav is None:
                return self.error_response("MOSS-TTS Nano did not return waveform audio")

            if isinstance(wav, torch.Tensor):
                wav = wav.detach().float().cpu().numpy()
            wav = self._waveform_to_soundfile_array(wav)

            if wav.size < 100:
                return self.error_response(f"Audio too short: {wav.size} samples")

            sample_rate = int(result.get("sample_rate", self.sample_rate))
            audio_base64 = self.audio_to_base64(wav, sample_rate)
            if os.path.exists(output_audio_path):
                os.remove(output_audio_path)
            return self.success_response(audio_base64, sample_rate)

        except Exception as e:
            import traceback
            try:
                if "output_audio_path" in locals() and os.path.exists(output_audio_path):
                    os.remove(output_audio_path)
            except Exception:
                pass
            return self.error_response(f"{e}\n{traceback.format_exc()}")