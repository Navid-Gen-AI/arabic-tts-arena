import modal
from typing import Optional
from models import BaseTTSModel, register_model
from app import app

moss_tts_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "libsndfile1", "espeak-ng", "git")
    .uv_pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy",
        "soundfile",
        "transformers>=4.36.0",
        "accelerate",
    )
    .run_commands(
        "pip install --upgrade pip && "
        "git clone https://github.com/OpenMOSS/MOSS-TTS.git /tmp/moss-tts && "
        "cd /tmp/moss-tts && pip install . && "
        "rm -rf /tmp/moss-tts"
    )
)


@register_model
@app.cls(
    image=moss_tts_image,
    gpu="A10G",  # 8B model needs more VRAM than a T4
    scaledown_window=300,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class MossTTSModel(BaseTTSModel):
    """OpenMOSS MOSS-TTS — 8B parameter multilingual TTS with Arabic support."""

    model_id = "moss_tts"
    display_name = "MOSS-TTS"
    model_url = "https://huggingface.co/OpenMOSS-Team/MOSS-TTS"

    @modal.enter()
    def load_model(self):
        """Load the MOSS-TTS model when container starts."""
        import torch
        from transformers import AutoModel, AutoProcessor

        # Required for MOSS-TTS inference
        torch.backends.cuda.enable_cudnn_sdp(False)

        model_name = "OpenMOSS-Team/MOSS-TTS"

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")

        # Get sample rate — try processor config first, then model config, then default
        if hasattr(self.processor, 'model_config') and hasattr(self.processor.model_config, 'sampling_rate'):
            self.sample_rate = self.processor.model_config.sampling_rate
        elif hasattr(self.model.config, 'sampling_rate'):
            self.sample_rate = self.model.config.sampling_rate
        else:
            self.sample_rate = 24000  # MOSS-TTS default
        print(f"✅ MOSS-TTS loaded on CUDA (sr={self.sample_rate})")

    @modal.method()
    def synthesize(self, text: str, speaker_wav: Optional[str] = None) -> dict:
        """Synthesize Arabic text to speech."""
        try:
            import torch
            import numpy as np

            # Build the input message for MOSS-TTS
            message = self.processor.build_user_message(text=text)
            inputs = self.processor.to_model_inputs(
                [message],
                return_tensors="pt",
            ).to("cuda")

            # Generate audio tokens
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=4096,
                    temperature=1.7,
                    top_p=0.8,
                    top_k=25,
                    repetition_penalty=1.0,
                )

            # Decode the generated tokens to audio
            result_message = self.processor.decode(outputs)
            audio_codes = result_message.audio_codes_list[0]

            # Convert audio codes to waveform
            wav = self.processor.codes_to_wav(audio_codes)

            if isinstance(wav, torch.Tensor):
                wav = wav.squeeze().cpu().numpy()
            elif not isinstance(wav, np.ndarray):
                wav = np.array(wav)

            audio_base64 = self.audio_to_base64(wav, self.sample_rate)

            return self.success_response(audio_base64, self.sample_rate)
        except Exception as e:
            return self.error_response(e)
