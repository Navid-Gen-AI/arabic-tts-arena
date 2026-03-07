import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

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
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=moss_tts_image,
    gpu="A10G",  # 8B model needs more VRAM than a T4
    scaledown_window=300,
    retries=0,
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
    def synthesize(self, text: str) -> dict:
        """Synthesize Arabic text to speech."""
        try:
            import torch
            import numpy as np

            # MOSS-TTS expects a conversation-format input
            conversations = [
                {"role": "user", "content": text},
            ]

            inputs = self.processor(
                conversations=conversations,
                return_tensors="pt",
            ).to("cuda")

            # Generate — returns a list of AssistantMessage objects
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,
                )

            # outputs is a list — get the first message
            message = outputs[0] if isinstance(outputs, list) else outputs

            # Debug: log what the message looks like
            print(f"  message type: {type(message)}")
            if hasattr(message, '__dict__'):
                print(f"  message attrs: {list(message.__dict__.keys())}")

            # Extract audio codes from the message and decode to waveform
            wav = None

            if hasattr(message, 'audio_codes_list') and message.audio_codes_list:
                wav = self.processor.codes_to_wav(message.audio_codes_list[0])
            elif hasattr(message, 'audio_codes'):
                wav = self.processor.codes_to_wav(message.audio_codes)
            elif hasattr(message, 'audio'):
                wav = message.audio
            elif hasattr(message, 'content'):
                # content might be the audio directly or contain audio parts
                content = message.content
                if isinstance(content, (torch.Tensor, np.ndarray)):
                    wav = content
                elif isinstance(content, list):
                    for part in content:
                        if hasattr(part, 'audio_codes_list') and part.audio_codes_list:
                            wav = self.processor.codes_to_wav(part.audio_codes_list[0])
                            break
                        if hasattr(part, 'audio'):
                            wav = part.audio
                            break
            else:
                return self.error_response(
                    f"Cannot extract audio from {type(message).__name__}. "
                    f"Attrs: {[a for a in dir(message) if not a.startswith('_')]}"
                )

            if wav is None:
                return self.error_response(
                    f"Audio extraction returned None from {type(message).__name__}. "
                    f"Attrs: {[a for a in dir(message) if not a.startswith('_')]}"
                )

            if isinstance(wav, torch.Tensor):
                wav = wav.squeeze().cpu().float().numpy()
            elif not isinstance(wav, np.ndarray):
                wav = np.array(wav, dtype=np.float32)

            audio_base64 = self.audio_to_base64(wav, self.sample_rate)
            return self.success_response(audio_base64, self.sample_rate)
        except Exception as e:
            return self.error_response(e)
