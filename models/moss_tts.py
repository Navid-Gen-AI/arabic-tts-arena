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

            # Generate — returns an AssistantMessage-like object
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,
                )

            # Debug: log what we got back so we can trace the API
            print(f"  generate() returned: {type(outputs)}")
            if hasattr(outputs, '__dict__'):
                print(f"  attrs: {list(outputs.__dict__.keys())}")

            # Extract audio from the output — handle multiple API shapes
            wav = None

            # Shape 1: output has .audio attribute directly
            if hasattr(outputs, 'audio'):
                wav = outputs.audio

            # Shape 2: output has .audio_codes / .audio_codes_list
            elif hasattr(outputs, 'audio_codes'):
                wav = self.processor.codes_to_wav(outputs.audio_codes)
            elif hasattr(outputs, 'audio_codes_list'):
                wav = self.processor.codes_to_wav(outputs.audio_codes_list[0])

            # Shape 3: output is a message with .content containing audio
            elif hasattr(outputs, 'content'):
                content = outputs.content
                if isinstance(content, (torch.Tensor, np.ndarray)):
                    wav = content
                elif isinstance(content, list):
                    # Content might be a list of parts — find the audio part
                    for part in content:
                        if hasattr(part, 'audio'):
                            wav = part.audio
                            break
                        if isinstance(part, (torch.Tensor, np.ndarray)):
                            wav = part
                            break

            # Shape 4: processor.decode() handles the message
            if wav is None and hasattr(self.processor, 'decode'):
                decoded = self.processor.decode(outputs)
                if hasattr(decoded, 'audio'):
                    wav = decoded.audio
                elif hasattr(decoded, 'audio_codes_list'):
                    wav = self.processor.codes_to_wav(decoded.audio_codes_list[0])
                elif isinstance(decoded, (torch.Tensor, np.ndarray)):
                    wav = decoded

            if wav is None:
                return self.error_response(
                    f"Could not extract audio from {type(outputs).__name__}. "
                    f"Attrs: {dir(outputs)}"
                )

            if isinstance(wav, torch.Tensor):
                wav = wav.squeeze().cpu().float().numpy()
            elif not isinstance(wav, np.ndarray):
                wav = np.array(wav, dtype=np.float32)

            audio_base64 = self.audio_to_base64(wav, self.sample_rate)
            return self.success_response(audio_base64, self.sample_rate)
        except Exception as e:
            return self.error_response(e)
