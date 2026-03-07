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

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,
                )

            # Deep introspection of what generate() returns
            def describe(obj, depth=0, max_depth=3):
                prefix = "  " * depth
                t = type(obj).__name__
                if isinstance(obj, torch.Tensor):
                    return f"{prefix}Tensor shape={obj.shape} dtype={obj.dtype}"
                elif isinstance(obj, np.ndarray):
                    return f"{prefix}ndarray shape={obj.shape} dtype={obj.dtype}"
                elif isinstance(obj, (list, tuple)):
                    lines = [f"{prefix}{t} len={len(obj)}"]
                    if depth < max_depth:
                        for i, item in enumerate(obj[:3]):  # first 3 items
                            lines.append(f"{prefix}  [{i}]: {describe(item, depth+1, max_depth)}")
                        if len(obj) > 3:
                            lines.append(f"{prefix}  ... and {len(obj)-3} more")
                    return "\n".join(lines)
                elif isinstance(obj, dict):
                    lines = [f"{prefix}dict keys={list(obj.keys())[:10]}"]
                    if depth < max_depth:
                        for k in list(obj.keys())[:3]:
                            lines.append(f"{prefix}  {k}: {describe(obj[k], depth+1, max_depth)}")
                    return "\n".join(lines)
                elif hasattr(obj, '__dict__'):
                    attrs = {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
                    lines = [f"{prefix}{t} attrs={list(attrs.keys())}"]
                    if depth < max_depth:
                        for k in list(attrs.keys())[:5]:
                            lines.append(f"{prefix}  {k}: {describe(attrs[k], depth+1, max_depth)}")
                    return "\n".join(lines)
                else:
                    s = str(obj)
                    return f"{prefix}{t}: {s[:200]}"

            description = describe(outputs)
            print(f"  OUTPUTS STRUCTURE:\n{description}")

            # Now try to extract audio based on what we learned
            wav = None

            # If outputs is a list of tuples, the tuple likely contains
            # (text_tokens, audio_codes) or similar
            first = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            if isinstance(first, (list, tuple)):
                for i, element in enumerate(first):
                    if isinstance(element, torch.Tensor):
                        print(f"  tuple[{i}]: Tensor shape={element.shape} dtype={element.dtype}")
                        # Audio codes are typically 2D (codebooks x timesteps)
                        # or 1D waveform
                        if element.dim() >= 2 and hasattr(self.processor, 'codes_to_wav'):
                            try:
                                wav = self.processor.codes_to_wav(element)
                                print(f"  → codes_to_wav succeeded from tuple[{i}]")
                                break
                            except Exception as e:
                                print(f"  → codes_to_wav failed on tuple[{i}]: {e}")
                        elif element.dim() == 1 and element.shape[0] > 100:
                            # Might be raw waveform
                            wav = element
                            print(f"  → Using tuple[{i}] as raw waveform")
                            break
            elif isinstance(first, torch.Tensor):
                if first.dim() >= 2 and hasattr(self.processor, 'codes_to_wav'):
                    wav = self.processor.codes_to_wav(first)
                else:
                    wav = first

            if wav is None:
                return self.error_response(
                    f"Could not extract audio. Structure: {description[:500]}"
                )

            if isinstance(wav, torch.Tensor):
                wav = wav.squeeze().cpu().float().numpy()
            elif not isinstance(wav, np.ndarray):
                wav = np.array(wav, dtype=np.float32)

            audio_base64 = self.audio_to_base64(wav, self.sample_rate)
            return self.success_response(audio_base64, self.sample_rate)
        except Exception as e:
            return self.error_response(e)
