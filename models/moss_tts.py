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
        "rm -rf /tmp/moss-tts",
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=moss_tts_image,
    gpu="A10G",
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

        torch.backends.cuda.enable_cudnn_sdp(False)

        model_name = "OpenMOSS-Team/MOSS-TTS"

        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")

        # Get the UserMessage class from the model module
        self.UserMessage = self.model.UserMessage

        self.sample_rate = getattr(
            self.model.config, "sampling_rate", 24000
        )

        print(f"✅ MOSS-TTS loaded (sr={self.sample_rate})")
        print(f"   model type:     {type(self.model).__name__}")
        print(f"   processor type: {type(self.processor).__name__}")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Synthesize Arabic text to speech using the MOSS-TTS delay model."""
        try:
            import torch
            import numpy as np

            # 1. Build input using the model's UserMessage dataclass
            msg = self.UserMessage(text=text, language="ar")
            conversations = [msg.to_dict()]

            # 2. Tokenize via processor
            inputs = self.processor(
                conversations=conversations,
                return_tensors="pt",
            ).to("cuda")

            # 3. Generate delay-patterned audio codes
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=4096)

            # outputs is list[tuple[scalar, codes_tensor]]
            # Extract the audio codes (second element of first tuple)
            if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                if isinstance(outputs[0], tuple) and len(outputs[0]) >= 2:
                    audio_codes = outputs[0][1]  # shape: [n_codebooks, n_timesteps]
                elif isinstance(outputs, torch.Tensor):
                    audio_codes = outputs
                else:
                    return self.error_response(
                        f"Unexpected generate output structure: "
                        f"{type(outputs).__name__}"
                    )
            elif isinstance(outputs, torch.Tensor):
                audio_codes = outputs
            else:
                return self.error_response(
                    f"Empty generate output: {type(outputs).__name__}"
                )

            print(f"  audio_codes shape: {list(audio_codes.shape)}, "
                  f"dtype: {audio_codes.dtype}")

            # 4. Un-delay the interleaved codes
            audio_codes = self.processor.apply_de_delay_pattern(audio_codes)
            print(f"  after de-delay: {list(audio_codes.shape)}")

            # 5. Decode audio codes → waveform
            wav = self.processor.decode_audio_codes(audio_codes)
            print(f"  decoded wav type: {type(wav).__name__}")

            # Convert to numpy
            if isinstance(wav, torch.Tensor):
                wav = wav.squeeze().cpu().float().numpy()
            elif isinstance(wav, np.ndarray):
                wav = wav.squeeze().astype(np.float32)
            else:
                wav = np.array(wav, dtype=np.float32).squeeze()

            if wav.size < 100:
                return self.error_response(
                    f"Decoded audio too short: {wav.size} samples"
                )

            print(f"  ✅ wav shape: {wav.shape}, sr: {self.sample_rate}")
            audio_base64 = self.audio_to_base64(wav, self.sample_rate)
            return self.success_response(audio_base64, self.sample_rate)

        except Exception as e:
            return self.error_response(e)
