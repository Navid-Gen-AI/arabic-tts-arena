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
        # Pin to exact versions from the official MOSS-TTS Space
        "torch==2.9.1",
        "torchaudio==2.9.1",
        "transformers==5.0.0",
        "numpy==2.1.0",
        "soundfile==0.13.1",
        "accelerate",
        "safetensors==0.6.2",
        "einops==0.8.1",
        "scipy==1.16.2",
        "tiktoken==0.12.0",
        "tqdm==4.67.1",
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=moss_tts_image,
    gpu="A100-40GB",
    scaledown_window=60,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class MossTTSModel(BaseTTSModel):
    """OpenMOSS MOSS-TTS -- multilingual TTS with Arabic support.

    Inference follows the official HF Space:
    https://huggingface.co/spaces/OpenMOSS-Team/MOSS-TTS
    """

    model_id = "moss_tts"
    display_name = "MOSS-TTS"
    model_url = "https://huggingface.co/OpenMOSS-Team/MOSS-TTS"

    @modal.enter()
    def load_model(self):
        """Load model & processor exactly as the official Space does."""
        import torch
        from transformers import AutoModel, AutoProcessor

        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        model_name = "OpenMOSS-Team/MOSS-TTS"

        # -- Model (load first — it's the biggest piece) --
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
        )
        self.model.eval()

        # Free any cached allocator memory before loading the tokenizer
        torch.cuda.empty_cache()

        # -- Processor --
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True,
        )
        # Move the audio tokenizer to GPU and freeze (official code does this)
        if hasattr(self.processor, "audio_tokenizer"):
            self.processor.audio_tokenizer = (
                self.processor.audio_tokenizer.to("cuda")
            )
            self.processor.audio_tokenizer.eval()

        # Sample rate: official code reads processor.model_config.sampling_rate
        sr = 24000
        if hasattr(self.processor, "model_config"):
            sr = int(getattr(self.processor.model_config, "sampling_rate", sr))
        elif hasattr(self.model.config, "sampling_rate"):
            sr = int(self.model.config.sampling_rate)
        self.sample_rate = sr

        print(f"MOSS-TTS loaded (sr={self.sample_rate})")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Synthesize speech -- mirrors the official Space's run_inference()."""
        try:
            import torch
            import numpy as np

            # 1. Build conversation (double-nested list: batch of conversations)
            conversations = [
                [self.processor.build_user_message(text=text)]
            ]

            # 2. Tokenize (mode="generation" for text-only, no continuation)
            batch = self.processor(conversations, mode="generation")
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")

            # 3. Generate with official default sampling params
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=4096,
                    audio_temperature=1.7,
                    audio_top_p=0.8,
                    audio_top_k=25,
                    audio_repetition_penalty=1.0,
                )

            # 4. Decode outputs -> list of AssistantMessage
            messages = self.processor.decode(outputs)
            if not messages or messages[0] is None:
                return self.error_response(
                    "Model did not return a decodable audio result."
                )

            # 5. Extract waveform from the first message
            audio = messages[0].audio_codes_list[0]
            if isinstance(audio, torch.Tensor):
                audio_np = audio.detach().float().cpu().numpy()
            else:
                audio_np = np.asarray(audio, dtype=np.float32)

            if audio_np.ndim > 1:
                audio_np = audio_np.reshape(-1)
            audio_np = audio_np.astype(np.float32, copy=False)

            if audio_np.size < 100:
                return self.error_response(
                    f"Audio too short: {audio_np.size} samples"
                )

            audio_base64 = self.audio_to_base64(audio_np, self.sample_rate)
            return self.success_response(audio_base64, self.sample_rate)

        except Exception as e:
            import traceback
            return self.error_response(f"{e}\n{traceback.format_exc()}")
