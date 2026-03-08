import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

# Reference transcript that matches the bundled reference.wav
_REF_TRANSCRIPT = "لَا يَمُرُّ يَوْمٌ إِلَّا وَأَسْتَقْبِلُ عِدَّةَ رَسَائِلَ، تَتَضَمَّنُ أَسْئِلَةً مُلِحَّةْ."

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
    display_name = "Spark TTS (Arabic)"
    model_url = "https://huggingface.co/IbrahimSalah/Arabic-TTS-Spark"

    @modal.enter()
    def load_model(self):
        """Load model & processor exactly as the official Space does."""
        import torch
        from transformers import AutoModel, AutoProcessor

        model_name = "IbrahimSalah/Arabic-TTS-Spark"

        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True,
        ).eval().to("cuda")

        # The official Space sets processor.model = model
        self.processor.model = self.model

        self.ref_audio = "/root/spark-ref/reference.wav"
        self.ref_text = _REF_TRANSCRIPT
        self.sample_rate = 24000  # Will be confirmed from output

        print(f"✅ Arabic Spark-TTS loaded on CUDA (sr={self.sample_rate})")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Synthesize Arabic text — mirrors the official Space's generate_speech()."""
        try:
            import torch
            import numpy as np
            import os

            # Verify reference audio exists
            if not os.path.exists(self.ref_audio):
                return self.error_response(
                    f"Reference audio not found: {self.ref_audio}"
                )

            # Tokenize with reference audio + transcript (voice cloning)
            # Official Space uses text.lower()
            inputs = self.processor(
                text=text.lower(),
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
            output = self.processor.decode(
                generated_ids=output_ids,
                global_token_ids_prompt=global_tokens_prompt,
                input_ids_len=input_ids_len,
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
