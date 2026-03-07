import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

fish_speech_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "libsndfile1", "espeak-ng", "portaudio19-dev", "clang")
    .uv_pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy",
        "soundfile",
        "huggingface_hub",
        "fish-speech",
    )
    # Pre-download the OpenAudio S1-mini checkpoint
    .run_commands(
        # Create .project-root marker that pyrootutils expects
        "touch /usr/local/lib/python3.12/site-packages/.project-root",
        "python3 -c \"from huggingface_hub import snapshot_download; "
        "snapshot_download('fishaudio/openaudio-s1-mini', local_dir='/root/checkpoints/openaudio-s1-mini')\"",
        secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=fish_speech_image,
    gpu="T4",
    scaledown_window=300,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class FishSpeechModel(BaseTTSModel):
    """Fish Audio OpenAudio S1-mini — multilingual TTS with Arabic support."""

    model_id = "fish_speech"
    display_name = "Fish Speech S1-mini"
    model_url = "https://huggingface.co/fishaudio/s1-mini"

    @modal.enter()
    def load_model(self):
        """Load Fish Speech S1-mini models when container starts."""
        import torch
        from fish_speech.inference_engine import TTSInferenceEngine
        from fish_speech.models.dac.inference import load_model as load_decoder_model
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue

        checkpoint_path = "/root/checkpoints/openaudio-s1-mini"
        decoder_checkpoint = f"{checkpoint_path}/codec.pth"
        precision = torch.bfloat16

        # Launch the LLAMA text-to-semantic queue
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=checkpoint_path,
            device="cuda",
            precision=precision,
            compile=False,
        )

        # Load the DAC decoder model
        decoder_model = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=decoder_checkpoint,
            device="cuda",
        )

        # Create the inference engine
        self.engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=precision,
            compile=False,
        )

        # Get sample rate from decoder
        if hasattr(decoder_model, "spec_transform"):
            self.sample_rate = decoder_model.spec_transform.sample_rate
        else:
            self.sample_rate = getattr(decoder_model, "sample_rate", 44100)

        print(f"✅ Fish Speech S1-mini loaded on CUDA (sr={self.sample_rate})")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Synthesize Arabic text to speech."""
        try:
            import numpy as np
            from fish_speech.utils.schema import ServeTTSRequest

            req = ServeTTSRequest(
                text=text,
                max_new_tokens=1024,
                chunk_length=300,
                top_p=0.8,
                repetition_penalty=1.2,  # slightly higher to avoid Arabic phoneme loops
                temperature=0.7,         # lower = more stable Arabic output
            )

            audio_segments = []
            for result in self.engine.inference(req):
                if result.code == "final" and result.audio is not None:
                    _, audio_data = result.audio
                    audio_segments.append(audio_data)
                elif result.code == "error":
                    return self.error_response(str(result.error))

            if not audio_segments:
                return self.error_response("No audio generated")

            wav = np.concatenate(audio_segments) if len(audio_segments) > 1 else audio_segments[0]
            audio_base64 = self.audio_to_base64(wav, self.sample_rate)

            return self.success_response(audio_base64, self.sample_rate)
        except Exception as e:
            return self.error_response(e)
