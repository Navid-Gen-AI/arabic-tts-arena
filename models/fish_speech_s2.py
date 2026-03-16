import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

fish_speech_s2_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04",
        add_python="3.12",
    )
    # clang needed to compile pyaudio's C extension (Modal's Python was built with clang)
    # git needed for editable install from cloned repo
    .apt_install("ffmpeg", "libsox-dev", "portaudio19-dev", "clang", "git")
    .run_commands(
        # Clone repo and install in editable mode so all subpackages are available
        "git clone --depth 1 https://github.com/fishaudio/fish-speech.git /opt/fish-speech",
        "cd /opt/fish-speech && pip install -e .",
    )
    .run_commands(
        "touch /opt/fish-speech/.project-root",
        "python3 -c \"from huggingface_hub import snapshot_download; "
        "snapshot_download('fishaudio/s2-pro', local_dir='/root/checkpoints/s2-pro')\"",
        secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=fish_speech_s2_image,
    gpu="A10G",
    scaledown_window=300,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class FishSpeechS2Model(BaseTTSModel):
    """Fish Audio S2 Pro — next-gen multilingual TTS with Arabic support."""

    model_id = "fish_speech_s2"
    display_name = "Fish Speech S2 Pro"
    model_url = "https://huggingface.co/fishaudio/s2-pro"
    gpu = "A10G"

    @modal.enter()
    def load_model(self):
        """Load Fish Speech S2 Pro models when container starts."""
        import torch
        from fish_speech.inference_engine import TTSInferenceEngine
        from fish_speech.models.dac.inference import load_model as load_decoder_model
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue

        checkpoint_path = "/root/checkpoints/s2-pro"
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

        print(f"✅ Fish Speech S2 Pro loaded on CUDA (sr={self.sample_rate})")

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
                repetition_penalty=1.2,
                temperature=0.7,
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
