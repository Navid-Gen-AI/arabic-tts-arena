"""Fish Speech S2 Pro — Fish Audio's next-gen multilingual TTS with Arabic support.

Runs the open-source fish-speech engine (LLAMA text-to-semantic queue + DAC
decoder) installed editable from the upstream repo.

Model: https://huggingface.co/fishaudio/s2-pro
Repo:  https://github.com/fishaudio/fish-speech
"""

import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

MODEL_REPO = "fishaudio/s2-pro"
CHECKPOINT_DIR = "/root/checkpoints/s2-pro"
FISH_SPEECH_DIR = "/opt/fish-speech"


fish_speech_s2_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04",
        add_python="3.12",
    )
    # clang: pyaudio's C extension needs it (Modal's Python was built with clang).
    .apt_install("ffmpeg", "libsox-dev", "portaudio19-dev", "clang", "git")
    .run_commands(
        # Clone repo and install in editable mode so all subpackages are importable.
        f"git clone --depth 1 https://github.com/fishaudio/fish-speech.git {FISH_SPEECH_DIR}",
        f"cd {FISH_SPEECH_DIR} && pip install -e .",
        # descript-audiotools (transitive via descript-audio-codec) pins
        # protobuf<3.20, which breaks Modal's injected client at startup
        # (modal/_environments.py needs EnumTypeWrapper.ValueType, added in
        # protobuf 3.20). fish-speech declares a protobuf>=3.20 override in
        # [tool.uv], but `pip` ignores [tool.uv] overrides, so force a
        # compatible protobuf here (matches fish-speech's own <6.0.0 cap).
        "pip install --upgrade 'protobuf>=4.25.3,<6.0.0'",
    )
    # Downloads must be `python3 -c` shell commands, not .run_function():
    # run_function imports this module in a bare build container where the
    # local app/models sources don't exist yet. Layer structure and command
    # strings match the originally-deployed image so Modal's cache reuses it.
    .run_commands(
        # fish-speech resolves paths relative to this marker file.
        f"touch {FISH_SPEECH_DIR}/.project-root",
        "python3 -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{MODEL_REPO}', local_dir='{CHECKPOINT_DIR}')\"",
        secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=fish_speech_s2_image,
    gpu="A10G",
    scaledown_window=120,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    timeout=600,
)
class FishSpeechS2Model(BaseTTSModel):
    """Fish Audio S2 Pro — next-gen multilingual TTS with Arabic support."""

    model_id = "fish_speech_s2"
    display_name = "Fish Speech S2 Pro"
    model_url = "https://huggingface.co/fishaudio/s2-pro"
    gpu = "A10G"

    @modal.enter()
    def load_model(self):
        import torch
        from fish_speech.inference_engine import TTSInferenceEngine
        from fish_speech.models.dac.inference import load_model as load_decoder_model
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue

        precision = torch.bfloat16

        llama_queue = launch_thread_safe_queue(
            checkpoint_path=CHECKPOINT_DIR,
            device="cuda",
            precision=precision,
            compile=False,
        )
        decoder_model = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=f"{CHECKPOINT_DIR}/codec.pth",
            device="cuda",
        )
        self.engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=precision,
            compile=False,
        )

        if hasattr(decoder_model, "spec_transform"):
            self.sample_rate = decoder_model.spec_transform.sample_rate
        else:
            self.sample_rate = getattr(decoder_model, "sample_rate", 44100)

        print(f"✅ Fish Speech S2 Pro loaded on CUDA (sr={self.sample_rate})")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        try:
            import time
            import numpy as np
            from fish_speech.utils.schema import ServeTTSRequest

            start = time.perf_counter()
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

            return self.success_response(
                self.audio_to_base64(wav, self.sample_rate), self.sample_rate,
                inference_seconds=time.perf_counter() - start,
            )
        except Exception as e:
            return self.error_response(e)
