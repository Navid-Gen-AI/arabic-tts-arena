"""Higgs TTS 3 (4B) — Boson AI multilingual TTS (Arabic in the production tier).

The official repo needs a vLLM-Omni/SGLang-Omni server, so we use the
plain-transformers port (same weights, direct `generate_speech()` API).
It clones a fixed MSA reference voice: without one the model picks a random
"smart voice" per generation, and votes would measure voice lottery instead
of the model.

Model: https://huggingface.co/bosonai/higgs-tts-3-4b
Port:  https://huggingface.co/multimodalart/higgs-audio-v3-tts-4b-transformers
"""

import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

MODEL_REPO = "multimodalart/higgs-audio-v3-tts-4b-transformers"
# The port runs trust_remote_code — pin so upstream pushes can't change what we execute.
MODEL_REVISION = "30f01593ee6a12efa586c92455afe4b76e45095d"

# The port's get_audio_codec() lazily pulls this separate ~800 MB codec repo
# (config.audio_tokenizer_id) — pre-download it at build so container start
# never downloads it. Transformers-native model, no remote code.
CODEC_REPO = "bosonai/higgs-audio-v2-tokenizer"
CODEC_REVISION = "403fbacf2f60caaa102f893fdfabb694619b2417"

# Fixed MSA reference voice — same clip + transcript pair the Habibi TTS
# integration used (official Habibi-TTS Space assets).
REF_AUDIO_PATH = "/root/ref/msa_ref.wav"
REF_TEXT = (
    "كان اللعيب حاضرًا في العديد من الأنشطة والفعاليات المرتبطة بكأس العالم، "
    "مما سمح للجماهير بالتفاعل معه والتقاط الصور التذكارية."
)


higgs_tts_3_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04",
        add_python="3.12",
    )
    # build-essential: torch dispatches some eager ops to triton kernels that
    # JIT-compile C at runtime and need a host compiler (cc).
    .apt_install("ffmpeg", "libsndfile1", "build-essential")
    .uv_pip_install(
        "torch",
        "torchaudio",
        "transformers>=5.5",
        "accelerate",
        "numpy",
        "soundfile",
        "huggingface_hub[hf_xet]",
    )
    # Pinned weights + reference clip resampled to mono 24 kHz WAV. Must be
    # `python3 -c` shell commands, not .run_function() (which imports this
    # module in a bare build container where the local app/models sources
    # don't exist yet).
    .run_commands(
        "python3 -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{MODEL_REPO}', revision='{MODEL_REVISION}')\"",
        # The port's remote code loads the codec by repo id (revision 'main'),
        # so alongside the pinned snapshot we write refs/main into the cache —
        # otherwise offline resolution of 'main' at runtime fails.
        "python3 -c \"from huggingface_hub import snapshot_download; from pathlib import Path; "
        f"p = Path(snapshot_download('{CODEC_REPO}', revision='{CODEC_REVISION}')); "
        "refs = p.parents[1] / 'refs'; refs.mkdir(exist_ok=True); "
        f"(refs / 'main').write_text('{CODEC_REVISION}')\"",
        "python3 -c \"from huggingface_hub import hf_hub_download; "
        "hf_hub_download('chenxie95/Habibi-TTS', 'assets/MSA.mp3', "
        "repo_type='space', local_dir='/root')\"",
        f"mkdir -p /root/ref && ffmpeg -i /root/assets/MSA.mp3 -ar 24000 -ac 1 {REF_AUDIO_PATH}",
        secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    )
    # Everything is baked into the image above; container start must never
    # reach the Hub (a hung download here is a startup-timeout kill).
    .env({"HF_HUB_OFFLINE": "1"})
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=higgs_tts_3_image,
    gpu="A10G",  # 4B params in bf16 (~9 GB) + audio codec fit comfortably in 24 GB
    scaledown_window=120,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    timeout=600,
)
class HiggsTTS3Model(BaseTTSModel):
    """Higgs TTS 3 (4B) — Boson AI multilingual TTS, cloning a fixed MSA voice."""

    model_id = "higgs_tts_3"
    display_name = "Higgs TTS 3"
    model_url = "https://huggingface.co/bosonai/higgs-tts-3-4b"
    gpu = "A10G"

    @modal.enter()
    def load_model(self):
        import soundfile as sf
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # trust_remote_code also here: AutoTokenizer consults the repo's custom
        # AutoConfig, and without the flag transformers falls into an interactive
        # [y/N] prompt that hangs a headless container until the startup timeout.
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_REPO, revision=MODEL_REVISION, trust_remote_code=True,
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_REPO,
                revision=MODEL_REVISION,
                trust_remote_code=True,
                dtype=torch.bfloat16,
            )
            .eval()
            .to("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.get_audio_codec()  # preload the 24 kHz codec
        self.sample_rate = int(self.model.config.sample_rate)

        data, self.ref_sr = sf.read(REF_AUDIO_PATH, dtype="float32", always_2d=True)
        self.ref_wav = torch.from_numpy(data).mean(dim=1)  # mono [L]

        # Warm-up: triton JIT-compiles its kernels on the first forward pass;
        # do it at container start so it never lands on a voter's request.
        with torch.inference_mode():
            self.model.generate_speech(
                "مرحبا",
                self.tokenizer,
                max_new_tokens=64,
                reference_audio=self.ref_wav,
                reference_sample_rate=self.ref_sr,
                reference_text=REF_TEXT,
            )

        print(f"✅ Higgs TTS 3 loaded (sr={self.sample_rate})")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        try:
            import torch

            text = (text or "").strip()
            if not text:
                return self.error_response("Input text is empty")

            # Fixed seed keeps the sampled delivery stable across battles.
            torch.manual_seed(42)

            with torch.inference_mode():
                audio = self.model.generate_speech(
                    text,
                    self.tokenizer,
                    max_new_tokens=2048,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=50,
                    reference_audio=self.ref_wav,
                    reference_sample_rate=self.ref_sr,
                    reference_text=REF_TEXT,
                )

            wav = audio.detach().float().cpu().numpy()
            if wav.size < 100:
                return self.error_response(f"Audio too short: {wav.size} samples")

            return self.success_response(
                self.audio_to_base64(wav, self.sample_rate), self.sample_rate,
            )

        except Exception as e:
            import traceback
            return self.error_response(f"{e}\n{traceback.format_exc()}")
