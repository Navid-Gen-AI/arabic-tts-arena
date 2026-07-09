import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

# Fasih-TTS-V1 — Coqui XTTS v2 fine-tuned for Modern Standard Arabic (Fusha) with a
# single professional "news-anchor" male voice. Ships with a full Arabic front-end
# (normalize -> expand numbers -> CATT diacritization -> sacred-term lexicon -> chunk)
# and precomputed speaker-conditioning latents, so no reference-audio cloning step is
# needed at inference time.
# Ref: https://huggingface.co/NightPrince/Fasih-TTS-V1
# Source (front-end + serving code, MIT): https://github.com/NightPrinceY/Fasih-TTS-V1

# Pinned to a known-good commit of the upstream repo for reproducible builds.
_FASIH_REPO = "https://github.com/NightPrinceY/Fasih-TTS-V1.git"
_FASIH_COMMIT = "efa28e75131e4040d4b824d14e879e08a6e3b9fd"

# CATT (github.com/abjadai/catt, MIT) diacritizer checkpoint — same one the model's
# own `fasih_tts_server` Docker image bakes in.
_CATT_CKPT_URL = "https://github.com/abjadai/catt/releases/download/v2/best_ed_mlm_ns_epoch_178.pt"
_CATT_CKPT_PATH = "/root/models/catt/best_ed_mlm_ns_epoch_178.pt"

fasih_tts_v1_image = (
    # Same base image the author validated the model against (torch 2.3.1 / cuda 12.1) —
    # XTTS's GPT head is fp32-only on Turing GPUs (T4/2080 Ti), so we keep the exact
    # torch/cuda combination from `fasih_tts_server/Dockerfile` instead of a newer one.
    modal.Image.from_registry("pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime")
    .apt_install("ffmpeg", "libsndfile1", "git", "curl")
    # Versions match `fasih_tts_server/requirements.txt` from the official repo exactly.
    .uv_pip_install(
        "torchaudio==2.3.1",
        "coqui-tts==0.27.5",
        "transformers>=4.57,<5.0",
        "huggingface_hub>=0.25",
        "pytorch-lightning>=2.4",
        "kaldialign>=0.9",
        "pyarabic>=0.6",
        "num2words>=0.5",
        "librosa>=0.10",
        "soundfile>=0.12",
        "pyyaml>=6.0",
        "numpy<2.0",
    )
    .run_commands(
        # Vendor the model's own Arabic text front-end (normalize/numbers/diacritize/
        # chunk) and vendored CATT diacritizer — this is the same `src/tts` package
        # `fasih_tts_server/Dockerfile` copies into its serving image.
        f"git clone {_FASIH_REPO} /root/fasih_src "
        f"&& cd /root/fasih_src && git checkout {_FASIH_COMMIT}",
        # CATT diacritizer checkpoint — required so bare (non-diacritized) Arabic
        # input is pronounced correctly, exactly like the official server.
        f"mkdir -p /root/models/catt && curl -fsSL -o {_CATT_CKPT_PATH} {_CATT_CKPT_URL}",
        # Model weights: config, vocab, checkpoint, and precomputed speaker-conditioning
        # latents (gpt_cond_latent + speaker_embedding) baked in at build time.
        "python3 -c \""
        "from huggingface_hub import snapshot_download; "
        "snapshot_download("
        "    repo_id='NightPrince/Fasih-TTS-V1',"
        "    local_dir='/root/model',"
        "    allow_patterns=['model.pth', 'config.json', 'vocab.json', 'speaker_latents.pt'],"
        ")\"",
        secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=fasih_tts_v1_image,
    gpu="T4",  # Turing (sm_75) — same GPU family the model was trained/benchmarked on
    scaledown_window=120,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class FasihTTSV1Model(BaseTTSModel):
    """Fasih-TTS-V1 — Arabic (MSA/Fusha) professional male TTS, fine-tuned from XTTS v2.

    Source: https://huggingface.co/NightPrince/Fasih-TTS-V1
    """

    model_id = "fasih_tts_v1"
    display_name = "Fasih TTS V1"
    model_url = "https://huggingface.co/NightPrince/Fasih-TTS-V1"
    gpu = "T4"

    @modal.enter()
    def load_model(self):
        """Load the fine-tuned XTTS checkpoint, speaker latents, and Arabic text pipeline."""
        import sys
        import numpy as np
        import torch
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        # Make the vendored `tts` front-end package (cloned into the image above)
        # importable, matching how the official server locates it next to itself.
        sys.path.insert(0, "/root/fasih_src/src")
        from tts.text.pipeline import TextPipeline

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.temperature = 0.65  # matches the model's documented inference defaults
        self.sample_rate = 24000

        cfg = XttsConfig()
        cfg.load_json("/root/model/config.json")
        self.model = Xtts.init_from_config(cfg)
        self.model.load_checkpoint(
            cfg,
            checkpoint_path="/root/model/model.pth",
            vocab_path="/root/model/vocab.json",
            use_deepspeed=False,
        )
        self.model.to(self.device).eval()

        # Precomputed speaker-conditioning latents ship with the model, so no
        # reference-audio cloning step is needed at inference time.
        latents = torch.load("/root/model/speaker_latents.pt", map_location=self.device)
        self.gpt_cond = latents["gpt_cond_latent"].to(self.device)
        self.speaker = latents["speaker_embedding"].to(self.device)

        # CATT diacritizer — auto-diacritizes bare (non-diacritized) Arabic input.
        # Loading is wrapped in try/except because the upstream TextPipeline itself
        # falls back gracefully (no lexicon overrides) when assets are unavailable;
        # we mirror that same "degrade, don't crash" behavior for the diacritizer.
        diacritizer = None
        try:
            from tts.text.diacritize import Diacritizer

            diacritizer = Diacritizer(ckpt=_CATT_CKPT_PATH, device=self.device)
        except Exception as e:
            print(f"⚠️ CATT diacritizer unavailable, falling back to raw text: {e}")

        self.pipe = TextPipeline(diacritizer=diacritizer)
        # 120ms silence between chunks, matching the model's own engine/server code.
        self._gap = np.zeros(int(self.sample_rate * 0.12), dtype=np.float32)

        print(f"✅ Fasih-TTS-V1 loaded on {self.device.upper()} (sr={self.sample_rate})")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Run the model's full Arabic front-end, then synthesize speech."""
        try:
            import numpy as np
            import torch

            text = text.strip()
            if not text:
                return self.error_response("Input text is empty")

            # normalize -> expand numbers -> diacritize-if-needed -> sacred-term
            # lexicon -> <=160-char chunks, exactly as the official serving code.
            chunks = self.pipe.prepare_chunks(text)
            if not chunks:
                return self.error_response("Input text produced no synthesizable chunks")

            print(f"[fasih_tts_v1] text: {text[:80]}")

            pieces = []
            with torch.inference_mode():
                for i, chunk in enumerate(chunks):
                    out = self.model.inference(
                        chunk,
                        "ar",
                        self.gpt_cond,
                        self.speaker,
                        temperature=self.temperature,
                        repetition_penalty=2.0,
                        enable_text_splitting=False,
                    )
                    pieces.append(np.asarray(out["wav"], dtype=np.float32))
                    if i < len(chunks) - 1:
                        pieces.append(self._gap)

            wav = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
            if wav.size < 100:
                return self.error_response(f"Audio too short: {wav.size} samples")

            print(f"[fasih_tts_v1] audio: len={wav.size}, sr={self.sample_rate}")

            audio_base64 = self.audio_to_base64(wav, self.sample_rate)
            return self.success_response(audio_base64, self.sample_rate)

        except Exception as e:
            import traceback
            return self.error_response(f"{e}\n{traceback.format_exc()}")
