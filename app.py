"""Arabic TTS Arena — Modal app, images, and volumes.

This is the entry point for `modal deploy app.py`.
"""

import modal

# =============================================================================
# Modal App & Storage
# =============================================================================

app = modal.App("arabic-tts-arena")

votes_volume = modal.Volume.from_name("arabic-tts-votes", create_if_missing=True)

# =============================================================================
# Container Images
# =============================================================================

# ── Base image for open-source GPU models ──────────────────────────────────
# Shared CUDA + PyTorch layer. Each model extends this with its own deps.
base_gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "libsndfile1", "espeak-ng", "git")
    .uv_pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "scipy",
        "numpy",
        "soundfile",
        "librosa",
        "huggingface_hub",
        "pydantic",
    )
)

# ── Base image for closed-source / API-based models ───────────────────────
# Lightweight — no GPU, no ML libs. Just HTTP + audio processing.
base_api_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "requests",
        "httpx",
        "numpy",
        "soundfile",
        "pydantic",
    )
)

# ── Lightweight image for arena services (voting, leaderboard) ────────────
service_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("pydantic", "huggingface_hub")
)

# =============================================================================
# Import models & service so Modal discovers all cls/functions on this app
# =============================================================================

def register_all():
    """Import models and service modules so Modal discovers their classes.

    Wrapped in a function to make the dependency explicit and avoid
    accidental re-import side effects.
    """
    import models      # noqa: F401  — triggers model auto-discovery
    import service     # noqa: F401  — ArenaService + cron job


register_all()
