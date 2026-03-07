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

# Each TTS model defines its own image in its model file.
# Only the arena service image is defined here.

# Shared list of local modules every model container needs
LOCAL_MODULES = ("app", "models")

# ── Lightweight image for arena services (voting, leaderboard) ────────────
service_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("pydantic", "huggingface_hub")
    .add_local_python_source("app", "service", "storage", "elo", "models")
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
