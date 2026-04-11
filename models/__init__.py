"""
Arabic TTS Arena — Models package.

Contains:
- MODEL_REGISTRY: dict mapping model_id → {class_name, display_name}
- BaseTTSModel: base class all TTS models inherit from
- register_model: decorator to register a model
- Auto-discovery of all model files in this directory

To add a new model, create models/your_model.py and use @register_model.
"""

import io
import importlib
import pkgutil
import base64
from pathlib import Path

# =============================================================================
# Model Registry
# =============================================================================

MODEL_REGISTRY: dict[str, dict[str, str]] = {}


def register_model(cls):
    """
    Decorator to register a TTS model class.

    The class must have `model_id` and `display_name` class attributes.
    Stores the Python class name (for Modal lookup), display name (for UI),
    and model_url (for leaderboard links).

    Optional class attributes:
        gpu: str — GPU type used for inference (e.g. "T4", "A10G", "A100-40GB").
            Shown in the leaderboard tooltip. Empty string or absent for
            API-based models that don't use a GPU.
        open_weight: bool — True (default) for open-weight models, False for
            proprietary / closed-source API models.
    """
    model_id = getattr(cls, "model_id", None)
    if model_id is None:
        raise ValueError(f"Model class {cls.__name__} must have a 'model_id' class attribute")
    display_name = getattr(cls, "display_name", None) or cls.__name__
    model_url = getattr(cls, "model_url", "")
    gpu = getattr(cls, "gpu", "")
    open_weight = getattr(cls, "open_weight", True)
    MODEL_REGISTRY[model_id] = {
        "class_name": cls.__name__,
        "display_name": display_name,
        "model_url": model_url,
        "gpu": gpu,
        "open_weight": open_weight,
    }
    return cls


# =============================================================================
# Base TTS Model
# =============================================================================

class BaseTTSModel:
    """
    Base class for TTS models.

    Subclasses must define:
        model_id:      str — unique identifier (lowercase, underscores)
        display_name:  str — human-readable name shown in the UI

    And implement:
        load_model()            — called once on container start (@modal.enter())
        synthesize(text) -> dict — generate audio (@modal.method())
    """

    model_id: str = ""
    display_name: str = ""

    def load_model(self):
        raise NotImplementedError

    def synthesize(self, text: str) -> dict:
        raise NotImplementedError

    @staticmethod
    def audio_to_base64(wav_array, sample_rate: int) -> str:
        """Convert a numpy audio array to a base64-encoded WAV string."""
        import soundfile as sf
        import numpy as np

        if not isinstance(wav_array, np.ndarray):
            wav_array = np.array(wav_array, dtype=np.float32)
        peak = max(abs(wav_array.max()), abs(wav_array.min()))
        if peak > 1.0:
            wav_array = wav_array / peak

        buffer = io.BytesIO()
        sf.write(buffer, wav_array, sample_rate, format="WAV")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def success_response(self, audio_base64: str, sample_rate: int) -> dict:
        return {
            "success": True,
            "audio_base64": audio_base64,
            "sample_rate": sample_rate,
            "model_id": self.model_id,
        }

    def error_response(self, error: str) -> dict:
        return {
            "success": False,
            "error": str(error),
            "model_id": self.model_id,
        }


# =============================================================================
# Auto-discover model files
# =============================================================================

def _discover_models():
    """Import all .py files in models/ to trigger @register_model decorators."""
    package_dir = Path(__file__).parent
    skip = {"__init__", "example_api_model", "lahgtna", "speecht5_ar", "oute_tts", "spark_tts", "silma_tts_v1_large"}  # skip base stub, example template & deprecated models
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name in skip:
            continue
        importlib.import_module(f".{module_info.name}", __package__)


_discover_models()


__all__ = ["MODEL_REGISTRY", "BaseTTSModel", "register_model"]
