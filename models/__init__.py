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
import functools
import pkgutil
import base64
import time
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
# Automatic synthesis timing
# =============================================================================

def _timed_synthesize(fn):
    """Wrap a synthesize() implementation so successful responses carry
    `inference_seconds` automatically. An explicit value passed by the model
    (e.g. custom streaming measurement) is never overridden."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        result = fn(self, *args, **kwargs)
        if (
            isinstance(result, dict)
            and result.get("success")
            and "inference_seconds" not in result
        ):
            result["inference_seconds"] = round(time.perf_counter() - start, 2)
        return result

    wrapper._auto_timed = True
    return wrapper


def _install_auto_timing(cls):
    """Patch a BaseTTSModel subclass so its synthesize() is auto-timed.

    Runs from BaseTTSModel.__init_subclass__, i.e. when the class body
    finishes executing — after @modal.method() has wrapped synthesize but
    before @app.cls / @register_model run. For a @modal.method() the real
    function lives on the wrapper's `raw_f`, which is what Modal executes in
    the container (this module is re-imported there, so the patch applies
    remotely too). If Modal ever changes that internal, we degrade
    gracefully: service.py falls back to wall-clock latency.
    """
    obj = cls.__dict__.get("synthesize")
    if obj is None:
        return
    # Plain function (no @modal.method()) — wrap directly.
    if not hasattr(obj, "_get_raw_f"):
        if callable(obj) and not getattr(obj, "_auto_timed", False):
            setattr(cls, "synthesize", _timed_synthesize(obj))
        return
    impl = next(
        (v for k, v in vars(obj).items() if k.startswith("_sync_original")),
        None,
    )
    raw = getattr(impl, "raw_f", None)
    if callable(raw):
        if not getattr(raw, "_auto_timed", False):
            impl.raw_f = _timed_synthesize(raw)
    else:
        print(
            f"⚠️ auto-timing not applied to {cls.__name__}.synthesize "
            "(Modal internals changed?); latency falls back to wall time"
        )


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

    Synthesis timing is automatic: every subclass's synthesize() is wrapped at
    class-definition time, and successful responses gain `inference_seconds`
    (model-side synthesis wall time) without any per-model code.
    """

    model_id: str = ""
    display_name: str = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _install_auto_timing(cls)

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

    def success_response(
        self,
        audio_base64: str,
        sample_rate: int,
        inference_seconds: float | None = None,
    ) -> dict:
        """Build a success payload.

        inference_seconds is normally filled in automatically by the
        synthesize() timing wrapper (see _timed_synthesize). Pass it
        explicitly only when the model measures a more precise window
        itself — an explicit value always wins.
        """
        response = {
            "success": True,
            "audio_base64": audio_base64,
            "sample_rate": sample_rate,
            "model_id": self.model_id,
        }
        if inference_seconds is not None:
            response["inference_seconds"] = round(inference_seconds, 2)
        return response

    def error_response(self, error: str) -> dict:
        return {
            "success": False,
            "error": str(error),
            "model_id": self.model_id,
        }


# =============================================================================
# Auto-discover model files
# =============================================================================

# Retired models — no longer deployed on Modal but kept for leaderboard history.
# To retire a model: add its entry here (copy display_name, model_url, gpu,
# open_weight from the model file). _SKIP is derived automatically.
RETIRED_MODELS: dict[str, dict] = {
    "lahgtna": {
        "display_name": "Lahgtna",
        "model_url": "https://huggingface.co/oddadmix/lahgtna-chatterbox-v0",
        "gpu": "T4",
        "open_weight": True,
    },
    "speecht5_ar": {
        "display_name": "SpeechT5 Arabic",
        "model_url": "https://huggingface.co/MBZUAI/speecht5_tts_clartts_ar",
        "gpu": "T4",
        "open_weight": True,
    },
    "oute_tts": {
        "display_name": "OuteTTS 1.0",
        "model_url": "https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B",
        "gpu": "A10G",
        "open_weight": True,
    },
    "spark_tts": {
        "display_name": "Arabic Spark TTS",
        "model_url": "https://huggingface.co/IbrahimSalah/Arabic-TTS-Spark",
        "gpu": "T4",
        "open_weight": True,
    },
    "silma_tts_v1_large": {
        "display_name": "SILMA TTS v1 Large",
        "model_url": "https://silma.ai/arabic-tts-models",
        "gpu": "",
        "open_weight": False,
    },
    "moss_tts": {
        "display_name": "MOSS-TTS",
        "model_url": "https://huggingface.co/OpenMOSS-Team/MOSS-TTS",
        "gpu": "A100-40GB",
        "open_weight": True,
    },
    "chatterbox": {
        "display_name": "Multilingual Chatterbox",
        "model_url": "https://huggingface.co/ResembleAI/chatterbox",
        "gpu": "T4",
        "open_weight": True,
    },
    "fish_speech": {
        "display_name": "Fish Speech S1-mini",
        "model_url": "https://huggingface.co/fishaudio/s1-mini",
        "gpu": "T4",
        "open_weight": True,
    },
    "kani_tts": {
        "display_name": "KaniTTS Arabic",
        "model_url": "https://huggingface.co/nineninesix/kani-tts-400m-ar",
        "gpu": "T4",
        "open_weight": True,
    },
    "lahgtna_v2": {
        "display_name": "Lahgatna 2.0",
        "model_url": "https://huggingface.co/oddadmix/lahgtna-chatterbox-v1",
        "gpu": "T4",
        "open_weight": True,
    },
    "moss_tts_nano": {
        "display_name": "MOSS-TTS Nano",
        "model_url": "https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M",
        "gpu": "T4",
        "open_weight": True,
    },
    "arabic_f5_tts": {
        "display_name": "Arabic F5-TTS",
        "model_url": "https://huggingface.co/IbrahimSalah/Arabic-F5-TTS-v2",
        "gpu": "T4",
        "open_weight": True,
    },
    "habibi_tts": {
        "display_name": "Habibi TTS",
        "model_url": "https://github.com/SWivid/Habibi-TTS",
        "gpu": "T4",
        "open_weight": True,
    },
}

# Files to skip during auto-discovery (not real models, or retired — see above)
_SKIP = {"__init__", "example_api_model"} | set(RETIRED_MODELS)


def _discover_models():
    """Import all .py files in models/ to trigger @register_model decorators."""
    package_dir = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name in _SKIP:
            continue
        importlib.import_module(f".{module_info.name}", __package__)


_discover_models()

# Tag active models, then merge retired models into the registry
for _info in MODEL_REGISTRY.values():
    _info.setdefault("retired", False)

for _mid, _meta in RETIRED_MODELS.items():
    MODEL_REGISTRY[_mid] = {
        "class_name": "",
        "display_name": _meta["display_name"],
        "model_url": _meta["model_url"],
        "gpu": _meta["gpu"],
        "open_weight": _meta["open_weight"],
        "retired": True,
    }


__all__ = ["MODEL_REGISTRY", "RETIRED_MODELS", "BaseTTSModel", "register_model"]
