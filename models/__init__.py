"""
Arabic TTS Arena — Models package.

Contains:
- MODEL_REGISTRY: dict mapping model_id → {class_name, display_name}
- BaseTTSModel: base class all TTS models inherit from
- register_model: decorator to register a model
- arena_cls: @app.cls wrapper that skips Modal deployment for retired models
- Auto-discovery of all model files in this directory

To add a new model, create models/your_model.py and use @register_model.
To retire a model, set `retired = True` on its class (and flip it back to
return the model to the arena).
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
        retired: bool — True for models no longer deployed on Modal but kept
            in the registry for leaderboard history. Pair with @arena_cls so
            the class is also skipped at deploy time.
    """
    model_id = getattr(cls, "model_id", None)
    if model_id is None:
        raise ValueError(f"Model class {cls.__name__} must have a 'model_id' class attribute")
    display_name = getattr(cls, "display_name", None) or cls.__name__
    model_url = getattr(cls, "model_url", "")
    gpu = getattr(cls, "gpu", "")
    open_weight = getattr(cls, "open_weight", True)
    retired = getattr(cls, "retired", False)
    MODEL_REGISTRY[model_id] = {
        "class_name": cls.__name__,
        "display_name": display_name,
        "model_url": model_url,
        "gpu": gpu,
        "open_weight": open_weight,
        "retired": retired,
    }
    return cls


def arena_cls(**modal_kwargs):
    """Drop-in replacement for @app.cls that respects the `retired` flag.

    Active models are wrapped with @app.cls exactly as before. Retired models
    (class attribute `retired = True`) are returned unwrapped, so they are
    never registered on the Modal app or deployed — while @register_model
    above still records their metadata for leaderboard history.
    """
    def wrap(cls):
        if getattr(cls, "retired", False):
            return cls
        # Imported here, not at module top: this package is bundled into every
        # model container, and importing `app` lazily keeps the two modules
        # free of an import-order dependency.
        from app import app
        return app.cls(**modal_kwargs)(cls)
    return wrap


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
    before @arena_cls / @register_model run. For a @modal.method() the real
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

    Error policy: synthesize() RAISES on failure (ValueError for bad input,
    RuntimeError for backend/API failures) — never catch-and-return an error
    dict. service.py catches exceptions at the .remote() call site and degrades
    gracefully; swallowing errors model-side hides failures from Modal's
    dashboard and logs. Pair with retries=0 on @arena_cls so broken models fail
    fast and visibly.

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


# =============================================================================
# Auto-discover model files
# =============================================================================

# Files to skip during auto-discovery (not real models)
_SKIP = {"__init__", "example_api_model"}


def _discover_models():
    """Import all .py files in models/ to trigger @register_model decorators."""
    package_dir = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name in _SKIP:
            continue
        importlib.import_module(f".{module_info.name}", __package__)


_discover_models()


__all__ = ["MODEL_REGISTRY", "BaseTTSModel", "register_model", "arena_cls"]
