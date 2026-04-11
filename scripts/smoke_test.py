"""
Post-deploy smoke test — verifies every registered TTS model can synthesize.

Runs after `modal deploy` in CI to catch container boot failures, missing
dependencies, or broken model weights before the frontend serves traffic.

Each model gets a hard timeout (default 180 s).  If the container keeps
crashing / respawning and never returns, the test fails fast instead of
hanging forever.

Usage:
    python scripts/smoke_test.py                                      # test all models
    python scripts/smoke_test.py habibi_tts lahgtna_v2                # test specific models (tab-completable)
    python scripts/smoke_test.py habibi_tts -t "مرحبا كيف حالك"      # custom text
    python scripts/smoke_test.py --text "أهلاً وسهلاً"               # custom text, all models
    python scripts/smoke_test.py --timeout 300 habibi_tts             # custom timeout

Tab completion (optional, one-time setup):
    pip install argcomplete
    activate-global-python-argcomplete          # or add: eval "$(register-python-argcomplete smoke_test.py)"

Exit codes:
    0  — all models passed
    1  — one or more models failed
"""
# PYTHON_ARGCOMPLETE_OK

import argparse
import base64
import os
import sys
import time
import modal
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# ---------------------------------------------------------------------------
# Import the local model registry so we can offer tab-completion choices
# without hitting the remote Modal backend.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models import MODEL_REGISTRY  # noqa: E402

APP_NAME = "arabic-tts-arena"
DEFAULT_TEXT = "السلام عليكم ورحمة الله وبركاته"  # short Arabic phrase — fast to synthesize
TIMEOUT_SECONDS = 180  # hard cap per model (includes cold start + inference)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "smoke_test_outputs")


def _build_parser() -> argparse.ArgumentParser:
    available = sorted(MODEL_REGISTRY.keys())
    parser = argparse.ArgumentParser(
        description="Smoke-test deployed Arabic TTS models on Modal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available models:\n  " + "\n  ".join(available),
    )
    parser.add_argument(
        "models",
        nargs="*",
        choices=available,
        default=[],
        metavar="MODEL",
        help="Model(s) to test (tab-completable). Omit to test all.",
    )
    parser.add_argument(
        "-t", "--text",
        default=DEFAULT_TEXT,
        help=f"Arabic text to synthesize (default: '{DEFAULT_TEXT}')",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=TIMEOUT_SECONDS,
        help=f"Per-model timeout in seconds (default: {TIMEOUT_SECONDS})",
    )

    # Enable argcomplete if installed (pip install argcomplete)
    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    return parser


def _call_model(class_name: str, text: str) -> dict:
    """Run the remote synthesis call (executed in a thread so we can time-out)."""
    cls = modal.Cls.from_name(APP_NAME, class_name)
    return cls().synthesize.remote(text)


def smoke_test_model(model_id: str, class_name: str, text: str, timeout: int) -> bool:
    """Call synthesize() on a single model with a hard timeout."""
    print(f"\n{'='*50}")
    print(f"🧪 Testing: {model_id} ({class_name})")
    print(f"   Text: {text}")
    print(f"{'='*50}")

    start = time.time()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call_model, class_name, text)
            result = future.result(timeout=timeout)

        elapsed = time.time() - start

        if not result.get("success"):
            print(f"  ❌ FAIL — synthesis returned error: {result.get('error')}")
            print(f"  ⏱  {elapsed:.1f}s")
            return False

        audio_b64 = result.get("audio_base64", "")
        sr = result.get("sample_rate", 0)

        if not audio_b64 or len(audio_b64) < 100:
            print(f"  ❌ FAIL — audio_base64 is empty or too short ({len(audio_b64)} chars)")
            print(f"  ⏱  {elapsed:.1f}s")
            return False

        # Save the audio locally so it can be listened to
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"{model_id}.wav")
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(audio_b64))
        print(f"  💾 Saved: {os.path.abspath(out_path)}")

        print(f"  ✅ PASS — {len(audio_b64):,} chars base64, sr={sr}")
        print(f"  ⏱  {elapsed:.1f}s")
        return True

    except TimeoutError:
        elapsed = time.time() - start
        print(f"  ❌ FAIL — timed out after {timeout}s (container may be crash-looping)")
        print(f"  ⏱  {elapsed:.1f}s")
        return False

    except Exception as e:
        elapsed = time.time() - start
        print(f"  ❌ FAIL — exception: {e}")
        print(f"  ⏱  {elapsed:.1f}s")
        return False


def main():
    args = _build_parser().parse_args()

    # Use the local MODEL_REGISTRY (already populated via auto-discovery)
    registry = MODEL_REGISTRY
    print(f"✅ Found {len(registry)} registered models: {', '.join(sorted(registry.keys()))}")

    # Filter to requested models (argparse already validates choices)
    if args.models:
        registry = {k: v for k, v in registry.items() if k in args.models}

    if not registry:
        print("❌ No models to test.")
        sys.exit(1)

    print(f"\n🔤 Text: {args.text}")
    print(f"⏱  Timeout: {args.timeout}s per model")

    # Run smoke tests
    results: dict[str, bool] = {}
    for model_id, info in registry.items():
        class_name = info["class_name"]
        passed = smoke_test_model(model_id, class_name, args.text, args.timeout)
        results[model_id] = passed

    # Summary
    passed_list = [m for m, ok in results.items() if ok]
    failed_list = [m for m, ok in results.items() if not ok]

    print(f"\n{'='*50}")
    print(f"📊 Results: {len(passed_list)} passed, {len(failed_list)} failed out of {len(results)}")
    print(f"{'='*50}")

    if passed_list:
        print(f"  ✅ Passed: {', '.join(passed_list)}")
    if failed_list:
        print(f"  ❌ Failed: {', '.join(failed_list)}")

    sys.exit(1 if failed_list else 0)


if __name__ == "__main__":
    main()
