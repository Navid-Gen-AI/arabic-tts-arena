"""
Post-deploy smoke test — verifies every registered TTS model can synthesize.

Runs after `modal deploy` in CI to catch container boot failures, missing
dependencies, or broken model weights before the frontend serves traffic.

Each model gets a hard timeout (default 180 s).  If the container keeps
crashing / respawning and never returns, the test fails fast instead of
hanging forever.

Usage:
    python scripts/smoke_test.py                         # test all models
    python scripts/smoke_test.py chatterbox habibi_tts   # test specific models

Exit codes:
    0  — all models passed
    1  — one or more models failed
"""

import sys
import time
import modal
from concurrent.futures import ThreadPoolExecutor, TimeoutError

APP_NAME = "arabic-tts-arena"
TEST_TEXT = "مرحباً"  # short Arabic phrase — fast to synthesize
TIMEOUT_SECONDS = 180  # hard cap per model (includes cold start + inference)


def _call_model(class_name: str) -> dict:
    """Run the remote synthesis call (executed in a thread so we can time-out)."""
    cls = modal.Cls.from_name(APP_NAME, class_name)
    return cls().synthesize.remote(TEST_TEXT)


def smoke_test_model(model_id: str, class_name: str) -> bool:
    """Call synthesize() on a single model with a hard timeout."""
    print(f"\n{'='*50}")
    print(f"🧪 Testing: {model_id} ({class_name})")
    print(f"{'='*50}")

    start = time.time()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call_model, class_name)
            result = future.result(timeout=TIMEOUT_SECONDS)

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

        print(f"  ✅ PASS — {len(audio_b64):,} chars base64, sr={sr}")
        print(f"  ⏱  {elapsed:.1f}s")
        return True

    except TimeoutError:
        elapsed = time.time() - start
        print(f"  ❌ FAIL — timed out after {TIMEOUT_SECONDS}s (container may be crash-looping)")
        print(f"  ⏱  {elapsed:.1f}s")
        return False

    except Exception as e:
        elapsed = time.time() - start
        print(f"  ❌ FAIL — exception: {e}")
        print(f"  ⏱  {elapsed:.1f}s")
        return False


def main():
    # Fetch the model registry from the deployed backend
    print("📡 Fetching model registry from Modal backend...")
    try:
        service = modal.Cls.from_name(APP_NAME, "ArenaService")
        registry = service().get_model_registry.remote()
    except Exception as e:
        print(f"❌ Could not reach ArenaService: {e}")
        sys.exit(1)

    if not registry:
        print("❌ Model registry is empty — nothing to test.")
        sys.exit(1)

    print(f"✅ Found {len(registry)} models: {', '.join(registry.keys())}")

    # Filter to specific models if args provided
    targets = sys.argv[1:]
    if targets:
        missing = [m for m in targets if m not in registry]
        if missing:
            print(f"⚠️  Unknown model(s): {', '.join(missing)}")
        registry = {k: v for k, v in registry.items() if k in targets}

    if not registry:
        print("❌ No models to test.")
        sys.exit(1)

    # Run smoke tests
    results: dict[str, bool] = {}
    for model_id, info in registry.items():
        class_name = info["class_name"]
        passed = smoke_test_model(model_id, class_name)
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
