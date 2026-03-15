"""Arabic TTS Arena — ArenaService (vote recording, queries) and leaderboard cron."""

import os
import json
import base64
import time
from datetime import datetime, timezone
from typing import Optional
from dataclasses import asdict

import random

import modal

from app import app, service_image, votes_volume
from models import MODEL_REGISTRY
from storage import Vote, load_votes, append_vote, save_audio, normalize_text, build_audio_cache
from elo import compute_leaderboard

# Probability of returning cached audio when a cache hit exists.
# Set < 1.0 so a fraction of requests still trigger fresh synthesis
# for variety (models use stochastic sampling).
CACHE_HIT_RATE = 0.7


# =============================================================================
# Arena Service
# =============================================================================

@app.cls(
    image=service_image,
    volumes={"/data": votes_volume},
    timeout=600,
)
class ArenaService:
    """Handles vote persistence and leaderboard computation on Modal."""

    @modal.enter()
    def _build_cache(self):
        """Build in-memory audio cache index from stored votes on container start."""
        self._audio_cache = build_audio_cache()
        print(f"✅ Audio cache loaded: {len(self._audio_cache)} entries")

    # -----------------------------------------------------------------
    # Synthesis with cache
    # -----------------------------------------------------------------

    @modal.method()
    def synthesize_or_cache(self, text: str, model_id: str) -> dict:
        """Return cached audio when available, otherwise synthesize fresh.

        With probability (1 - CACHE_HIT_RATE) a cache hit is deliberately
        skipped so users still hear varied outputs from stochastic models.

        For models that support multiple dialects, a dialect is randomly
        sampled and passed to synthesize() so users experience the full
        quality spectrum of each model.
        """
        norm = normalize_text(text)
        cache_key = (norm, model_id)

        # --- try cache ---
        if cache_key in self._audio_cache and random.random() < CACHE_HIT_RATE:
            audio_path = self._audio_cache[cache_key]
            if os.path.exists(audio_path):
                try:
                    with open(audio_path, "rb") as f:
                        audio_b64 = base64.b64encode(f.read()).decode()
                    print(f"⚡ Cache hit for ({model_id}, {norm[:40]}…)")
                    return {
                        "success": True,
                        "audio_base64": audio_b64,
                        "model_id": model_id,
                        "cached": True,
                    }
                except Exception as e:
                    print(f"⚠️ Cache read failed, falling through to synthesis: {e}")

        # --- cache miss → call the model ---
        try:
            model_info = MODEL_REGISTRY[model_id]
            ModelCls = modal.Cls.from_name("arabic-tts-arena", model_info["class_name"])

            # If the model supports dialects, randomly sample one
            dialects = model_info.get("dialects", [])

            t0 = time.perf_counter()
            if dialects:
                dialect = random.choice(dialects)
                print(f"🗣️ Sampling dialect '{dialect}' for {model_id}")
                result = ModelCls().synthesize.remote(text, dialect=dialect)
            else:
                result = ModelCls().synthesize.remote(text)
            latency = round(time.perf_counter() - t0, 2)

            if isinstance(result, dict):
                result["latency_seconds"] = latency
            return result
        except Exception as e:
            return {"success": False, "error": str(e), "model_id": model_id}

    # -----------------------------------------------------------------
    # Vote recording
    # -----------------------------------------------------------------

    @modal.method()
    def record_vote(
        self,
        session_id: str,
        text: str,
        model_a: str,
        model_b: str,
        winner: str,
        audio_a_base64: Optional[str] = None,
        audio_b_base64: Optional[str] = None,
        latency_a: Optional[float] = None,
        latency_b: Optional[float] = None,
    ) -> dict:
        """Record a vote, save audio files, and commit the volume."""
        try:
            audio_path_a = save_audio(session_id, "a", audio_a_base64) if audio_a_base64 else None
            audio_path_b = save_audio(session_id, "b", audio_b_base64) if audio_b_base64 else None

            vote = Vote(
                session_id=session_id,
                text=text,
                model_a=model_a,
                model_b=model_b,
                winner=winner,
                audio_path_a=audio_path_a,
                audio_path_b=audio_path_b,
                latency_a=latency_a,
                latency_b=latency_b,
            )
            append_vote(vote)
            votes_volume.commit()

            # Update in-memory cache so subsequent requests benefit immediately
            norm = normalize_text(text)
            if audio_path_a:
                self._audio_cache[(norm, model_a)] = audio_path_a
            if audio_path_b:
                self._audio_cache[(norm, model_b)] = audio_path_b

            return {"success": True}
        except Exception as e:
            print(f"❌ record_vote failed: {e}")
            return {"success": False, "error": str(e)}

    @modal.method()
    def get_leaderboard(self) -> dict:
        """Compute and return the current leaderboard."""
        stats = compute_leaderboard(load_votes(), MODEL_REGISTRY)
        return {
            mid: {
                "name": s.name,
                "model_url": s.model_url,
                "elo": round(s.elo, 1),
                "ci": round(s.ci, 1),
                "wins": s.wins,
                "losses": s.losses,
                "ties": s.ties,
                "battles": s.battles,
                "win_rate": s.win_rate,
                "avg_latency": round(s.avg_latency, 1) if s.avg_latency is not None else None,
            }
            for mid, s in stats.items()
        }

    @modal.method()
    def get_model_registry(self) -> dict:
        """Return the current MODEL_REGISTRY so the frontend can discover models.

        Returns dict like:
            {"chatterbox": {"class_name": "ChatterboxModel", "display_name": "Chatterbox"}, ...}
        """
        return dict(MODEL_REGISTRY)

    @modal.method()
    def get_vote_count(self) -> int:
        return len(load_votes())

    @modal.method()
    def export_votes(self) -> list[dict]:
        """Return all votes as dicts (for dataset export scripts)."""
        return [asdict(v) for v in load_votes()]

    @modal.method()
    def get_audio_file(self, audio_path: str) -> Optional[str]:
        """Return a single audio file as a base64 string."""
        if audio_path and os.path.exists(audio_path):
            with open(audio_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        return None


# =============================================================================
# Cron Job — push leaderboard.json to HuggingFace Space daily
# =============================================================================

@app.function(
    image=service_image,
    volumes={"/data": votes_volume},
    schedule=modal.Cron("0 0 * * *"),
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
def update_leaderboard_file():
    """Recompute Bradley-Terry ratings from all votes and upload leaderboard.json."""
    from huggingface_hub import HfApi

    votes = load_votes()
    stats = compute_leaderboard(votes, MODEL_REGISTRY)
    ranked = sorted(stats.values(), key=lambda s: s.elo, reverse=True)

    leaderboard_data = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "total_votes": len(votes),
        "models": [
            {
                "rank": i + 1,
                "model_id": s.model_id,
                "name": s.name,
                "model_url": s.model_url,
                "elo": round(s.elo, 1),
                "ci": round(s.ci, 1),
                "wins": s.wins,
                "losses": s.losses,
                "ties": s.ties,
                "battles": s.battles,
                "win_rate": s.win_rate,
                "avg_latency": round(s.avg_latency, 1) if s.avg_latency is not None else None,
            }
            for i, s in enumerate(ranked)
        ],
    }

    HfApi().upload_file(
        path_or_fileobj=json.dumps(leaderboard_data, indent=2).encode(),
        path_in_repo="leaderboard.json",
        repo_id="Navid-AI/Arabic-TTS-Arena",
        repo_type="space",
    )
    print(f"✅ Leaderboard updated — {len(votes)} votes, {len(ranked)} models")
