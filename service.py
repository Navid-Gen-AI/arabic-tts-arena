"""Arabic TTS Arena — ArenaService (vote recording, queries) and leaderboard cron."""

import os
import json
import base64
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional
from dataclasses import asdict

import random
import modal

from app import app, service_image, votes_volume
from models import MODEL_REGISTRY
from storage import (
    Vote,
    load_votes,
    append_vote,
    save_audio,
    normalize_text,
    build_audio_cache,
    legacy_normalize_text,
    list_vote_file_records,
    update_vote_file,
    collect_referenced_audio_paths,
    LEGACY_VOTES_FILE,
)
from elo import compute_leaderboard

# Probability of returning cached audio when a cache hit exists.
# Set < 1.0 so a fraction of requests still trigger fresh synthesis
# for variety (models use stochastic sampling).
CACHE_HIT_RATE = 0.85


def _parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


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

            t0 = time.perf_counter()
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
                "gpu": s.gpu,
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
        return {
            model_id: dict(info)
            for model_id, info in MODEL_REGISTRY.items()
            if not info.get("retired", False)
        }

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

    @modal.method()
    def inspect_or_purge_suspicious_cache_entries(
        self,
        cutoff_timestamp: Optional[str] = None,
        purge: bool = False,
        purge_all_before: bool = False,
        delete_audio_files: bool = True,
    ) -> dict:
        """Inspect or purge suspicious cached audio entries from vote history.

        Suspicious entries are detected when multiple distinct exact prompts map
        to the same legacy cache key for the same model. Since the pre-fix race
        cannot be reconstructed from metadata alone, `purge_all_before=True`
        invalidates every cached audio reference before the cutoff.
        """
        cutoff_dt = _parse_timestamp(cutoff_timestamp)
        if purge_all_before and cutoff_dt is None:
            return {
                "success": False,
                "error": "cutoff_timestamp is required when purge_all_before=True",
            }

        records = list_vote_file_records()
        legacy_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        scoped_entries: list[dict] = []
        file_data_map: dict[str, dict] = {}

        for filepath, data, vote in records:
            vote_dt = _parse_timestamp(vote.timestamp)
            if cutoff_dt is not None and vote_dt is not None and vote_dt >= cutoff_dt:
                continue

            file_key = str(filepath)
            file_data_map[file_key] = dict(data)

            for side, model_id, audio_path in (
                ("a", vote.model_a, vote.audio_path_a),
                ("b", vote.model_b, vote.audio_path_b),
            ):
                if not audio_path:
                    continue
                entry = {
                    "vote_file": file_key,
                    "side": side,
                    "model_id": model_id,
                    "timestamp": vote.timestamp,
                    "text": vote.text,
                    "canonical_text": normalize_text(vote.text),
                    "legacy_text": legacy_normalize_text(vote.text),
                    "audio_path": audio_path,
                }
                scoped_entries.append(entry)
                legacy_groups[(entry["legacy_text"], model_id)].append(entry)

        collision_groups: list[dict] = []
        collision_entries: list[dict] = []
        for (legacy_text, model_id), entries in legacy_groups.items():
            canonical_texts = sorted({entry["canonical_text"] for entry in entries})
            if len(canonical_texts) <= 1:
                continue
            collision_groups.append(
                {
                    "model_id": model_id,
                    "legacy_text": legacy_text,
                    "canonical_texts": canonical_texts,
                    "entries": [
                        {
                            "vote_file": entry["vote_file"],
                            "side": entry["side"],
                            "timestamp": entry["timestamp"],
                            "text": entry["text"],
                            "audio_path": entry["audio_path"],
                        }
                        for entry in entries
                    ],
                }
            )
            collision_entries.extend(entries)

        target_entries = scoped_entries if purge_all_before else collision_entries
        unique_target_files = {entry["vote_file"] for entry in target_entries}
        unique_target_audio = {entry["audio_path"] for entry in target_entries}

        report = {
            "success": True,
            "purge": purge,
            "purge_all_before": purge_all_before,
            "cutoff_timestamp": cutoff_timestamp,
            "legacy_vote_file_present": LEGACY_VOTES_FILE.exists(),
            "scanned_vote_files": len(records),
            "scanned_audio_entries": len(scoped_entries),
            "collision_group_count": len(collision_groups),
            "collision_entry_count": len(collision_entries),
            "target_entry_count": len(target_entries),
            "target_vote_file_count": len(unique_target_files),
            "target_audio_file_count": len(unique_target_audio),
            "collision_groups": collision_groups,
            "warning": (
                "Race-poisoned cache entries are not directly detectable from vote metadata; "
                "use purge_all_before with a cutoff to invalidate all pre-fix cached audio."
            ),
        }

        if not purge:
            return report

        updated_files: set[str] = set()
        purged_audio_paths: set[str] = set()
        for entry in target_entries:
            payload = file_data_map.get(entry["vote_file"])
            if payload is None:
                continue
            field_name = f"audio_path_{entry['side']}"
            current_path = payload.get(field_name)
            if not current_path:
                continue
            purged_audio_paths.add(current_path)
            payload[field_name] = None
            updated_files.add(entry["vote_file"])

        for file_key in updated_files:
            update_vote_file(file_key, file_data_map[file_key])

        removed_audio_files: list[str] = []
        if delete_audio_files and purged_audio_paths:
            still_referenced = collect_referenced_audio_paths()
            for audio_path in sorted(purged_audio_paths):
                if audio_path in still_referenced or not os.path.exists(audio_path):
                    continue
                try:
                    os.remove(audio_path)
                    removed_audio_files.append(audio_path)
                except OSError:
                    continue

        self._audio_cache = build_audio_cache()
        votes_volume.commit()

        report.update(
            {
                "updated_vote_files": sorted(updated_files),
                "updated_vote_file_count": len(updated_files),
                "removed_audio_files": removed_audio_files,
                "removed_audio_file_count": len(removed_audio_files),
                "refreshed_cache_entries": len(self._audio_cache),
            }
        )
        return report


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

    # Active models ranked first; retired models appended at the bottom.
    active = [s for s in ranked if not MODEL_REGISTRY.get(s.model_id, {}).get("retired", False)]
    retired = [s for s in ranked if MODEL_REGISTRY.get(s.model_id, {}).get("retired", False)]
    ordered = active + retired

    leaderboard_data = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "total_votes": len(votes),
        "models": [
            {
                "rank": i + 1,
                "model_id": s.model_id,
                "name": s.name,
                "model_url": s.model_url,
                "gpu": s.gpu,
                "elo": round(s.elo, 1),
                "ci": round(s.ci, 1),
                "wins": s.wins,
                "losses": s.losses,
                "ties": s.ties,
                "battles": s.battles,
                "win_rate": s.win_rate,
                "avg_latency": round(s.avg_latency, 1) if s.avg_latency is not None else None,
                "open_weight": s.open_weight,
                "retired": MODEL_REGISTRY.get(s.model_id, {}).get("retired", False),
            }
            for i, s in enumerate(ordered)
        ],
    }

    HfApi().upload_file(
        path_or_fileobj=json.dumps(leaderboard_data, indent=2).encode(),
        path_in_repo="leaderboard.json",
        repo_id="Navid-AI/Arabic-TTS-Arena",
        repo_type="space",
    )
    print(f"✅ Leaderboard updated — {len(votes)} votes, {len(ranked)} models")
