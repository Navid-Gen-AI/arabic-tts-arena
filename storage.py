"""Arabic TTS Arena — Vote data model and persistence (per-vote JSON files + audio).

Each vote is written to its own JSON file under /data/votes/ to avoid
race conditions when multiple Modal containers commit the volume
concurrently.  The legacy /data/votes.jsonl is still read (if present)
for backward compatibility.
"""

import json
import os
import time
import uuid
import base64
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# Vote Data Model
# =============================================================================

@dataclass
class Vote:
    """A single vote record."""
    session_id: str
    text: str
    model_a: str
    model_b: str
    winner: str  # 'model_a' | 'model_b' | 'both_good' | 'both_bad'
    audio_path_a: Optional[str] = None
    audio_path_b: Optional[str] = None
    latency_a: Optional[float] = None   # synthesis time in seconds (None = cached / legacy)
    latency_b: Optional[float] = None
    timestamp: str = field(default_factory=_now_iso)


# =============================================================================
# Paths (inside the Modal Volume mounted at /data)
# =============================================================================

VOTES_DIR = Path("/data/votes")            # NEW: one JSON file per vote
LEGACY_VOTES_FILE = Path("/data/votes.jsonl")  # OLD: kept for backward compat
AUDIO_DIR = Path("/data/audio")


# =============================================================================
# Read / Write helpers
# =============================================================================

def _parse_vote(data: dict) -> Vote:
    """Safely construct a Vote from a dict, ignoring unknown keys."""
    known = {f.name for f in Vote.__dataclass_fields__.values()}
    return Vote(**{k: v for k, v in data.items() if k in known})


def load_votes() -> list[Vote]:
    """Load all votes from individual JSON files *and* the legacy JSONL.

    Returns a combined list sorted by timestamp.  Duplicates (same
    session_id) are removed so migrated votes aren't double-counted.
    """
    seen_sessions: set[str] = set()
    votes: list[Vote] = []

    # 1. Legacy votes.jsonl (read first so new files take precedence)
    if LEGACY_VOTES_FILE.exists():
        try:
            with open(LEGACY_VOTES_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        vote = _parse_vote(data)
                        if vote.session_id not in seen_sessions:
                            seen_sessions.add(vote.session_id)
                            votes.append(vote)
                    except Exception:
                        continue  # skip corrupted lines
        except Exception:
            pass

    # 2. Individual vote files (one JSON per vote)
    if VOTES_DIR.exists():
        for filepath in sorted(VOTES_DIR.glob("*.json")):
            try:
                with open(filepath, "r") as f:
                    data = json.loads(f.read())
                vote = _parse_vote(data)
                if vote.session_id not in seen_sessions:
                    seen_sessions.add(vote.session_id)
                    votes.append(vote)
            except Exception:
                continue  # skip corrupted files

    votes.sort(key=lambda v: v.timestamp)
    return votes


def append_vote(vote: Vote) -> str:
    """Write a single vote as a unique JSON file and return its path.

    Each vote gets its own file so concurrent Modal containers can
    never overwrite each other's data on volume commit.
    The file is fsynced to ensure durability before commit().
    """
    VOTES_DIR.mkdir(parents=True, exist_ok=True)

    ts_ms = int(time.time() * 1000)
    unique_id = uuid.uuid4().hex[:12]
    filepath = VOTES_DIR / f"{ts_ms}_{unique_id}.json"

    data = json.dumps(asdict(vote), ensure_ascii=False)
    with open(filepath, "w") as f:
        f.write(data + "\n")
        f.flush()
        os.fsync(f.fileno())

    return str(filepath)


def save_audio(session_id: str, suffix: str, audio_base64: str) -> str:
    """Persist an audio file and return its path."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIO_DIR / f"{session_id}_{suffix}.wav"
    with open(path, "wb") as f:
        f.write(base64.b64decode(audio_base64))
    return str(path)


# =============================================================================
# Text normalisation (for cache key consistency)
# =============================================================================


def normalize_text(text: str) -> str:
    """Canonicalize text for exact cache-key matching.

    Normalization steps:
        - Unicode NFC normalization (e.g. combine letters + diacritics)
        - Trim leading/trailing whitespace
        - Collapse internal whitespace to single spaces
    """
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


# =============================================================================
# Audio cache index  —  (normalized_text, model_id) → list[audio_path]
# =============================================================================

# Maximum number of audio variants kept per (text, model) pair in the
# in-memory index. Older variants beyond this cap are dropped from the
# random-selection pool (their files are NOT deleted — votes still
# reference them) so a viral prompt can't bloat the index unbounded.
MAX_CACHE_VARIANTS_PER_KEY = 10


def _file_size(path: str) -> Optional[int]:
    """Return file size in bytes, or None if unreadable."""
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _append_variant(bucket: list[str], audio_path: str) -> None:
    """Append `audio_path` to `bucket`, dedup by file size, cap to limit.

    WAV files are uncompressed PCM, so byte-identical audio always has
    the same file size, and different audio almost never does.  This is
    a fast O(1)-per-file metadata check (no file reads) that works
    reliably for dedup on network-mounted volumes like Modal's.
    """
    new_size = _file_size(audio_path)
    if new_size is None:
        return  # unreadable file, skip

    for existing in bucket:
        if _file_size(existing) == new_size:
            return  # same size = same audio content for WAV

    bucket.append(audio_path)
    if len(bucket) > MAX_CACHE_VARIANTS_PER_KEY:
        del bucket[: len(bucket) - MAX_CACHE_VARIANTS_PER_KEY]


def build_audio_cache() -> dict[tuple[str, str], list[str]]:
    """Scan all votes and build an in-memory lookup of cached audio variants.

    Returns a dict mapping (normalized_text, model_id) → list[audio_path].
    Multiple distinct audios for the same (text, model) are all retained
    (up to MAX_CACHE_VARIANTS_PER_KEY) so the service can randomly pick
    one at hit time, defeating byte-level fingerprinting attacks.
    Votes are processed in timestamp order (see load_votes), so when the
    list is trimmed the most recent variants survive.
    """
    cache: dict[tuple[str, str], list[str]] = {}
    for vote in load_votes():
        norm = normalize_text(vote.text)
        if vote.audio_path_a:
            _append_variant(cache.setdefault((norm, vote.model_a), []), vote.audio_path_a)
        if vote.audio_path_b:
            _append_variant(cache.setdefault((norm, vote.model_b), []), vote.audio_path_b)
    return cache
