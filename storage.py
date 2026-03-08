"""Arabic TTS Arena — Vote data model and persistence (JSONL + audio)."""

import json
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
    timestamp: str = field(default_factory=_now_iso)


# =============================================================================
# Paths (inside the Modal Volume mounted at /data)
# =============================================================================

VOTES_FILE = Path("/data/votes.jsonl")
AUDIO_DIR = Path("/data/audio")


# =============================================================================
# Read / Write helpers
# =============================================================================

def load_votes() -> list[Vote]:
    """Read all votes from the JSONL file."""
    if not VOTES_FILE.exists():
        return []
    votes = []
    with open(VOTES_FILE, "r") as f:
        for line in f:
            if line.strip():
                votes.append(Vote(**json.loads(line)))
    return votes


def append_vote(vote: Vote):
    """Append a single vote to the JSONL file."""
    VOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(VOTES_FILE, "a") as f:
        f.write(json.dumps(asdict(vote), ensure_ascii=False) + "\n")


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
    """Normalize Arabic text for cache-key matching.

    1. Unicode NFC normalization
    2. Strip leading/trailing whitespace
    3. Collapse internal whitespace runs to a single space
    4. Remove Arabic diacritics (tashkeel) — U+064B..U+065F, U+0670
    """
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    # Remove common Arabic diacritical marks (tashkeel)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    return text


# =============================================================================
# Audio cache index  —  (normalized_text, model_id) → audio_path
# =============================================================================

def build_audio_cache() -> dict[tuple[str, str], str]:
    """Scan votes.jsonl and build an in-memory lookup of cached audio.

    Returns a dict mapping (normalized_text, model_id) → audio_path.
    When multiple votes exist for the same key, the *last* entry wins
    (most recent audio).
    """
    cache: dict[tuple[str, str], str] = {}
    for vote in load_votes():
        norm = normalize_text(vote.text)
        if vote.audio_path_a:
            cache[(norm, vote.model_a)] = vote.audio_path_a
        if vote.audio_path_b:
            cache[(norm, vote.model_b)] = vote.audio_path_b
    return cache
