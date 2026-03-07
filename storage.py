"""Arabic TTS Arena — Vote data model and persistence (JSONL + audio)."""

import json
import base64
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
