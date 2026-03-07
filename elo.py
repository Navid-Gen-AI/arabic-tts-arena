"""Arabic TTS Arena — ELO calculation and leaderboard computation."""

from storage import Vote


def calculate_elo(
    winner_elo: float, loser_elo: float, k: float = 32,
) -> tuple[float, float]:
    """Return (new_winner_elo, new_loser_elo) after a match."""
    expected_w = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    expected_l = 1 - expected_w
    return (
        winner_elo + k * (1 - expected_w),
        loser_elo - k * expected_l,
    )


def calculate_elo_draw(
    elo_a: float, elo_b: float, k: float = 32,
) -> tuple[float, float]:
    """Return (new_elo_a, new_elo_b) after a draw (0.5 result for each)."""
    expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    expected_b = 1 - expected_a
    return (
        elo_a + k * (0.5 - expected_a),
        elo_b + k * (0.5 - expected_b),
    )


class ModelStats:
    """ELO statistics for a single model."""

    def __init__(self, model_id: str, name: str, elo: float = 1500.0):
        self.model_id = model_id
        self.name = name
        self.elo = elo
        self.wins = 0
        self.losses = 0
        self.ties = 0

    @property
    def battles(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def win_rate(self) -> float:
        return round(self.wins / self.battles * 100, 1) if self.battles > 0 else 0.0


def compute_leaderboard(
    votes: list[Vote],
    registered_models: dict[str, str],
) -> dict[str, ModelStats]:
    """Replay all votes chronologically to produce per-model ELO stats.

    Args:
        votes: list of Vote records.
        registered_models: MODEL_REGISTRY dict (model_id → display_name).
    """
    stats: dict[str, ModelStats] = {}

    # Seed stats for every registered model
    for model_id, info in registered_models.items():
        # Support both old format (str) and new format (dict)
        if isinstance(info, dict):
            display_name = info.get("display_name", model_id)
        else:
            display_name = info
        stats[model_id] = ModelStats(model_id=model_id, name=display_name)

    for vote in votes:
        # Handle models that aren't in the current registry (removed/renamed)
        for mid in (vote.model_a, vote.model_b):
            if mid not in stats:
                stats[mid] = ModelStats(model_id=mid, name=mid)

        a, b = stats[vote.model_a], stats[vote.model_b]

        if vote.winner == "model_a":
            a.wins += 1
            b.losses += 1
            a.elo, b.elo = calculate_elo(a.elo, b.elo)
        elif vote.winner == "model_b":
            b.wins += 1
            a.losses += 1
            b.elo, a.elo = calculate_elo(b.elo, a.elo)
        elif vote.winner in ("both_good", "both_bad"):
            a.ties += 1
            b.ties += 1
            a.elo, b.elo = calculate_elo_draw(a.elo, b.elo)

    return stats
