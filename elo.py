"""Arabic TTS Arena — Bradley-Terry rating and leaderboard computation.

Uses maximum-likelihood estimation (MLE) of Bradley-Terry model parameters
to derive ratings from pairwise votes.  Ties (both_good / both_bad) are
handled via the half-win approach: each tie counts as 0.5 wins for both
players, which is the standard approximation used by LMArena / Chatbot Arena.

Ratings are reported on a familiar 0–2000-ish scale centred at 1000 (the
base rating for a model with no votes).
"""

from __future__ import annotations

import math
import random as _random
from collections import defaultdict

from storage import Vote

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_RATING = 1000.0          # starting / anchor rating
_SCALE = 400.0                # Elo-style scale factor (log-base-10)
_BT_MAX_ITER = 200            # max Newton iterations for BT fit
_BT_TOL = 1e-6                # convergence tolerance
_BOOTSTRAP_ROUNDS = 200       # resamples for 95 % confidence interval


# ---------------------------------------------------------------------------
# Bradley-Terry MLE (iterative algorithm)
# ---------------------------------------------------------------------------

def _fit_bradley_terry(
    model_ids: list[str],
    win_matrix: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute BT strength parameters via iterative MM algorithm.

    Uses the standard minorisation-maximisation (MM) update from
    Hunter (2004) which is guaranteed to converge.

    Args:
        model_ids: list of all model identifiers.
        win_matrix: win_matrix[i][j] = number of times model i beat model j
                    (ties contribute 0.5 to each side).

    Returns:
        dict mapping model_id → strength parameter (positive float).
    """
    n = len(model_ids)
    if n == 0:
        return {}

    # Initialise all strengths equally
    p: dict[str, float] = {mid: 1.0 for mid in model_ids}

    # Pre-compute total games between each pair
    games: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for i in model_ids:
        for j in model_ids:
            if i != j:
                games[i][j] = win_matrix[i][j] + win_matrix[j][i]

    for _iteration in range(_BT_MAX_ITER):
        p_old = dict(p)

        for i in model_ids:
            numerator = sum(win_matrix[i][j] for j in model_ids if j != i)
            denominator = 0.0
            for j in model_ids:
                if j == i:
                    continue
                n_ij = games[i][j]
                if n_ij > 0:
                    denominator += n_ij / (p[i] + p[j])

            if denominator > 0 and numerator > 0:
                p[i] = numerator / denominator
            # else: model has no wins — keep previous value (will get base rating)

        # Normalise so geometric mean = 1 (prevents drift)
        log_mean = sum(math.log(v) for v in p.values()) / n
        for mid in model_ids:
            p[mid] /= math.exp(log_mean)

        # Check convergence
        max_diff = max(abs(p[mid] - p_old[mid]) for mid in model_ids)
        if max_diff < _BT_TOL:
            break

    return p


def _strengths_to_ratings(
    strengths: dict[str, float],
    base: float = BASE_RATING,
    scale: float = _SCALE,
) -> dict[str, float]:
    """Convert raw BT strength parameters to human-readable ratings.

    The mapping is:  rating = base + scale * log10(strength)
    Since strengths are normalised so their geometric mean = 1,
    the average rating will equal `base`.
    """
    ratings: dict[str, float] = {}
    for mid, s in strengths.items():
        ratings[mid] = base + scale * math.log10(s) if s > 0 else base
    return ratings


# ---------------------------------------------------------------------------
# ModelStats — public interface (unchanged field names for compatibility)
# ---------------------------------------------------------------------------

class ModelStats:
    """Bradley-Terry statistics for a single model."""

    def __init__(self, model_id: str, name: str, elo: float = BASE_RATING, model_url: str = "", gpu: str = ""):
        self.model_id = model_id
        self.name = name
        self.model_url = model_url
        self.gpu = gpu           # GPU type (e.g. "T4", "A10G", "A100-40GB")
        self.elo = elo          # field kept as "elo" for JSON/frontend compat
        self.ci = 0.0           # 95 % confidence-interval half-width (±)
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self._latency_samples: list[float] = []  # raw latency values for averaging

    @property
    def battles(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def win_rate(self) -> float:
        return round(self.wins / self.battles * 100, 1) if self.battles > 0 else 0.0

    @property
    def avg_latency(self) -> float | None:
        """Average synthesis latency in seconds, or None if no data."""
        if not self._latency_samples:
            return None
        return sum(self._latency_samples) / len(self._latency_samples)


# ---------------------------------------------------------------------------
# Leaderboard computation
# ---------------------------------------------------------------------------

def compute_leaderboard(
    votes: list[Vote],
    registered_models: dict[str, str],
) -> dict[str, ModelStats]:
    """Fit a Bradley-Terry model to all votes and return per-model stats.

    Args:
        votes: list of Vote records.
        registered_models: MODEL_REGISTRY dict (model_id → display_name or dict).
    """
    stats: dict[str, ModelStats] = {}

    # Seed stats for every registered model
    for model_id, info in registered_models.items():
        if isinstance(info, dict):
            display_name = info.get("display_name", model_id)
            model_url = info.get("model_url", "")
            gpu = info.get("gpu", "")
        else:
            display_name = info
            model_url = ""
            gpu = ""
        stats[model_id] = ModelStats(model_id=model_id, name=display_name, model_url=model_url, gpu=gpu)

    # Build win matrix from votes (ties → 0.5 win each)
    win_matrix: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for vote in votes:
        # Handle models that aren't in the current registry
        for mid in (vote.model_a, vote.model_b):
            if mid not in stats:
                stats[mid] = ModelStats(model_id=mid, name=mid)

        a_id, b_id = vote.model_a, vote.model_b

        if vote.winner == "model_a":
            stats[a_id].wins += 1
            stats[b_id].losses += 1
            win_matrix[a_id][b_id] += 1.0
        elif vote.winner == "model_b":
            stats[b_id].wins += 1
            stats[a_id].losses += 1
            win_matrix[b_id][a_id] += 1.0
        elif vote.winner in ("both_good", "both_bad"):
            stats[a_id].ties += 1
            stats[b_id].ties += 1
            win_matrix[a_id][b_id] += 0.5
            win_matrix[b_id][a_id] += 0.5

        # Accumulate latency samples (only non-null = fresh synthesis)
        if vote.latency_a is not None:
            stats[a_id]._latency_samples.append(vote.latency_a)
        if vote.latency_b is not None:
            stats[b_id]._latency_samples.append(vote.latency_b)

    # Fit BT model on models that have at least one battle
    active_ids = [mid for mid, s in stats.items() if s.battles > 0]

    if active_ids:
        strengths = _fit_bradley_terry(active_ids, win_matrix)
        ratings = _strengths_to_ratings(strengths, base=BASE_RATING)
        for mid in active_ids:
            stats[mid].elo = ratings[mid]

    # ── Bootstrap 95 % confidence intervals ───────────────────────
    # Build a flat list of (model_a, model_b, score_a, score_b) for resampling
    vote_tuples: list[tuple[str, str, float, float]] = []
    for vote in votes:
        a_id, b_id = vote.model_a, vote.model_b
        if vote.winner == "model_a":
            vote_tuples.append((a_id, b_id, 1.0, 0.0))
        elif vote.winner == "model_b":
            vote_tuples.append((a_id, b_id, 0.0, 1.0))
        elif vote.winner in ("both_good", "both_bad"):
            vote_tuples.append((a_id, b_id, 0.5, 0.5))

    if len(vote_tuples) >= 5 and active_ids:
        rng = _random.Random(42)          # deterministic seed
        bootstrap_ratings: dict[str, list[float]] = defaultdict(list)

        for _ in range(_BOOTSTRAP_ROUNDS):
            sample = rng.choices(vote_tuples, k=len(vote_tuples))

            # Build win matrix for this bootstrap sample
            b_wins: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
            b_models: set[str] = set()
            for a, b, sa, sb in sample:
                b_wins[a][b] += sa
                b_wins[b][a] += sb
                b_models.add(a)
                b_models.add(b)

            b_ids = sorted(b_models)
            b_strengths = _fit_bradley_terry(b_ids, b_wins)
            b_ratings = _strengths_to_ratings(b_strengths, base=BASE_RATING)
            for mid in b_ids:
                bootstrap_ratings[mid].append(b_ratings[mid])

        for mid, samples in bootstrap_ratings.items():
            if mid in stats and len(samples) >= 10:
                samples.sort()
                lo = samples[int(len(samples) * 0.025)]
                hi = samples[int(len(samples) * 0.975)]
                stats[mid].ci = (hi - lo) / 2.0

    # Models with zero battles keep the default BASE_RATING (ci stays 0)

    return stats
