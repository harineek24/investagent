"""Scores past agent decisions once enough time has passed.

This closes the loop that src/memory.py sets up: decisions are saved with
price_at_decision, and once a decision is at least MIN_AGE_DAYS old, the
realized price is compared against the signal direction to determine
was_correct. That feeds get_agent_accuracy(), which the debate and
Portfolio Manager prompts use to weight agents by track record.
"""

from src.memory import get_unscored_decisions, update_decision_outcome, record_agent_score
from src.tools.api import get_current_price

MIN_AGE_DAYS = 30

# Moves smaller than this are treated as "no real move" for neutral calls.
NEUTRAL_BAND_PCT = 2.0


def _was_correct(signal: str, pct_change: float) -> bool:
    if signal == "bullish":
        return pct_change > NEUTRAL_BAND_PCT
    if signal == "bearish":
        return pct_change < -NEUTRAL_BAND_PCT
    return abs(pct_change) <= NEUTRAL_BAND_PCT


def score_pending_decisions(min_age_days: int = MIN_AGE_DAYS) -> int:
    """Score decisions old enough to evaluate. Returns count scored."""
    pending = get_unscored_decisions(min_age_days=min_age_days)
    scored = 0

    for decision in pending:
        current_price = get_current_price(decision["ticker"])
        if current_price is None or not decision["price_at_decision"]:
            continue

        pct_change = (current_price - decision["price_at_decision"]) / decision["price_at_decision"] * 100
        correct = _was_correct(decision["signal"], pct_change)
        outcome = "correct" if correct else "incorrect"

        update_decision_outcome(decision["id"], current_price, outcome)
        record_agent_score(decision["agent"], decision["ticker"], decision["signal"], correct)
        scored += 1

    return scored
