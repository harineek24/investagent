"""Shared state for the multi-agent hedge fund graph."""

from typing import Annotated
import operator
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


def merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dicts, with b overwriting a."""
    merged = a.copy()
    merged.update(b)
    return merged


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    tickers: list[str]
    portfolio: dict
    start_date: str
    end_date: str
    analyst_signals: Annotated[dict, merge_dicts]
    show_reasoning: bool
    # Agentic additions
    debate_required: bool          # True if analysts disagree enough
    debate_summary: dict           # Output from debate/synthesis node
    agreement_score: float         # 0.0-1.0, how much analysts agree
