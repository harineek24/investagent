import pytest
from src.agents.debate import check_agreement


def _signals(*pairs):
    """Build analyst_signals: list of (agent_name, ticker, signal)."""
    out = {}
    for agent, ticker, signal in pairs:
        out.setdefault(agent, {})[ticker] = {"signal": signal}
    return out


def test_unanimous_agreement_no_debate():
    state = {
        "tickers": ["AAPL"],
        "analyst_signals": _signals(
            ("Value Analyst", "AAPL", "bullish"),
            ("Growth Analyst", "AAPL", "bullish"),
            ("Technical Analyst", "AAPL", "bullish"),
        ),
    }
    result = check_agreement(state)
    assert result["agreement_score"] == 1.0
    assert result["debate_required"] is False


def test_split_signals_trigger_debate():
    state = {
        "tickers": ["AAPL"],
        "analyst_signals": _signals(
            ("Value Analyst", "AAPL", "bullish"),
            ("Growth Analyst", "AAPL", "bearish"),
            ("Technical Analyst", "AAPL", "neutral"),
        ),
    }
    result = check_agreement(state)
    assert result["agreement_score"] == pytest.approx(1 / 3)
    assert result["debate_required"] is True


def test_risk_and_portfolio_manager_signals_excluded():
    state = {
        "tickers": ["AAPL"],
        "analyst_signals": _signals(
            ("Value Analyst", "AAPL", "bullish"),
            ("Growth Analyst", "AAPL", "bullish"),
            ("Risk Manager", "AAPL", "bearish"),
            ("Portfolio Manager", "AAPL", "bearish"),
        ),
    }
    result = check_agreement(state)
    assert result["agreement_score"] == 1.0
    assert result["debate_required"] is False


def test_no_analyst_signals_defaults_to_full_agreement():
    state = {"tickers": ["AAPL"], "analyst_signals": {}}
    result = check_agreement(state)
    assert result == {"agreement_score": 1.0, "debate_required": False}
