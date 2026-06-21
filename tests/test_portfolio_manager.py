from src.agents.portfolio_manager import _majority_vote_fallback


def _signals(**per_agent):
    """Helper: {"Value Analyst": ("bullish", 70), ...} -> analyst_signals dict for one ticker."""
    return {
        agent: {"AAPL": {"signal": signal, "confidence": conf}}
        for agent, (signal, conf) in per_agent.items()
    }


def test_majority_vote_no_signals_returns_hold():
    result = _majority_vote_fallback("AAPL", {}, {}, {})
    assert result["action"] == "hold"
    assert result["quantity"] == 0
    assert result["confidence"] == 10


def test_majority_vote_bullish_consensus_buys():
    analyst_signals = _signals(
        **{
            "Value Analyst": ("bullish", 80),
            "Growth Analyst": ("bullish", 70),
            "Technical Analyst": ("bearish", 30),
        }
    )
    risk_data = {"AAPL": {"max_shares": 30}}
    portfolio = {"positions": {}}

    result = _majority_vote_fallback("AAPL", analyst_signals, risk_data, portfolio)
    assert result["action"] == "buy"
    assert result["quantity"] == 10  # 30 // 3


def test_majority_vote_bearish_consensus_with_existing_position_sells():
    analyst_signals = _signals(
        **{
            "Value Analyst": ("bearish", 80),
            "Growth Analyst": ("bearish", 70),
        }
    )
    risk_data = {"AAPL": {"max_shares": 30}}
    portfolio = {"positions": {"AAPL": {"shares": 30}}}

    result = _majority_vote_fallback("AAPL", analyst_signals, risk_data, portfolio)
    assert result["action"] == "sell"
    assert result["quantity"] == 10  # 30 // 3


def test_majority_vote_bearish_consensus_without_position_holds():
    analyst_signals = _signals(
        **{
            "Value Analyst": ("bearish", 80),
            "Growth Analyst": ("bearish", 70),
        }
    )
    risk_data = {"AAPL": {"max_shares": 30}}
    portfolio = {"positions": {}}

    result = _majority_vote_fallback("AAPL", analyst_signals, risk_data, portfolio)
    assert result["action"] == "hold"


def test_majority_vote_low_confidence_holds():
    analyst_signals = _signals(
        **{
            "Value Analyst": ("bullish", 30),
            "Growth Analyst": ("bullish", 20),
        }
    )
    risk_data = {"AAPL": {"max_shares": 30}}
    portfolio = {"positions": {}}

    result = _majority_vote_fallback("AAPL", analyst_signals, risk_data, portfolio)
    assert result["action"] == "hold"


def test_majority_vote_ignores_risk_manager_and_pm_signals():
    analyst_signals = _signals(
        **{
            "Value Analyst": ("bullish", 80),
            "Risk Manager": ("bearish", 90),
            "Portfolio Manager": ("bearish", 90),
        }
    )
    risk_data = {"AAPL": {"max_shares": 30}}
    portfolio = {"positions": {}}

    result = _majority_vote_fallback("AAPL", analyst_signals, risk_data, portfolio)
    assert result["action"] == "buy"
