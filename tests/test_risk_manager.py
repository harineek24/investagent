import importlib
risk_manager_module = importlib.import_module("src.agents.risk_manager")
risk_manager = risk_manager_module.risk_manager


def _state(tickers, portfolio, prices, monkeypatch):
    monkeypatch.setattr(
        risk_manager_module, "get_current_price", lambda ticker: prices.get(ticker)
    )
    return {"tickers": tickers, "portfolio": portfolio, "show_reasoning": False}


def test_caps_position_at_20_percent_of_portfolio(monkeypatch):
    state = _state(
        ["AAPL"],
        {"cash": 100_000, "positions": {"AAPL": {"shares": 0, "avg_cost": 0}}},
        {"AAPL": 100.0},
        monkeypatch,
    )
    result = risk_manager(state)
    signal = result["analyst_signals"]["Risk Manager"]["AAPL"]
    assert signal["remaining_limit_usd"] == 20_000.0
    assert signal["max_shares"] == 200


def test_existing_position_reduces_remaining_limit(monkeypatch):
    state = _state(
        ["AAPL"],
        {"cash": 100_000, "positions": {"AAPL": {"shares": 100, "avg_cost": 90}}},
        {"AAPL": 100.0},
        monkeypatch,
    )
    result = risk_manager(state)
    signal = result["analyst_signals"]["Risk Manager"]["AAPL"]
    # total_value = 100k cash + 100*100 = 110k; max per ticker = 22k; already holding 10k
    assert signal["remaining_limit_usd"] == 12_000.0


def test_cannot_exceed_available_cash(monkeypatch):
    state = _state(
        ["AAPL"],
        {"cash": 5_000, "positions": {"AAPL": {"shares": 0, "avg_cost": 0}}},
        {"AAPL": 100.0},
        monkeypatch,
    )
    result = risk_manager(state)
    signal = result["analyst_signals"]["Risk Manager"]["AAPL"]
    # 20% of 5k = 1k, well under cash, so limit stays at 1k
    assert signal["remaining_limit_usd"] == 1_000.0


def test_missing_price_returns_zero_limit(monkeypatch):
    state = _state(
        ["AAPL"],
        {"cash": 100_000, "positions": {"AAPL": {"shares": 0, "avg_cost": 0}}},
        {},
        monkeypatch,
    )
    result = risk_manager(state)
    signal = result["analyst_signals"]["Risk Manager"]["AAPL"]
    assert signal["remaining_limit_usd"] == 0
    assert signal["reasoning"] == "Could not fetch price"
