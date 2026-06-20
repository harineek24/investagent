import pytest
from src import memory, scoring


@pytest.fixture
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_scoring.db")
    monkeypatch.setattr(memory, "DB_PATH", db_path)
    memory.init_db()
    return db_path


def _save_old_decision(db, ticker, agent, signal, price, days_ago=31):
    import sqlite3
    from datetime import datetime, timedelta

    memory.save_decision(
        run_id="run1", ticker=ticker, agent=agent, signal=signal,
        confidence=70, reasoning="test", action="hold", quantity=0, price=price,
    )
    old_ts = (datetime.now() - timedelta(days=days_ago)).isoformat()
    conn = sqlite3.connect(db)
    conn.execute("UPDATE decisions SET timestamp = ? WHERE ticker = ? AND agent = ?", (old_ts, ticker, agent))
    conn.commit()
    conn.close()


def test_bullish_signal_correct_when_price_rose(db, monkeypatch):
    _save_old_decision(db, "AAPL", "Value Analyst", "bullish", price=100.0)
    monkeypatch.setattr(scoring, "get_current_price", lambda ticker: 110.0)

    scored = scoring.score_pending_decisions()
    assert scored == 1

    accuracy = memory.get_agent_accuracy("Value Analyst")
    assert accuracy["total_signals"] == 1
    assert accuracy["correct_signals"] == 1


def test_bullish_signal_incorrect_when_price_fell(db, monkeypatch):
    _save_old_decision(db, "AAPL", "Value Analyst", "bullish", price=100.0)
    monkeypatch.setattr(scoring, "get_current_price", lambda ticker: 90.0)

    scoring.score_pending_decisions()
    accuracy = memory.get_agent_accuracy("Value Analyst")
    assert accuracy["correct_signals"] == 0


def test_neutral_signal_correct_within_band(db, monkeypatch):
    _save_old_decision(db, "AAPL", "Technical Analyst", "neutral", price=100.0)
    monkeypatch.setattr(scoring, "get_current_price", lambda ticker: 101.0)

    scoring.score_pending_decisions()
    accuracy = memory.get_agent_accuracy("Technical Analyst")
    assert accuracy["correct_signals"] == 1


def test_recent_decision_not_scored_yet(db, monkeypatch):
    _save_old_decision(db, "AAPL", "Value Analyst", "bullish", price=100.0, days_ago=5)
    monkeypatch.setattr(scoring, "get_current_price", lambda ticker: 200.0)

    scored = scoring.score_pending_decisions()
    assert scored == 0
    accuracy = memory.get_agent_accuracy("Value Analyst")
    assert accuracy["total_signals"] == 0


def test_decision_only_scored_once(db, monkeypatch):
    _save_old_decision(db, "AAPL", "Value Analyst", "bullish", price=100.0)
    monkeypatch.setattr(scoring, "get_current_price", lambda ticker: 110.0)

    scoring.score_pending_decisions()
    scored_again = scoring.score_pending_decisions()
    assert scored_again == 0
