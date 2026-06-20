import sqlite3
import pytest
from src import memory


@pytest.fixture
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_memory.db")
    monkeypatch.setattr(memory, "DB_PATH", db_path)
    memory.init_db()
    return db_path


def test_init_db_creates_tables(db):
    conn = sqlite3.connect(db)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"decisions", "runs", "agent_scores"} <= tables


def test_save_and_get_past_decisions(db):
    memory.save_decision(
        run_id="run1", ticker="AAPL", agent="Value Analyst",
        signal="bullish", confidence=80, reasoning="cheap valuation",
        action="buy", quantity=10, price=150.0,
    )
    decisions = memory.get_past_decisions("AAPL")
    assert len(decisions) == 1
    assert decisions[0]["agent"] == "Value Analyst"
    assert decisions[0]["signal"] == "bullish"


def test_get_agent_accuracy_no_data_returns_none(db):
    result = memory.get_agent_accuracy("Growth Analyst")
    assert result == {"agent": "Growth Analyst", "total_signals": 0, "accuracy": None}


def test_get_agent_accuracy_computes_percentage(db):
    memory.record_agent_score("Growth Analyst", "AAPL", "bullish", True)
    memory.record_agent_score("Growth Analyst", "AAPL", "bearish", False)
    memory.record_agent_score("Growth Analyst", "MSFT", "bullish", True)

    result = memory.get_agent_accuracy("Growth Analyst")
    assert result["total_signals"] == 3
    assert result["correct_signals"] == 2
    assert result["accuracy"] == pytest.approx(66.7, abs=0.1)


def test_save_run_and_update_result(db):
    memory.save_run("run1", ["AAPL"], "groq", 100000)
    memory.update_run_result("run1", 110000, 10.0)
    runs = memory.get_recent_runs()
    assert runs[0]["id"] == "run1"
    assert runs[0]["final_value"] == 110000
    assert runs[0]["return_pct"] == 10.0
