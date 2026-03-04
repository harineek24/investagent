"""Agent Memory - SQLite-based persistent memory for agents.

Stores past decisions, outcomes, and agent performance so agents can
learn from their track record across sessions.

Uses SQLite (file-based, no server needed, free, deploys anywhere).
"""

import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.environ.get("INVESTAGENT_DB", "investagent_memory.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            agent TEXT NOT NULL,
            signal TEXT,
            confidence REAL,
            reasoning TEXT,
            action TEXT,
            quantity INTEGER DEFAULT 0,
            price_at_decision REAL,
            price_after_30d REAL,
            outcome TEXT
        );

        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            tickers TEXT NOT NULL,
            provider TEXT,
            initial_cash REAL,
            final_value REAL,
            return_pct REAL
        );

        CREATE TABLE IF NOT EXISTS agent_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent TEXT NOT NULL,
            ticker TEXT NOT NULL,
            signal TEXT,
            was_correct INTEGER,
            timestamp TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_decisions_ticker ON decisions(ticker);
        CREATE INDEX IF NOT EXISTS idx_decisions_agent ON decisions(agent);
        CREATE INDEX IF NOT EXISTS idx_agent_scores_agent ON agent_scores(agent);
    """)
    conn.close()


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def save_run(run_id: str, tickers: list[str], provider: str, initial_cash: float):
    """Save a new run."""
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO runs (id, timestamp, tickers, provider, initial_cash) VALUES (?, ?, ?, ?, ?)",
        (run_id, datetime.now().isoformat(), json.dumps(tickers), provider, initial_cash),
    )
    conn.commit()
    conn.close()


def save_decision(
    run_id: str,
    ticker: str,
    agent: str,
    signal: str,
    confidence: float,
    reasoning: dict | str,
    action: str = "",
    quantity: int = 0,
    price: float = 0,
):
    """Save an agent's decision for a ticker."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO decisions
        (run_id, timestamp, ticker, agent, signal, confidence, reasoning, action, quantity, price_at_decision)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            datetime.now().isoformat(),
            ticker,
            agent,
            signal,
            confidence,
            json.dumps(reasoning) if isinstance(reasoning, dict) else str(reasoning),
            action,
            quantity,
            price,
        ),
    )
    conn.commit()
    conn.close()


def update_run_result(run_id: str, final_value: float, return_pct: float):
    """Update run with final results."""
    conn = _get_conn()
    conn.execute(
        "UPDATE runs SET final_value = ?, return_pct = ? WHERE id = ?",
        (final_value, return_pct, run_id),
    )
    conn.commit()
    conn.close()


def record_agent_score(agent: str, ticker: str, signal: str, was_correct: bool):
    """Record whether an agent's signal was correct."""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO agent_scores (agent, ticker, signal, was_correct, timestamp) VALUES (?, ?, ?, ?, ?)",
        (agent, ticker, signal, int(was_correct), datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Read operations (used by agents to learn from history)
# ---------------------------------------------------------------------------

def get_past_decisions(ticker: str, limit: int = 20) -> list[dict]:
    """Get recent past decisions for a ticker across all agents."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT agent, signal, confidence, action, quantity, price_at_decision, outcome, timestamp
        FROM decisions WHERE ticker = ? ORDER BY timestamp DESC LIMIT ?""",
        (ticker, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_agent_accuracy(agent: str) -> dict:
    """Get an agent's historical accuracy."""
    conn = _get_conn()
    total = conn.execute(
        "SELECT COUNT(*) as cnt FROM agent_scores WHERE agent = ?", (agent,)
    ).fetchone()["cnt"]

    if total == 0:
        conn.close()
        return {"agent": agent, "total_signals": 0, "accuracy": None}

    correct = conn.execute(
        "SELECT COUNT(*) as cnt FROM agent_scores WHERE agent = ? AND was_correct = 1", (agent,)
    ).fetchone()["cnt"]

    conn.close()
    return {
        "agent": agent,
        "total_signals": total,
        "correct_signals": correct,
        "accuracy": round(correct / total * 100, 1) if total > 0 else 0,
    }


def get_all_agent_accuracies() -> list[dict]:
    """Get accuracy for all agents."""
    conn = _get_conn()
    agents = conn.execute("SELECT DISTINCT agent FROM agent_scores").fetchall()
    conn.close()
    return [get_agent_accuracy(row["agent"]) for row in agents]


def get_recent_runs(limit: int = 10) -> list[dict]:
    """Get recent runs with results."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_ticker_history(ticker: str) -> list[dict]:
    """Get full decision history for a ticker."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT d.*, r.return_pct as run_return
        FROM decisions d LEFT JOIN runs r ON d.run_id = r.id
        WHERE d.ticker = ? ORDER BY d.timestamp DESC""",
        (ticker,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Initialize on import
init_db()
