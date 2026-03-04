"""Risk Manager Agent

Controls position sizing and risk limits:
- Maximum 20% of portfolio per position
- Accounts for existing holdings
- Ensures cash constraints are respected
- Calculates per-ticker position limits
"""

import json
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState
from src.tools.api import get_current_price
from src.utils.display import show_agent_reasoning, progress_message

AGENT_NAME = "Risk Manager"


def risk_manager(state: AgentState):
    """Calculate position limits based on risk constraints."""
    progress_message(AGENT_NAME, "running")
    tickers = state["tickers"]
    portfolio = state["portfolio"]
    signals = {}

    cash = portfolio.get("cash", 100000)
    positions = portfolio.get("positions", {})

    # Calculate total portfolio value
    total_value = cash
    position_values = {}
    current_prices = {}

    for ticker in tickers:
        price = get_current_price(ticker)
        if price is None:
            continue
        current_prices[ticker] = price
        qty = positions.get(ticker, {}).get("shares", 0)
        pos_value = abs(qty) * price
        position_values[ticker] = pos_value
        total_value += pos_value

    for ticker in tickers:
        price = current_prices.get(ticker)
        if price is None:
            signals[ticker] = {
                "agent": AGENT_NAME,
                "remaining_limit_usd": 0,
                "current_price": 0,
                "reasoning": "Could not fetch price",
            }
            continue

        max_position_pct = 0.20  # 20% max per ticker
        max_position_usd = total_value * max_position_pct

        current_pos_value = position_values.get(ticker, 0)
        remaining_limit = max(0, max_position_usd - current_pos_value)
        remaining_limit = min(remaining_limit, cash)  # can't exceed cash

        signals[ticker] = {
            "agent": AGENT_NAME,
            "remaining_limit_usd": round(remaining_limit, 2),
            "max_shares": int(remaining_limit / price) if price > 0 else 0,
            "current_price": round(price, 2),
            "portfolio_value": round(total_value, 2),
            "current_position_value": round(current_pos_value, 2),
            "reasoning": {
                "portfolio_value": round(total_value, 2),
                "max_per_ticker": round(max_position_usd, 2),
                "existing_position": round(current_pos_value, 2),
                "available_cash": round(cash, 2),
                "remaining_limit": round(remaining_limit, 2),
            },
        }

    if state.get("show_reasoning"):
        show_agent_reasoning(signals, AGENT_NAME)

    progress_message(AGENT_NAME, "done")

    return {
        "messages": [HumanMessage(content=json.dumps(signals), name=AGENT_NAME)],
        "analyst_signals": {AGENT_NAME: signals},
    }
