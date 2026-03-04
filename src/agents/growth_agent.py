"""Growth Analyst Agent

Analyzes stocks through a growth investing lens:
- Prioritizes revenue and earnings growth rates
- Looks for expanding margins and market opportunity
- Favors innovation and disruption potential
- Inspired by Lynch/Wood growth-seeking approach
"""

import json
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState
from src.tools.api import get_financial_metrics, get_prices
from src.utils.display import show_agent_reasoning, progress_message

AGENT_NAME = "Growth Analyst"


def growth_agent(state: AgentState):
    """Analyze tickers for growth potential."""
    progress_message(AGENT_NAME, "running")
    tickers = state["tickers"]
    signals = {}

    for ticker in tickers:
        try:
            metrics = get_financial_metrics(ticker)
            prices = get_prices(ticker, state["start_date"], state["end_date"])

            score = 0
            details = {}

            # --- Revenue growth ---
            rev_growth = metrics.get("revenue_growth")
            if rev_growth is not None:
                details["revenue_growth"] = round(rev_growth, 4)
                if rev_growth > 0.25:
                    score += 3
                elif rev_growth > 0.15:
                    score += 2
                elif rev_growth > 0.05:
                    score += 1
                elif rev_growth < 0:
                    score -= 2

            # --- Earnings growth ---
            earn_growth = metrics.get("earnings_growth")
            if earn_growth is not None:
                details["earnings_growth"] = round(earn_growth, 4)
                if earn_growth > 0.25:
                    score += 3
                elif earn_growth > 0.10:
                    score += 2
                elif earn_growth > 0:
                    score += 1
                elif earn_growth < 0:
                    score -= 2

            # --- Forward PE (growth at reasonable price) ---
            fpe = metrics.get("forward_pe")
            pe = metrics.get("pe_ratio")
            if fpe and pe and pe > 0:
                peg_approx = pe / max(((rev_growth or 0) * 100), 1) if rev_growth and rev_growth > 0 else None
                if peg_approx is not None:
                    details["approx_peg"] = round(peg_approx, 2)
                    if peg_approx < 1.0:
                        score += 2
                    elif peg_approx < 2.0:
                        score += 1
                    elif peg_approx > 3.0:
                        score -= 1

            # --- Margin expansion ---
            op_margin = metrics.get("operating_margin")
            if op_margin is not None:
                details["operating_margin"] = round(op_margin, 4)
                if op_margin > 0.25:
                    score += 2
                elif op_margin > 0.15:
                    score += 1
                elif op_margin < 0:
                    score -= 1

            # --- Price momentum (growth stocks tend to have momentum) ---
            if not prices.empty and len(prices) > 20:
                recent_return = (prices["Close"].iloc[-1] / prices["Close"].iloc[-20] - 1)
                details["20d_return"] = round(recent_return, 4)
                if recent_return > 0.10:
                    score += 1
                elif recent_return < -0.15:
                    score -= 1

            # --- Generate signal ---
            max_score = 11
            confidence = round(min(abs(score) / max_score, 1.0) * 100)

            if score >= 4:
                signal = "bullish"
            elif score <= -2:
                signal = "bearish"
            else:
                signal = "neutral"

            signals[ticker] = {
                "agent": AGENT_NAME,
                "signal": signal,
                "confidence": confidence,
                "reasoning": details,
            }

        except Exception as e:
            signals[ticker] = {
                "agent": AGENT_NAME,
                "signal": "neutral",
                "confidence": 0,
                "reasoning": {"error": str(e)},
            }

    if state.get("show_reasoning"):
        show_agent_reasoning(signals, AGENT_NAME)

    progress_message(AGENT_NAME, "done")

    return {
        "messages": [HumanMessage(content=json.dumps(signals), name=AGENT_NAME)],
        "analyst_signals": {AGENT_NAME: signals},
    }
