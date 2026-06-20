"""Value Analyst Agent

Analyzes stocks through a classic value investing lens:
- Seeks margin of safety between price and intrinsic value
- Favors strong fundamentals: low debt, high ROE, consistent earnings
- Applies DCF and owner-earnings valuation models
- Conservative approach inspired by Buffett/Graham principles
"""

import json
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState
from src.tools.api import get_financial_metrics, get_prices, get_income_statement, get_cash_flow
from src.utils.display import show_agent_reasoning, progress_message

AGENT_NAME = "Value Analyst"


def value_agent(state: AgentState):
    """Analyze tickers from a value investing perspective."""
    progress_message(AGENT_NAME, "running")
    tickers = state["tickers"]
    end_date = state["end_date"]
    signals = {}

    for ticker in tickers:
        try:
            metrics = get_financial_metrics(ticker)
            prices = get_prices(ticker, state["start_date"], end_date)

            score = 0
            details = {}

            # --- Profitability ---
            roe = metrics.get("roe")
            if roe is not None:
                details["roe"] = round(roe, 4)
                if roe > 0.15:
                    score += 2
                elif roe > 0.10:
                    score += 1
                elif roe < 0.05:
                    score -= 1

            margin = metrics.get("profit_margin")
            if margin is not None:
                details["profit_margin"] = round(margin, 4)
                if margin > 0.20:
                    score += 2
                elif margin > 0.10:
                    score += 1

            # --- Financial health ---
            de = metrics.get("debt_to_equity")
            if de is not None:
                details["debt_to_equity"] = round(de, 2)
                if de < 50:
                    score += 2
                elif de < 100:
                    score += 1
                else:
                    score -= 1

            cr = metrics.get("current_ratio")
            if cr is not None:
                details["current_ratio"] = round(cr, 2)
                if cr > 1.5:
                    score += 1
                elif cr < 1.0:
                    score -= 1

            # --- Valuation ---
            pe = metrics.get("pe_ratio")
            if pe is not None:
                details["pe_ratio"] = round(pe, 2)
                if 0 < pe < 15:
                    score += 2
                elif 0 < pe < 20:
                    score += 1
                elif pe > 30:
                    score -= 1

            pb = metrics.get("pb_ratio")
            if pb is not None:
                details["pb_ratio"] = round(pb, 2)
                if 0 < pb < 1.5:
                    score += 2
                elif 0 < pb < 3:
                    score += 1
                elif pb > 5:
                    score -= 1

            # --- Owner earnings estimate ---
            fcf = metrics.get("free_cash_flow")
            mcap = metrics.get("market_cap")
            if fcf and mcap and mcap > 0:
                fcf_yield = fcf / mcap
                details["fcf_yield"] = round(fcf_yield, 4)
                if fcf_yield > 0.08:
                    score += 2
                elif fcf_yield > 0.05:
                    score += 1

            # --- Generate signal ---
            max_score = 12
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
