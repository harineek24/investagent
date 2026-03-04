"""Contrarian Analyst Agent

Analyzes stocks from a contrarian/deep-value perspective:
- Looks for oversold stocks with solid fundamentals
- Seeks disconnects between price and underlying value
- Favors stocks trading near 52-week lows with improving metrics
- Inspired by Burry/Ackman contrarian strategies
"""

import json
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState
from src.tools.api import get_financial_metrics, get_prices
from src.utils.display import show_agent_reasoning, progress_message

AGENT_NAME = "Contrarian Analyst"


def contrarian_agent(state: AgentState):
    """Find contrarian opportunities where the market may be wrong."""
    progress_message(AGENT_NAME, "running")
    tickers = state["tickers"]
    signals = {}

    for ticker in tickers:
        try:
            metrics = get_financial_metrics(ticker)
            prices = get_prices(ticker, state["start_date"], state["end_date"])

            score = 0
            details = {}

            # --- Price vs 52-week range ---
            high52 = metrics.get("52_week_high")
            low52 = metrics.get("52_week_low")
            if high52 and low52 and not prices.empty:
                current = prices["Close"].iloc[-1]
                range_pct = (current - low52) / (high52 - low52) if (high52 - low52) > 0 else 0.5
                details["52w_range_position"] = round(range_pct, 2)
                if range_pct < 0.3:
                    score += 2  # near bottom - contrarian bullish
                elif range_pct < 0.5:
                    score += 1
                elif range_pct > 0.9:
                    score -= 2  # near top - contrarian bearish

            # --- Price vs moving averages ---
            avg200 = metrics.get("200_day_avg")
            if avg200 and not prices.empty:
                current = prices["Close"].iloc[-1]
                discount = (current - avg200) / avg200
                details["discount_to_200d"] = round(discount, 4)
                if discount < -0.20:
                    score += 2
                elif discount < -0.10:
                    score += 1
                elif discount > 0.20:
                    score -= 1

            # --- Solid fundamentals despite depressed price ---
            fcf = metrics.get("free_cash_flow")
            if fcf is not None and fcf > 0:
                details["positive_fcf"] = True
                score += 1

            de = metrics.get("debt_to_equity")
            if de is not None:
                details["debt_to_equity"] = round(de, 2)
                if de < 80:
                    score += 1

            pe = metrics.get("pe_ratio")
            if pe is not None:
                details["pe_ratio"] = round(pe, 2)
                if 0 < pe < 12:
                    score += 2
                elif 0 < pe < 18:
                    score += 1

            # --- Insider buying (contrarian signal) ---
            roe = metrics.get("roe")
            if roe and roe > 0.10:
                details["roe"] = round(roe, 4)
                score += 1

            # --- Volatility (contrarians seek high vol = opportunity) ---
            if not prices.empty and len(prices) > 20:
                returns = prices["Close"].pct_change().dropna()
                vol = returns.std() * (252 ** 0.5)
                details["annualized_vol"] = round(vol, 4)
                if vol > 0.4:
                    score += 1  # high vol = opportunity

            # --- Generate signal ---
            max_score = 10
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
