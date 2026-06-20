"""Fundamental Analyst Agent

Comprehensive fundamental analysis:
- Profitability: ROE, margins, earnings quality
- Financial health: debt ratios, liquidity, cash flow
- Valuation: P/E, P/B, P/S multiples
- Growth: revenue and earnings trajectory
"""

import json
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState
from src.tools.api import get_financial_metrics
from src.utils.display import show_agent_reasoning, progress_message

AGENT_NAME = "Fundamental Analyst"


def fundamental_agent(state: AgentState):
    """Broad fundamental analysis across multiple dimensions."""
    progress_message(AGENT_NAME, "running")
    tickers = state["tickers"]
    signals = {}

    for ticker in tickers:
        try:
            metrics = get_financial_metrics(ticker)

            profitability_score = 0
            health_score = 0
            valuation_score = 0
            growth_score = 0
            details = {}

            # --- Profitability ---
            roe = metrics.get("roe")
            if roe is not None:
                details["roe"] = round(roe, 4)
                if roe > 0.20:
                    profitability_score += 2
                elif roe > 0.10:
                    profitability_score += 1
                elif roe < 0:
                    profitability_score -= 2

            pm = metrics.get("profit_margin")
            if pm is not None:
                details["profit_margin"] = round(pm, 4)
                if pm > 0.20:
                    profitability_score += 2
                elif pm > 0.10:
                    profitability_score += 1
                elif pm < 0:
                    profitability_score -= 1

            om = metrics.get("operating_margin")
            if om is not None:
                details["operating_margin"] = round(om, 4)
                if om > 0.20:
                    profitability_score += 1
                elif om < 0:
                    profitability_score -= 1

            details["profitability_score"] = profitability_score

            # --- Financial health ---
            de = metrics.get("debt_to_equity")
            if de is not None:
                details["debt_to_equity"] = round(de, 2)
                if de < 30:
                    health_score += 2
                elif de < 80:
                    health_score += 1
                elif de > 150:
                    health_score -= 2

            cr = metrics.get("current_ratio")
            if cr is not None:
                details["current_ratio"] = round(cr, 2)
                if cr > 2.0:
                    health_score += 2
                elif cr > 1.5:
                    health_score += 1
                elif cr < 1.0:
                    health_score -= 2

            fcf = metrics.get("free_cash_flow")
            if fcf is not None:
                details["free_cash_flow"] = fcf
                if fcf > 0:
                    health_score += 1
                else:
                    health_score -= 1

            details["health_score"] = health_score

            # --- Valuation ---
            pe = metrics.get("pe_ratio")
            if pe is not None:
                details["pe_ratio"] = round(pe, 2)
                if 0 < pe < 15:
                    valuation_score += 2
                elif 0 < pe < 25:
                    valuation_score += 1
                elif pe > 40:
                    valuation_score -= 2

            pb = metrics.get("pb_ratio")
            if pb is not None:
                details["pb_ratio"] = round(pb, 2)
                if 0 < pb < 2:
                    valuation_score += 1
                elif pb > 8:
                    valuation_score -= 1

            ps = metrics.get("ps_ratio")
            if ps is not None:
                details["ps_ratio"] = round(ps, 2)
                if 0 < ps < 2:
                    valuation_score += 1
                elif ps > 10:
                    valuation_score -= 1

            details["valuation_score"] = valuation_score

            # --- Growth ---
            rg = metrics.get("revenue_growth")
            if rg is not None:
                details["revenue_growth"] = round(rg, 4)
                if rg > 0.20:
                    growth_score += 2
                elif rg > 0.05:
                    growth_score += 1
                elif rg < 0:
                    growth_score -= 1

            eg = metrics.get("earnings_growth")
            if eg is not None:
                details["earnings_growth"] = round(eg, 4)
                if eg > 0.20:
                    growth_score += 2
                elif eg > 0.05:
                    growth_score += 1
                elif eg < 0:
                    growth_score -= 1

            details["growth_score"] = growth_score

            # --- Aggregate ---
            total = profitability_score + health_score + valuation_score + growth_score
            max_total = 16
            confidence = round(min(abs(total) / max_total, 1.0) * 100)

            if total >= 5:
                signal = "bullish"
            elif total <= -3:
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
