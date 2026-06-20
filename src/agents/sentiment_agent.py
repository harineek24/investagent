"""Sentiment Analyst Agent

Analyzes market sentiment signals:
- News sentiment from recent articles
- Insider trading patterns (buying vs selling)
- Analyst recommendations
- Combines signals with weighted scoring
"""

import json
import pandas as pd
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState
from src.tools.api import get_company_news, get_insider_trades, get_recommendations
from src.utils.display import show_agent_reasoning, progress_message
from src.utils.llm import get_llm

AGENT_NAME = "Sentiment Analyst"


def sentiment_agent(state: AgentState):
    """Analyze sentiment signals for each ticker."""
    progress_message(AGENT_NAME, "running")
    tickers = state["tickers"]
    signals = {}

    for ticker in tickers:
        try:
            score = 0
            details = {}

            # --- News sentiment ---
            news = get_company_news(ticker)
            if news:
                headlines = [n["title"] for n in news if n.get("title")]
                details["num_articles"] = len(headlines)
                details["recent_headlines"] = headlines[:5]

                # Simple keyword-based sentiment (no LLM cost)
                positive_words = {"beat", "surge", "gain", "profit", "growth", "upgrade",
                                  "buy", "bull", "record", "high", "strong", "soar", "rally"}
                negative_words = {"miss", "drop", "loss", "decline", "downgrade", "sell",
                                  "bear", "low", "weak", "crash", "fall", "cut", "warn"}

                pos_count = 0
                neg_count = 0
                for h in headlines:
                    h_lower = h.lower()
                    pos_count += sum(1 for w in positive_words if w in h_lower)
                    neg_count += sum(1 for w in negative_words if w in h_lower)

                news_sentiment = pos_count - neg_count
                details["news_pos_signals"] = pos_count
                details["news_neg_signals"] = neg_count

                if news_sentiment > 2:
                    score += 2
                elif news_sentiment > 0:
                    score += 1
                elif news_sentiment < -2:
                    score -= 2
                elif news_sentiment < 0:
                    score -= 1

            # --- Insider trades ---
            insiders = get_insider_trades(ticker)
            if isinstance(insiders, pd.DataFrame) and not insiders.empty:
                details["insider_trades_found"] = len(insiders)
                # Look at recent transactions
                buy_cols = [c for c in insiders.columns if "buy" in c.lower() or "purchase" in c.lower()]
                sell_cols = [c for c in insiders.columns if "sale" in c.lower() or "sell" in c.lower()]

                # Simple heuristic: more insider buying = bullish
                if "Text" in insiders.columns:
                    texts = insiders["Text"].astype(str).str.lower()
                    buys = texts.str.contains("purchase|buy|acquisition", na=False).sum()
                    sells = texts.str.contains("sale|sell|dispos", na=False).sum()
                    details["insider_buys"] = int(buys)
                    details["insider_sells"] = int(sells)
                    if buys > sells:
                        score += 1
                    elif sells > buys * 2:
                        score -= 1

            # --- Analyst recommendations ---
            recs = get_recommendations(ticker)
            if isinstance(recs, pd.DataFrame) and not recs.empty:
                recent = recs.tail(5)
                details["recent_recommendations"] = len(recent)
                if "To Grade" in recent.columns:
                    grades = recent["To Grade"].astype(str).str.lower()
                    buys = grades.str.contains("buy|overweight|outperform", na=False).sum()
                    sells = grades.str.contains("sell|underweight|underperform", na=False).sum()
                    details["analyst_buys"] = int(buys)
                    details["analyst_sells"] = int(sells)
                    if buys > sells:
                        score += 1
                    elif sells > buys:
                        score -= 1

            # --- Generate signal ---
            max_score = 4
            confidence = round(min(abs(score) / max_score, 1.0) * 100)

            if score >= 2:
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
