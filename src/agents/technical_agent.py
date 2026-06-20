"""Technical Analyst Agent

Pure price-action and indicator-based analysis:
- Trend detection via moving averages (EMA 8/21/55)
- Momentum via RSI and MACD
- Mean reversion via Bollinger Bands
- Volume confirmation
"""

import json
import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState
from src.tools.api import get_prices
from src.utils.display import show_agent_reasoning, progress_message

AGENT_NAME = "Technical Analyst"


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def calc_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(series: pd.Series):
    ema12 = calc_ema(series, 12)
    ema26 = calc_ema(series, 26)
    macd_line = ema12 - ema26
    signal_line = calc_ema(macd_line, 9)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def technical_agent(state: AgentState):
    """Analyze tickers using technical indicators."""
    progress_message(AGENT_NAME, "running")
    tickers = state["tickers"]
    signals = {}

    for ticker in tickers:
        try:
            prices = get_prices(ticker, state["start_date"], state["end_date"])
            if prices.empty or len(prices) < 55:
                signals[ticker] = {
                    "agent": AGENT_NAME,
                    "signal": "neutral",
                    "confidence": 0,
                    "reasoning": {"error": "Insufficient price data"},
                }
                continue

            close = prices["Close"]
            volume = prices["Volume"]
            score = 0
            details = {}

            # --- Trend (EMA crossovers) ---
            ema8 = calc_ema(close, 8).iloc[-1]
            ema21 = calc_ema(close, 21).iloc[-1]
            ema55 = calc_ema(close, 55).iloc[-1]
            current = close.iloc[-1]

            details["ema_8"] = round(ema8, 2)
            details["ema_21"] = round(ema21, 2)
            details["ema_55"] = round(ema55, 2)

            if ema8 > ema21 > ema55:
                score += 2
                details["trend"] = "strong_uptrend"
            elif ema8 > ema21:
                score += 1
                details["trend"] = "uptrend"
            elif ema8 < ema21 < ema55:
                score -= 2
                details["trend"] = "strong_downtrend"
            elif ema8 < ema21:
                score -= 1
                details["trend"] = "downtrend"
            else:
                details["trend"] = "sideways"

            # --- RSI ---
            rsi = calc_rsi(close).iloc[-1]
            details["rsi"] = round(rsi, 2)
            if rsi < 30:
                score += 2
                details["rsi_signal"] = "oversold"
            elif rsi < 40:
                score += 1
                details["rsi_signal"] = "near_oversold"
            elif rsi > 70:
                score -= 2
                details["rsi_signal"] = "overbought"
            elif rsi > 60:
                score -= 1
                details["rsi_signal"] = "near_overbought"
            else:
                details["rsi_signal"] = "neutral"

            # --- MACD ---
            macd_line, macd_signal, macd_hist = calc_macd(close)
            hist_val = macd_hist.iloc[-1]
            hist_prev = macd_hist.iloc[-2]
            details["macd_histogram"] = round(hist_val, 4)

            if hist_val > 0 and hist_prev <= 0:
                score += 2
                details["macd_signal"] = "bullish_cross"
            elif hist_val > 0:
                score += 1
                details["macd_signal"] = "bullish"
            elif hist_val < 0 and hist_prev >= 0:
                score -= 2
                details["macd_signal"] = "bearish_cross"
            elif hist_val < 0:
                score -= 1
                details["macd_signal"] = "bearish"

            # --- Bollinger Bands ---
            bb_upper, bb_mid, bb_lower = calc_bollinger(close)
            bb_u = bb_upper.iloc[-1]
            bb_l = bb_lower.iloc[-1]
            details["bb_position"] = round((current - bb_l) / (bb_u - bb_l) if (bb_u - bb_l) > 0 else 0.5, 2)

            if current < bb_l:
                score += 1
                details["bb_signal"] = "below_lower"
            elif current > bb_u:
                score -= 1
                details["bb_signal"] = "above_upper"

            # --- Volume confirmation ---
            vol_avg = volume.rolling(20).mean().iloc[-1]
            vol_current = volume.iloc[-1]
            if vol_avg > 0:
                vol_ratio = vol_current / vol_avg
                details["volume_ratio"] = round(vol_ratio, 2)
                if vol_ratio > 1.5 and score > 0:
                    score += 1
                elif vol_ratio > 1.5 and score < 0:
                    score -= 1

            # --- Generate signal ---
            max_score = 8
            confidence = round(min(abs(score) / max_score, 1.0) * 100)

            if score >= 3:
                signal = "bullish"
            elif score <= -3:
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
