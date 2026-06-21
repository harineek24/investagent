"""ReAct Agent Factory — turns analysts into tool-using, reasoning agents.

Instead of hardcoded scoring:
  Before: fetch data → if metric > threshold → score++
  After:  LLM decides what data to fetch → reasons about it → may fetch more → concludes

This is the core of what makes the system agentic:
  1. Agents choose which tools to call (autonomy)
  2. Agents reason about results before calling more tools (planning)
  3. Agents loop until they have enough info (self-directed)
  4. Max iterations prevent runaway costs on free tiers
"""

import json
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState
from src.utils.llm import get_llm
from src.utils.display import show_agent_reasoning, progress_message, console

# Max tool-call iterations per agent per ticker (keeps Groq free tier safe)
MAX_ITERATIONS = 4


# ---------------------------------------------------------------------------
# Tool registry: functions agents can call autonomously
# ---------------------------------------------------------------------------

_TOOL_DESCRIPTIONS = {
    "get_prices": (
        "get_prices(ticker, start_date, end_date) -> OHLCV DataFrame summary",
        "price trends, moving averages, volume analysis, momentum",
    ),
    "get_financial_metrics": (
        "get_financial_metrics(ticker) -> dict with 28 financial fields",
        "P/E, P/B, ROE, margins, growth rates, debt ratios",
    ),
    "get_company_news": (
        "get_company_news(ticker) -> list of recent news articles",
        "sentiment, catalysts, market-moving events",
    ),
    "get_insider_trades": (
        "get_insider_trades(ticker) -> insider buy/sell transactions",
        "insider confidence signals",
    ),
    "get_recommendations": (
        "get_recommendations(ticker) -> analyst upgrade/downgrade history",
        "Wall Street consensus",
    ),
    "get_sec_financial_facts": (
        "get_sec_financial_facts(ticker) -> structured XBRL data from SEC filings",
        "official revenue, net income, assets, debt (multi-year)",
    ),
    "get_sec_recent_filings": (
        "get_sec_recent_filings(ticker, filing_type) -> list of recent SEC filings",
        "checking when last 10-K/10-Q was filed",
    ),
}

# Tools most relevant to each agent's philosophy, in priority order.
# Agents can still call any tool — this just steers them toward the
# ones that matter for their lens first, so the 4-call budget isn't
# spent on irrelevant data.
AGENT_RECOMMENDED_TOOLS = {
    "Value Analyst": ["get_financial_metrics", "get_sec_financial_facts", "get_prices"],
    "Growth Analyst": ["get_financial_metrics", "get_sec_financial_facts", "get_recommendations"],
    "Contrarian Analyst": ["get_prices", "get_company_news", "get_insider_trades"],
    "Technical Analyst": ["get_prices"],
    "Fundamental Analyst": ["get_financial_metrics", "get_sec_financial_facts", "get_sec_recent_filings"],
    "Sentiment Analyst": ["get_company_news", "get_insider_trades", "get_recommendations"],
}


def _build_tool_descriptions(agent_name: str | None = None) -> str:
    """Build a description of available tools for the agent prompt.

    Recommended tools (matching the agent's philosophy) are listed first
    and flagged, so the agent spends its limited tool-call budget on
    the data that's actually relevant to its lens.
    """
    recommended = AGENT_RECOMMENDED_TOOLS.get(agent_name, [])
    ordered = recommended + [name for name in _TOOL_DESCRIPTIONS if name not in recommended]

    lines = ["Available tools (call by name with JSON args):", ""]
    for i, name in enumerate(ordered, 1):
        signature, use_for = _TOOL_DESCRIPTIONS[name]
        tag = " [RECOMMENDED for your philosophy]" if name in recommended else ""
        lines.append(f"{i}. {signature}{tag}")
        lines.append(f"   Use for: {use_for}")
        lines.append("")

    return "\n".join(lines).rstrip()


def _execute_tool(tool_name: str, args: dict) -> str:
    """Execute a tool call and return stringified result."""
    from src.tools.api import (
        get_prices, get_financial_metrics, get_company_news,
        get_insider_trades, get_recommendations,
    )
    from src.tools.sec_edgar import get_company_facts, get_recent_filings

    tool_map = {
        "get_prices": lambda a: _summarize_prices(
            get_prices(a["ticker"], a.get("start_date", ""), a.get("end_date", ""))
        ),
        "get_financial_metrics": lambda a: json.dumps(
            get_financial_metrics(a["ticker"]), default=str
        ),
        "get_company_news": lambda a: json.dumps(
            get_company_news(a["ticker"]), default=str
        ),
        "get_insider_trades": lambda a: _summarize_df(
            get_insider_trades(a["ticker"])
        ),
        "get_recommendations": lambda a: _summarize_df(
            get_recommendations(a["ticker"])
        ),
        "get_sec_financial_facts": lambda a: json.dumps(
            get_company_facts(a["ticker"]), default=str
        ),
        "get_sec_recent_filings": lambda a: json.dumps(
            get_recent_filings(a["ticker"], a.get("filing_type", "10-K")), default=str
        ),
    }

    func = tool_map.get(tool_name)
    if not func:
        return f"Unknown tool: {tool_name}"
    try:
        return func(args)
    except Exception as e:
        return f"Tool error: {e}"


def _summarize_prices(df) -> str:
    """Summarize a price DataFrame for the LLM (avoid dumping 90 rows)."""
    if df is None or df.empty:
        return "No price data available."
    close = df["Close"]
    volume = df["Volume"]
    return json.dumps({
        "data_points": len(df),
        "latest_close": round(float(close.iloc[-1]), 2),
        "period_start": round(float(close.iloc[0]), 2),
        "period_return_pct": round(float((close.iloc[-1] / close.iloc[0] - 1) * 100), 2),
        "period_high": round(float(close.max()), 2),
        "period_low": round(float(close.min()), 2),
        "avg_volume": int(volume.mean()),
        "ema_8": round(float(close.ewm(span=8).mean().iloc[-1]), 2),
        "ema_21": round(float(close.ewm(span=21).mean().iloc[-1]), 2),
        "ema_55": round(float(close.ewm(span=55).mean().iloc[-1]), 2) if len(close) >= 55 else None,
        "rsi_14": round(float(_calc_rsi(close)), 2),
        "volatility_pct": round(float(close.pct_change().std() * 100), 4),
        "last_5_closes": [round(float(x), 2) for x in close.tail(5).tolist()],
    })


def _calc_rsi(series, period=14) -> float:
    """Quick RSI calculation."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100
    return 100 - (100 / (1 + rs))


def _summarize_df(df) -> str:
    """Summarize a DataFrame for the LLM."""
    if df is None or df.empty:
        return "No data available."
    return df.head(10).to_string()


# ---------------------------------------------------------------------------
# The ReAct loop
# ---------------------------------------------------------------------------

REACT_SYSTEM_PROMPT = """You are {agent_name}, an expert stock analyst.

Your job: analyze {ticker} and produce a trading signal (bullish/neutral/bearish).

## Your Investment Philosophy
{philosophy}

## How You Work (ReAct Pattern)
You reason step-by-step, calling tools when you need data:

1. THINK: What do I need to know? What's my hypothesis?
2. ACT: Call a tool to get data
3. OBSERVE: Look at the result
4. REPEAT until you have enough information (max {max_iter} tool calls)
5. CONCLUDE: Give your final signal

## Tool Calling Format
To call a tool, respond with EXACTLY this format:
TOOL_CALL: tool_name({{"key": "value"}})

Example:
TOOL_CALL: get_financial_metrics({{"ticker": "AAPL"}})

## Final Answer Format
When you're ready to conclude, respond with EXACTLY:
FINAL_ANSWER: {{"signal": "bullish|neutral|bearish", "confidence": <0-100>, "reasoning": "<your analysis>"}}

## Rules
- Prioritize the tools marked [RECOMMENDED for your philosophy] below — they match your lens
- Call at least 2 different tools before concluding, preferring recommended ones first
- Don't call the same tool twice with the same args
- Think out loud — explain WHY you're calling each tool
- Your confidence should reflect the strength of evidence
- Be honest about uncertainty

{tools}

## Analysis Period
Start: {start_date}
End: {end_date}

Begin your analysis of {ticker} now."""


AGENT_PHILOSOPHIES = {
    "Value Analyst": (
        "You follow Buffett/Graham principles: margin of safety, intrinsic value, "
        "strong fundamentals at a reasonable price. You look for low P/E, low P/B, "
        "high ROE, strong free cash flow, and low debt. You're skeptical of hype."
    ),
    "Growth Analyst": (
        "You follow Lynch/Wood principles: revenue growth, earnings acceleration, "
        "expanding margins, market opportunity. You care about PEG ratio, revenue "
        "growth >20%, and operating leverage. You accept higher valuations for faster growth."
    ),
    "Contrarian Analyst": (
        "You look for oversold opportunities others are missing. Stocks near 52-week "
        "lows with solid fundamentals, high short interest, negative sentiment that's "
        "overdone. You buy fear and sell greed."
    ),
    "Technical Analyst": (
        "You focus on price action and indicators: EMA crossovers (8/21/55), RSI for "
        "overbought/oversold, MACD trends, volume confirmation, Bollinger Bands. "
        "You look for trend direction and momentum."
    ),
    "Fundamental Analyst": (
        "You do comprehensive fundamental analysis across 4 dimensions: profitability "
        "(ROE, margins), financial health (debt, current ratio), valuation (P/E, P/B, P/S), "
        "and growth (revenue, earnings). You want the full picture."
    ),
    "Sentiment Analyst": (
        "You analyze market sentiment signals: news tone, insider buying/selling patterns, "
        "analyst recommendations and upgrades/downgrades. You look for divergences between "
        "sentiment and price."
    ),
}


def _data_fallback(ticker: str, start_date: str, end_date: str) -> dict:
    """Quick rule-based signal from data when LLM is unavailable (e.g., rate-limited).

    Uses financial metrics and price data to produce a basic signal
    so the system doesn't return 0 confidence on every rate-limit error.
    """
    try:
        from src.tools.api import get_prices, get_financial_metrics

        metrics = get_financial_metrics(ticker)
        score = 0

        if metrics:
            pe = metrics.get("pe_ratio")
            if pe is not None:
                if 0 < pe < 20:
                    score += 1
                elif pe > 35:
                    score -= 1

            roe = metrics.get("return_on_equity")
            if roe is not None:
                if roe > 0.15:
                    score += 1
                elif roe < 0.05:
                    score -= 1

            rev_growth = metrics.get("revenue_growth")
            if rev_growth is not None:
                if rev_growth > 0.10:
                    score += 1
                elif rev_growth < -0.05:
                    score -= 1

        df = get_prices(ticker, start_date, end_date)
        if df is not None and not df.empty:
            close = df["Close"]
            ret = float((close.iloc[-1] / close.iloc[0] - 1) * 100)
            if ret > 5:
                score += 1
            elif ret < -5:
                score -= 1

        if score >= 2:
            signal, conf = "bullish", 55
        elif score <= -2:
            signal, conf = "bearish", 55
        else:
            signal, conf = "neutral", 40

        return {"signal": signal, "confidence": conf, "reasoning": f"data fallback (score={score})"}

    except Exception:
        return {"signal": "neutral", "confidence": 20, "reasoning": "data fallback failed"}


def create_react_agent(agent_name: str):
    """Create a ReAct-style analyst agent.

    Returns a function compatible with LangGraph node signature.
    """
    philosophy = AGENT_PHILOSOPHIES.get(agent_name, "Analyze stocks thoroughly.")

    def agent_fn(state: AgentState):
        progress_message(agent_name, "running")
        tickers = state["tickers"]
        portfolio = state["portfolio"]
        start_date = state.get("start_date", "")
        end_date = state.get("end_date", "")
        show_reasoning = state.get("show_reasoning", False)

        llm_provider = portfolio.get("_llm_provider", "groq")
        llm_model = portfolio.get("_llm_model")
        llm = get_llm(llm_provider, llm_model, agent_name=agent_name)

        signals = {}

        for ticker in tickers:
            try:
                signal = _run_react_loop(
                    llm=llm,
                    agent_name=agent_name,
                    philosophy=philosophy,
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                )
                signals[ticker] = signal
            except Exception as e:
                # Try a quick data-driven fallback instead of 0 confidence
                console.print(f"  [red]LLM error for {ticker}:[/red] {e}")
                fallback = _data_fallback(ticker, start_date, end_date)
                fallback["agent"] = agent_name
                fallback["reasoning"] = f"Used data-driven fallback; {fallback['reasoning']}"
                signals[ticker] = fallback

        if show_reasoning:
            show_agent_reasoning(signals, agent_name)

        progress_message(agent_name, "done")

        return {
            "messages": [HumanMessage(content=json.dumps(signals), name=agent_name)],
            "analyst_signals": {agent_name: signals},
        }

    return agent_fn


def _run_react_loop(
    llm, agent_name: str, philosophy: str, ticker: str,
    start_date: str, end_date: str,
) -> dict:
    """Run the ReAct loop: Think → Act → Observe → Repeat → Conclude."""

    system_prompt = REACT_SYSTEM_PROMPT.format(
        agent_name=agent_name,
        philosophy=philosophy,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        max_iter=MAX_ITERATIONS,
        tools=_build_tool_descriptions(agent_name),
    )

    conversation = [system_prompt]
    tool_calls_made = 0

    for _ in range(MAX_ITERATIONS + 2):  # +2 for think + conclude steps
        # Get LLM response
        response = llm.invoke("\n\n".join(conversation))
        content = response.content.strip()
        conversation.append(f"Assistant: {content}")

        # Check if agent wants to call a tool
        if "TOOL_CALL:" in content:
            tool_line = content.split("TOOL_CALL:")[-1].strip()
            tool_name, tool_args = _parse_tool_call(tool_line)

            if tool_name and tool_calls_made < MAX_ITERATIONS:
                # Inject dates for price tool
                if tool_name == "get_prices" and isinstance(tool_args, dict):
                    tool_args.setdefault("start_date", start_date)
                    tool_args.setdefault("end_date", end_date)
                tool_args.setdefault("ticker", ticker)

                result = _execute_tool(tool_name, tool_args)
                tool_calls_made += 1
                conversation.append(
                    f"TOOL_RESULT ({tool_name}, call {tool_calls_made}/{MAX_ITERATIONS}):\n{result}"
                )
                continue

            elif tool_calls_made >= MAX_ITERATIONS:
                conversation.append(
                    "SYSTEM: Max tool calls reached. Please provide your FINAL_ANSWER now."
                )
                continue

        # Check if agent is concluding
        if "FINAL_ANSWER:" in content:
            return _parse_final_answer(content, agent_name)

    # Fallback if loop exhausts without FINAL_ANSWER
    return {
        "agent": agent_name,
        "signal": "neutral",
        "confidence": 30,
        "reasoning": "Analysis inconclusive — could not reach a clear conclusion.",
    }


def _parse_tool_call(tool_line: str) -> tuple[str | None, dict]:
    """Parse 'tool_name({"key": "val"})' format."""
    try:
        paren_idx = tool_line.index("(")
        tool_name = tool_line[:paren_idx].strip()
        args_str = tool_line[paren_idx + 1:].rstrip(")")
        args = json.loads(args_str)
        return tool_name, args
    except (ValueError, json.JSONDecodeError):
        return None, {}


def _parse_final_answer(content: str, agent_name: str) -> dict:
    """Parse the FINAL_ANSWER JSON from agent response."""
    try:
        json_str = content.split("FINAL_ANSWER:")[-1].strip()
        # Handle markdown code blocks
        if "```" in json_str:
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            json_str = json_str.strip()
        result = json.loads(json_str)
        result["agent"] = agent_name
        return result
    except (json.JSONDecodeError, IndexError):
        return {
            "agent": agent_name,
            "signal": "neutral",
            "confidence": 20,
            "reasoning": "Failed to parse final answer.",
        }
