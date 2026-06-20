"""Streamlit Dashboard for InvestAgent.

Free deployment: streamlit.io/cloud (connect GitHub repo, done).

Usage:
    streamlit run src/dashboard.py
"""

import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `from src.* import ...` works
# regardless of how Streamlit launches this file.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta

# Bridge Streamlit Cloud secrets → environment variables
# (Streamlit Cloud stores secrets in st.secrets, but our LLM code reads os.environ)
for key in ("GROQ_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
    if key not in os.environ:
        try:
            os.environ[key] = st.secrets[key]
        except (KeyError, FileNotFoundError):
            pass

st.set_page_config(
    page_title="InvestAgent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("InvestAgent")
st.sidebar.caption("Agentic AI Hedge Fund Simulator")

page = st.sidebar.radio(
    "Navigate",
    ["Analyze Stocks", "Agent Memory", "SEC Filings", "Backtest", "How It Works"],
)

provider = st.sidebar.selectbox(
    "LLM Provider",
    ["groq", "gemini", "ollama", "openai"],
    help="Groq and Gemini have free tiers. Ollama runs locally for free.",
)

_provider_key_env = {
    "groq": "GROQ_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
}
if provider in _provider_key_env:
    env_var = _provider_key_env[provider]
    if not os.environ.get(env_var):
        entered_key = st.sidebar.text_input(
            f"{provider.title()} API key",
            type="password",
            help="Stored only in this browser session's memory. Never logged or saved to disk.",
        )
        if entered_key:
            os.environ[env_var] = entered_key
    else:
        st.sidebar.caption(f"{env_var} loaded from environment/secrets.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Agentic features:** ReAct tool-use, "
    "conditional debate, multi-turn PM, agent memory"
)
st.sidebar.markdown(
    "**Cost: $0** — yfinance + SEC EDGAR (free data), "
    "Groq/Gemini/Ollama (free LLM), Streamlit Cloud (free hosting)"
)

# ---------------------------------------------------------------------------
# Page: Analyze Stocks
# ---------------------------------------------------------------------------

if page == "Analyze Stocks":
    st.title("Stock Analysis")
    st.markdown(
        "ReAct agents autonomously choose tools, reason step-by-step, "
        "and debate when they disagree."
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        tickers_input = st.text_input(
            "Tickers (space-separated)",
            value="AAPL MSFT NVDA",
            placeholder="AAPL MSFT TSLA",
        )
    with col2:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=90),
        )
    with col3:
        end_date = st.date_input("End Date", value=datetime.now())

    tickers = [t.strip().upper() for t in tickers_input.split() if t.strip()]

    if st.button("Run Analysis", type="primary", use_container_width=True):
        if not tickers:
            st.error("Please enter at least one ticker.")
        else:
            with st.spinner("Running agentic analysis... (agents are reasoning and calling tools)"):
                try:
                    from src.main import run_hedge_fund

                    result = run_hedge_fund(
                        tickers=tickers,
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d"),
                        provider=provider,
                        selected_analysts=None,
                        show_reasoning=False,
                    )

                    # Save to session state so results persist across page navigation
                    st.session_state["last_analysis"] = {
                        "result": result,
                        "tickers": tickers,
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d"),
                    }

                except Exception as e:
                    st.error(f"Error running analysis: {e}")
                    st.exception(e)

    # --- Display results (from session state — persists across page visits) ---
    if "last_analysis" in st.session_state:
        saved = st.session_state["last_analysis"]
        result = saved["result"]
        display_tickers = saved["tickers"]
        analyst_signals = result.get("analyst_signals", {})
        pm_decisions = analyst_signals.get("Portfolio Manager", {})

        st.divider()

        # --- Workflow Path ---
        st.subheader("Workflow Path")
        agreement = result.get("agreement_score", 1.0)
        debated = result.get("debate_required", False)

        path_parts = ["Analysts (ReAct)", "Agreement Check"]
        if debated:
            path_parts.append("**DEBATE**")
        path_parts.extend(["Risk Manager", "Portfolio Manager (multi-turn)"])
        path_str = " → ".join(path_parts)

        st.markdown(f"**Path taken:** {path_str}")
        st.metric("Agent Agreement", f"{agreement:.0%}")

        # --- Debate Summary ---
        debate_summary = result.get("debate_summary", {})
        if debate_summary:
            st.subheader("Debate Summary")
            for ticker in display_tickers:
                td = debate_summary.get(ticker)
                if td:
                    st.markdown(f"**{ticker}** — Leaning: `{td.get('leaning', 'neutral')}`")
                    st.markdown(f"> **Key disagreement:** {td.get('key_disagreement', 'N/A')}")
                    st.markdown(f"> **Strongest argument:** {td.get('strongest_argument', 'N/A')}")
                    st.markdown(f"> {td.get('synthesis', '')}")

        # --- Final Decisions ---
        st.subheader("Portfolio Decisions")
        for ticker in display_tickers:
            decision = pm_decisions.get(ticker, {})
            action = decision.get("action", "hold").upper()
            qty = decision.get("quantity", 0)
            conf = decision.get("confidence", 0)
            reason = decision.get("reasoning", "")

            color = {"BUY": "🟢", "SELL": "🔴"}.get(action, "🟡")
            st.markdown(f"### {color} {ticker}: **{action}** — {qty} shares (confidence: {conf}%)")
            st.markdown(f"> {reason}")

        # --- Signal Agreement Heatmap ---
        st.subheader("Agent Signal Matrix")
        matrix_data = {}
        for agent_name, signals in analyst_signals.items():
            if agent_name in ("Risk Manager", "Portfolio Manager"):
                continue
            row = {}
            for ticker in display_tickers:
                sig = signals.get(ticker, {})
                signal = sig.get("signal", "neutral")
                row[ticker] = {"bullish": 1, "neutral": 0, "bearish": -1}.get(signal, 0)
            matrix_data[agent_name] = row

        if matrix_data:
            df_matrix = pd.DataFrame(matrix_data).T
            st.dataframe(
                df_matrix.style.applymap(
                    lambda v: (
                        "background-color: #2e7d32; color: white" if v == 1
                        else "background-color: #c62828; color: white" if v == -1
                        else "background-color: #555"
                    )
                ),
                width="stretch",
            )
            st.caption("Green = Bullish (+1) | Gray = Neutral (0) | Red = Bearish (-1)")

        # --- Confidence Chart ---
        st.subheader("Agent Confidence Levels")
        conf_data = {}
        for agent_name, signals in analyst_signals.items():
            if agent_name in ("Risk Manager", "Portfolio Manager"):
                continue
            for ticker in display_tickers:
                sig = signals.get(ticker, {})
                conf_data.setdefault(ticker, {})[agent_name] = sig.get("confidence", 0)

        if conf_data:
            df_conf = pd.DataFrame(conf_data)
            st.bar_chart(df_conf)

        # --- Price Charts ---
        st.subheader("Price Charts")
        from src.tools.api import get_prices

        cols = st.columns(min(len(display_tickers), 3))
        for i, ticker in enumerate(display_tickers):
            with cols[i % len(cols)]:
                prices = get_prices(ticker, saved["start_date"], saved["end_date"])
                if not prices.empty:
                    st.markdown(f"**{ticker}**")
                    st.line_chart(prices["Close"])


# ---------------------------------------------------------------------------
# Page: Agent Memory
# ---------------------------------------------------------------------------

elif page == "Agent Memory":
    st.title("Agent Memory & Track Record")
    st.markdown("View past decisions and agent performance across sessions.")

    from src.memory import get_recent_runs, get_all_agent_accuracies, get_ticker_history
    from src.scoring import score_pending_decisions, MIN_AGE_DAYS

    if st.button("Score decisions older than 30 days"):
        scored = score_pending_decisions()
        if scored:
            st.success(f"Scored {scored} decision(s) against current prices.")
        else:
            st.info(f"No decisions are eligible yet (need to be {MIN_AGE_DAYS}+ days old).")

    # Agent accuracy
    st.subheader("Agent Accuracy Scores")
    accuracies = get_all_agent_accuracies()
    if accuracies:
        df_acc = pd.DataFrame(accuracies)
        st.dataframe(df_acc, width="stretch")
    else:
        st.info("No accuracy data yet. Run analysis a few times to build up history.")

    # Recent runs
    st.subheader("Recent Runs")
    runs = get_recent_runs(20)
    if runs:
        df_runs = pd.DataFrame(runs)
        st.dataframe(df_runs, width="stretch")
    else:
        st.info("No runs recorded yet.")

    # Ticker history lookup
    st.subheader("Ticker Decision History")
    lookup = st.text_input("Look up ticker history", placeholder="AAPL")
    if lookup:
        history = get_ticker_history(lookup.upper())
        if history:
            df_hist = pd.DataFrame(history)
            st.dataframe(df_hist, width="stretch")
        else:
            st.info(f"No history found for {lookup.upper()}")


# ---------------------------------------------------------------------------
# Page: SEC Filings
# ---------------------------------------------------------------------------

elif page == "SEC Filings":
    st.title("SEC EDGAR Filings")
    st.markdown("Free access to company filings and structured financial data from the SEC.")

    ticker = st.text_input("Ticker", value="AAPL")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("XBRL Financial Facts")
        if st.button("Fetch SEC Data", key="sec_facts"):
            with st.spinner("Fetching from SEC EDGAR..."):
                from src.tools.sec_edgar import get_company_facts

                facts = get_company_facts(ticker.upper())
                if facts:
                    for metric, data in facts.items():
                        val = data.get("value")
                        prior = data.get("prior_value")
                        if val is not None:
                            change = ""
                            if prior and prior != 0:
                                pct = (val - prior) / abs(prior) * 100
                                arrow = "↑" if pct > 0 else "↓"
                                change = f" {arrow} {abs(pct):.1f}%"
                            if isinstance(val, (int, float)) and abs(val) > 1_000_000:
                                st.metric(metric, f"${val / 1_000_000:,.1f}M", change or None)
                            else:
                                st.metric(metric, f"{val}", change or None)
                else:
                    st.warning("No SEC data found. Check ticker symbol.")

    with col2:
        st.subheader("Recent Filings")
        filing_type = st.selectbox("Filing Type", ["10-K", "10-Q", "8-K"])
        if st.button("Fetch Filings", key="sec_filings"):
            with st.spinner("Fetching filing list..."):
                from src.tools.sec_edgar import get_recent_filings

                filings = get_recent_filings(ticker.upper(), filing_type, count=5)
                if filings:
                    for f in filings:
                        st.markdown(f"**{f['type']}** — {f['date']}")
                        st.markdown(f"[View on SEC]({f['url']})")
                        st.markdown("---")
                else:
                    st.warning("No filings found.")


# ---------------------------------------------------------------------------
# Page: Backtest
# ---------------------------------------------------------------------------

elif page == "Backtest":
    st.title("Backtest")
    st.markdown("Simulate the agentic hedge fund over a historical period.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        bt_tickers = st.text_input("Tickers", value="AAPL MSFT", key="bt_tickers")
    with col2:
        bt_start = st.date_input("Start", value=datetime(2024, 1, 1), key="bt_start")
    with col3:
        bt_end = st.date_input("End", value=datetime(2024, 12, 31), key="bt_end")
    with col4:
        bt_step = st.number_input("Rebalance (days)", value=30, min_value=7, max_value=90)

    all_analysts = [
        "Value Analyst", "Growth Analyst", "Contrarian Analyst",
        "Technical Analyst", "Fundamental Analyst", "Sentiment Analyst",
    ]
    bt_analysts = st.multiselect(
        "Analysts (fewer = faster, avoids rate limits)",
        all_analysts,
        default=["Value Analyst", "Growth Analyst", "Sentiment Analyst"],
        help="3 analysts is ~2x faster than 6. Free-tier APIs rate-limit at ~30 req/min.",
    )

    if st.button("Run Backtest", type="primary", use_container_width=True):
        tickers = [t.strip().upper() for t in bt_tickers.split() if t.strip()]
        selected = bt_analysts if bt_analysts else None
        if not tickers:
            st.error("Enter at least one ticker.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def on_period_complete(period_num, total_periods, entry):
                """Update Streamlit progress bar after each period."""
                pct = period_num / max(total_periods, 1)
                progress_bar.progress(pct)
                ret = entry["return_pct"]
                status_text.markdown(
                    f"**Period {period_num}/{total_periods}** — "
                    f"{entry['date']} — "
                    f"Value: ${entry['total_value']:,.0f} ({ret:+.1f}%)"
                )

            try:
                from src.backtester import backtest

                history, portfolio = backtest(
                    tickers=tickers,
                    start_date=bt_start.strftime("%Y-%m-%d"),
                    end_date=bt_end.strftime("%Y-%m-%d"),
                    provider=provider,
                    step_days=bt_step,
                    on_period_complete=on_period_complete,
                    selected_analysts=selected,
                )

                progress_bar.progress(1.0)
                status_text.markdown("**Backtest complete!**")

                if history:
                    st.session_state["last_backtest"] = {
                        "history": history,
                        "portfolio": portfolio,
                        "tickers": tickers,
                    }

            except Exception as e:
                st.error(f"Backtest error: {e}")
                st.exception(e)

    # --- Display results (from session state — persists across page visits) ---
    if "last_backtest" in st.session_state:
        saved = st.session_state["last_backtest"]
        bt_history = saved["history"]
        bt_portfolio = saved["portfolio"]
        bt_display_tickers = saved["tickers"]

        st.divider()

        df_hist = pd.DataFrame(bt_history)
        final = bt_history[-1]

        # Only show the chart if portfolio value actually changed
        values = df_hist["total_value"]
        value_changed = values.max() - values.min() > 1  # more than $1 variation

        if value_changed:
            st.subheader("Portfolio Value Over Time")
            st.line_chart(df_hist.set_index("date")["total_value"])

        # Return
        col1, col2, col3 = st.columns(3)
        col1.metric("Initial", f"$100,000")
        col2.metric("Final", f"${final['total_value']:,.0f}")
        col3.metric("Return", f"{final['return_pct']:+.1f}%")

        if not value_changed:
            st.info(
                "Portfolio value was flat — the PM chose **hold** every period. "
                "This typically happens when analysts return low-confidence signals "
                "or when rate limits prevent LLM calls."
            )

        # Decision timeline
        st.subheader("Decision Timeline")
        st.dataframe(df_hist, width="stretch")

        # Final positions
        st.subheader("Final Positions")
        for t in bt_display_tickers:
            pos = bt_portfolio["positions"][t]
            st.write(f"**{t}**: {pos['shares']} shares (avg cost: ${pos['avg_cost']:.2f})")
        st.write(f"**Cash**: ${bt_portfolio['cash']:,.2f}")


# ---------------------------------------------------------------------------
# Page: How It Works
# ---------------------------------------------------------------------------

elif page == "How It Works":
    st.title("How InvestAgent Works")
    st.markdown(
        "A detailed walkthrough of every algorithm, agent, and design decision "
        "inside this multi-agent AI hedge fund simulator."
    )

    # ── 1. Overview ───────────────────────────────────────────────────────
    st.header("1. System Overview")
    st.markdown(
        "InvestAgent is a **multi-agent system** built on "
        "[LangGraph](https://github.com/langchain-ai/langgraph). "
        "Instead of a single prompt producing an answer, six autonomous analyst agents "
        "each independently research a stock, debate when they disagree, and a portfolio "
        "manager makes the final call. Everything runs on free-tier LLMs."
    )

    st.code("""
User Input (tickers, date range, cash)
    |
    v
+-------------------------------------------------------+
|  6 Analyst Agents run in PARALLEL                     |
|  (Value, Growth, Contrarian, Technical,               |
|   Fundamental, Sentiment)                             |
|  Each runs a ReAct loop: Think -> Act -> Observe x4   |
+-------------------------------------------------------+
    |
    v
Agreement Check  -- "Do the agents agree?"
    |                  (measure consensus)
    |
    +-- >= 60% agree -----> Risk Manager
    |
    +-- < 60% agree ------> Debate (LLM moderator)
                                |
                                v
                            Risk Manager
                                |
                                v
                         Portfolio Manager
                      (multi-turn ReAct agent)
                                |
                                v
                      Final BUY / SELL / HOLD
                      + Save to Memory (SQLite)
""", language=None)

    st.info(
        "**What makes this 'agentic' vs. a regular LLM call?** "
        "Each agent autonomously decides *which* tools to call and *how many times* "
        "to loop. The debate only happens when there's real disagreement. The portfolio manager "
        "can fetch additional data before deciding. None of this is hardcoded — the LLM controls the flow."
    )

    # ── 2. ReAct ──────────────────────────────────────────────────────────
    st.header("2. The ReAct Agent Pattern")
    st.markdown(
        "**ReAct** (Reasoning + Acting) is an agent design pattern from the "
        "[2022 paper by Yao et al.](https://arxiv.org/abs/2210.03629) "
        "The idea: instead of the LLM answering in one shot, it enters a loop:"
    )
    st.markdown("""
1. **Think** — "I need the financial metrics to check the P/E ratio."
2. **Act** — Calls `get_financial_metrics({"ticker": "AAPL"})`
3. **Observe** — Reads the result: P/E = 28.5, ROE = 147%, ...
4. **Repeat** — "Now I need price data for technical confirmation."
5. **Conclude** — Emits a final signal: bullish / neutral / bearish + confidence
""")
    st.markdown(
        "Each agent gets up to **4 tool calls** per ticker (`MAX_ITERATIONS = 4`). "
        "This prevents runaway API costs while still allowing meaningful research."
    )
    st.markdown(
        "The agent outputs `TOOL_CALL: tool_name({\"key\": \"value\"})` which the runner "
        "parses, executes, and appends back into the conversation. Key parameters like "
        "`ticker`, `start_date`, and `end_date` are auto-injected."
    )
    st.markdown(
        "If an agent uses all 4 iterations without concluding, the system returns a "
        "**neutral signal with 30% confidence** — a safe fallback."
    )

    # ── 3. The Six Analysts ───────────────────────────────────────────────
    st.header("3. The Six Analyst Agents")
    st.markdown(
        "Each analyst has a distinct **investment philosophy** embedded in its system prompt "
        "and a **scoring algorithm** that converts financial data into a signal. "
        "They all share the same 7 tools, but each cares about different metrics."
    )

    # Value
    st.subheader("3.1 Value Analyst")
    st.markdown(
        "**Philosophy:** Buffett & Graham — buy quality companies trading below intrinsic value. "
        "Looks for a 'margin of safety' between price and what the business is actually worth."
    )
    st.dataframe(
        pd.DataFrame({
            "Metric": ["ROE", "Profit Margin", "Debt/Equity", "Current Ratio", "P/E Ratio", "P/B Ratio", "FCF Yield"],
            "Bullish": [">15% (+2)", ">20% (+2)", "<50 (+2)", ">1.5 (+1)", "0-15 (+2)", "0-1.5 (+2)", ">8% (+2)"],
            "Neutral": ["10-15% (+1)", "10-20% (+1)", "50-100 (+1)", "1.0-1.5 (0)", "15-20 (+1)", "1.5-3 (+1)", "5-8% (+1)"],
            "Bearish": ["<5% (-1)", "<0% (-1)", ">100 (-1)", "<1.0 (-1)", ">30 (-1)", ">5 (-1)", "—"],
        }),
        width="stretch", hide_index=True,
    )
    st.markdown("Score ranges from **-12 to +12**. Score >= 4 = bullish, <= -2 = bearish.")

    # Growth
    st.subheader("3.2 Growth Analyst")
    st.markdown(
        "**Philosophy:** Peter Lynch & Cathie Wood — find companies growing revenue and earnings "
        "faster than the market. Willing to pay premium valuations for high growth."
    )
    st.dataframe(
        pd.DataFrame({
            "Metric": ["Revenue Growth", "Earnings Growth", "PEG Ratio", "Operating Margin", "20-Day Return"],
            "Bullish": [">25% (+3)", ">25% (+3)", "<1.0 (+2)", ">25% (+2)", ">10% (+1)"],
            "Bearish": ["<0% (-2)", "<0% (-2)", ">3.0 (-1)", "—", "<-15% (-1)"],
        }),
        width="stretch", hide_index=True,
    )
    st.success(
        "**Why PEG matters:** The PEG ratio (P/E / earnings growth) normalizes valuation for growth. "
        "PEG < 1.0 means you're paying less than '1x' for each point of growth — "
        "Lynch's favorite signal that a stock is undervalued relative to its trajectory."
    )

    # Contrarian
    st.subheader("3.3 Contrarian Analyst")
    st.markdown(
        "**Philosophy:** Michael Burry & Bill Ackman — 'buy fear, sell greed.' "
        "Looks for stocks the market has beaten down too far, as long as the business is fundamentally sound."
    )
    st.dataframe(
        pd.DataFrame({
            "Signal": [
                "Near 52-week low", "Near 52-week high", "Discount to 200D MA >20%",
                "Premium to 200D MA >20%", "Solid fundamentals + positive FCF",
                "High volatility >40%", "Low P/E <12",
            ],
            "Interpretation": [
                "Oversold, opportunity", "Overbought, risk", "Depressed price",
                "Overextended", "Value despite low price",
                "Price instability = opportunity", "Cheap despite problems",
            ],
            "Points": ["+2", "-2", "+2", "-1", "+2", "+1", "+2"],
        }),
        width="stretch", hide_index=True,
    )

    # Technical
    st.subheader("3.4 Technical Analyst")
    st.markdown(
        "**Philosophy:** Pure price action. Doesn't care about the company's business — "
        "only what the chart says about momentum, trend, and mean reversion."
    )
    st.markdown("**Indicator suite:**")

    tech_col1, tech_col2 = st.columns(2)
    with tech_col1:
        st.markdown("""
**EMA (Exponential Moving Averages)** — 8/21/55 periods
- EMA8 > EMA21 > EMA55 → Strong uptrend (+2)
- EMA8 < EMA21 < EMA55 → Strong downtrend (-2)

**RSI (Relative Strength Index, period=14)**
- RSI < 30 → Oversold (+2)
- RSI > 70 → Overbought (-2)

**Volume Confirmation**
- Volume > 1.5x 20-day avg validates conviction (+/- 1)
""")
    with tech_col2:
        st.markdown("""
**MACD (Moving Average Convergence Divergence)**
- Histogram crosses above 0 → Bullish (+2)
- Histogram crosses below 0 → Bearish (-2)

**Bollinger Bands** (20-period, 2 std dev)
- Price below lower band → Oversold (+1)
- Price above upper band → Overbought (-1)
""")
    st.markdown("Signal threshold: score >= 3 = bullish, <= -3 = bearish.")

    # Fundamental
    st.subheader("3.5 Fundamental Analyst")
    st.markdown(
        "**Philosophy:** Holistic analysis across four dimensions. Unlike value (cheapness) or "
        "growth (trajectory), tries to see the complete picture."
    )
    st.dataframe(
        pd.DataFrame({
            "Dimension": ["Profitability", "Financial Health", "Valuation", "Growth"],
            "Key Metrics": [
                "ROE, Profit Margin, Operating Margin",
                "Debt/Equity, Current Ratio, FCF",
                "P/E, P/B, P/S",
                "Revenue Growth, Earnings Growth",
            ],
            "Score Range": ["-3 to +5", "-3 to +4", "-4 to +4", "-2 to +4"],
        }),
        width="stretch", hide_index=True,
    )
    st.markdown("Total range ~**-8 to +8**. Score >= 5 = bullish, <= -3 = bearish.")

    # Sentiment
    st.subheader("3.6 Sentiment Analyst")
    st.markdown(
        "**Philosophy:** The market is driven by people. Track what they're saying, "
        "what insiders are doing, and what Wall Street recommends."
    )
    st.markdown("""
**Three independent signal sources:**

1. **News Sentiment** — Keyword matching on recent headlines. Counts positive words
   (*beat, surge, upgrade, record*) vs. negative (*miss, crash, downgrade, weak*).
2. **Insider Trades** — Executives buying their own stock = confidence (+1).
   Selling > 2x buys = warning (-1).
3. **Analyst Recommendations** — Buy upgrades vs. sell downgrades. More buys = +1.
""")
    st.markdown("Combined score range: **-4 to +4**. Score >= 2 = bullish, <= -2 = bearish.")

    # ── 4. Agreement & Debate ─────────────────────────────────────────────
    st.header("4. Agreement Check & Debate")
    st.markdown("After all six analysts finish, the system asks: **Do they agree?**")

    st.subheader("Measuring consensus")
    st.code("""
AAPL signals: [bullish, bullish, bullish, bearish, bearish, neutral]
Majority: bullish (3 out of 6) = 50% agreement

MSFT signals: [bullish, bullish, bullish, bullish, neutral, neutral]
Majority: bullish (4 out of 6) = 67% agreement

Overall agreement = average(50%, 67%) = 58.5%
Since 58.5% < 60% threshold → DEBATE TRIGGERED
""", language=None)

    st.subheader("The Debate mechanism")
    st.markdown(
        "A **senior investment committee moderator** (an LLM with a special prompt) receives "
        "every analyst's signal, confidence, and reasoning — plus each analyst's **historical "
        "track record** from the memory database."
    )
    st.markdown("The moderator produces a synthesis:")
    st.code("""
{
    "synthesis":            "Value and Fundamental see strong margins, but Technical
                             shows a bearish EMA crossover...",
    "leaning":              "bullish",
    "confidence_adjustment": +10,
    "key_disagreement":     "Short-term momentum vs. long-term value",
    "strongest_argument":   "Value Analyst (75% accuracy) — P/E of 14 with 22% ROE"
}
""", language="json")

    st.success(
        "**Why track records matter in debate:** If the Value Analyst has 75% historical accuracy "
        "and the Technical Analyst has 58%, the moderator rationally weights value higher. "
        "This is **meta-learning** — the system learns which of its own agents to trust."
    )

    # ── 5. Risk Manager ──────────────────────────────────────────────────
    st.header("5. Risk Manager")
    st.markdown(
        "Before the Portfolio Manager can act, the Risk Manager enforces hard constraints. "
        "This prevents any single stock from dominating the portfolio."
    )

    st.subheader("The 20% Rule")
    st.code("""
Portfolio total value = cash + sum(position values)
Max per ticker        = 20% of total
Remaining limit       = max(0, max_per_ticker - current_position_value)
Final limit           = min(remaining_limit, available_cash)
Max shares            = floor(final_limit / current_price)
""", language=None)

    st.markdown("**Example:**")
    st.code("""
Portfolio:  $100,000 cash + $50,000 positions = $150,000 total
20% max:    $30,000 per ticker
AAPL held:  $10,000 currently
Remaining:  $30,000 - $10,000 = $20,000
Cash OK:    $100,000 available (no constraint)
Final:      Can buy up to $20,000 more of AAPL
""", language=None)

    st.markdown(
        "The Portfolio Manager receives these limits and is **never allowed to exceed them**. "
        "Quantities are clamped after the PM decides."
    )

    # ── 6. Portfolio Manager ──────────────────────────────────────────────
    st.header("6. Portfolio Manager")
    st.markdown(
        "The PM is the **final decision maker** and is itself a ReAct agent with up to "
        "**3 tool calls** per ticker. It has the most context of any node."
    )

    st.subheader("Information available to the PM")
    st.dataframe(
        pd.DataFrame({
            "Input": [
                "6 analyst signals + confidence",
                "Debate synthesis (if triggered)",
                "Each analyst's historical accuracy",
                "Past decisions on this ticker",
                "Risk limits (max shares, remaining $)",
                "Current portfolio (cash, positions)",
                "Additional tool access",
            ],
            "Source": [
                "Analyst nodes",
                "Debate node",
                "Memory DB (agent_scores)",
                "Memory DB (decisions)",
                "Risk Manager",
                "State",
                "API tools (metrics, news, SEC)",
            ],
        }),
        width="stretch", hide_index=True,
    )

    st.subheader("Decision rules")
    st.markdown("""
1. Weight analyst signals by their track record accuracy
2. If debate occurred, seriously consider the synthesis
3. Only buy with strong conviction AND available limit
4. Only sell if holding shares AND bearish conviction
5. Never exceed the risk manager's max shares
6. Be conservative — when in doubt, hold
7. Explain the reasoning chain, not just the conclusion
""")

    st.markdown(
        "Even after the PM decides, a **safety clamp** ensures constraints are respected: "
        "buy quantity is capped at risk manager's max, sell quantity is capped at shares held."
    )

    # ── 7. Memory ─────────────────────────────────────────────────────────
    st.header("7. Memory & Learning")
    st.markdown(
        "InvestAgent uses a **SQLite database** (`investagent_memory.db`) to persist "
        "decisions across sessions. This is what separates it from a stateless LLM wrapper."
    )

    st.subheader("Three tables")
    st.dataframe(
        pd.DataFrame({
            "Table": ["decisions", "runs", "agent_scores"],
            "Stores": [
                "Every agent signal, PM action, price at decision, 30-day outcome",
                "Run metadata: tickers, provider, return %",
                "Per-agent accuracy: signal, was_correct (1/0)",
            ],
            "Used by": [
                "PM (reviews past decisions on this ticker)",
                "Dashboard, backtester",
                "Debate moderator, PM (track record weighting)",
            ],
        }),
        width="stretch", hide_index=True,
    )

    st.subheader("The learning loop")
    st.code("""
Run 1:  Agent makes prediction --> saved to memory
           |
           v  (30+ days later)
Run N:  Check actual price movement
        --> Mark outcome as "correct" or "incorrect"
        --> Update agent_scores
           |
           v
Run N+1: Debate moderator and PM receive track records
         --> Weight signals by historical accuracy
         --> Better-performing agents get more influence
""", language=None)

    st.success(
        "Over many runs, agents that are consistently wrong get discounted, and agents that "
        "are consistently right gain more influence. This is a lightweight form of "
        "**online meta-learning** without retraining any model weights."
    )

    # ── 8. LLM Routing ───────────────────────────────────────────────────
    st.header("8. Intelligent LLM Routing")
    st.markdown(
        "Not all agents need the same model. The system uses a **two-tier routing strategy** "
        "to balance cost and capability:"
    )

    st.dataframe(
        pd.DataFrame({
            "Agent": [
                "Value Analyst", "Growth Analyst", "Contrarian Analyst",
                "Technical Analyst", "Fundamental Analyst", "Sentiment Analyst",
                "Risk Manager", "**Portfolio Manager**", "**Debate Moderator**",
            ],
            "Tier": [
                "fast", "fast", "fast", "fast", "fast", "fast",
                "fast", "**smart**", "**smart**",
            ],
            "Groq Model": [
                "Llama 3.1 8B", "Llama 3.1 8B", "Llama 3.1 8B",
                "Llama 3.1 8B", "Llama 3.1 8B", "Llama 3.1 8B",
                "Llama 3.1 8B", "**Llama 3.3 70B**", "**Llama 3.3 70B**",
            ],
            "Gemini Model": [
                "Flash Lite", "Flash Lite", "Flash Lite",
                "Flash Lite", "Flash Lite", "Flash Lite",
                "Flash Lite", "**Flash**", "**Flash**",
            ],
        }),
        width="stretch", hide_index=True,
    )

    st.info(
        "**Why this works:** Analyst agents do relatively simple scoring (compare metrics to thresholds). "
        "An 8B model handles this well. But the Portfolio Manager must synthesize conflicting signals, "
        "weigh track records, consider risk limits, and form a nuanced decision — that's a job for a 70B model. "
        "All of this runs on **free-tier APIs**. Temperature = 0 for deterministic outputs."
    )

    # ── 9. Data Tools ─────────────────────────────────────────────────────
    st.header("9. Data Tools (All Free)")
    st.markdown(
        "Every tool in the system is **free, no API key required** (except the LLM itself). "
        "Two data sources: Yahoo Finance (via yfinance) and SEC EDGAR."
    )

    tool_col1, tool_col2 = st.columns(2)
    with tool_col1:
        st.subheader("Yahoo Finance (yfinance)")
        st.markdown("""
| Tool | Returns |
|------|---------|
| `get_prices` | OHLCV + EMAs (8/21/55), RSI-14, volatility |
| `get_financial_metrics` | 28 fields: P/E, ROE, margins, growth |
| `get_company_news` | Recent articles with titles & publishers |
| `get_insider_trades` | Insider buy/sell transactions |
| `get_recommendations` | Analyst upgrade/downgrade history |
""")
    with tool_col2:
        st.subheader("SEC EDGAR (no key needed)")
        st.markdown("""
| Tool | Returns |
|------|---------|
| `get_sec_financial_facts` | XBRL data: revenue, net income, assets |
| `get_sec_recent_filings` | 10-K, 10-Q filing dates & metadata |
""")

    st.markdown(
        "**Smart price summarization:** Raw price data can be 90+ rows. Instead of dumping it all "
        "into the LLM context, the system computes a summary: latest close, period return, EMAs, "
        "RSI, volatility, last 5 closes. This saves tokens and helps smaller models focus."
    )

    # ── 10. Backtesting ──────────────────────────────────────────────────
    st.header("10. Backtesting")
    st.markdown(
        "The backtester runs the **entire multi-agent pipeline** repeatedly over historical windows "
        "to simulate how the system would have performed over time."
    )

    st.subheader("Rolling window algorithm")
    st.code("""
|--- 90-day lookback ---|--- 30-day step ---|
                        ^                   ^
                    window_start        window_end

Step 1: Run hedge fund on window --> get buy/sell/hold decisions
Step 2: Execute decisions against the portfolio
Step 3: Record portfolio value
Step 4: Slide window forward by 30 days
Step 5: Repeat until end date
""", language=None)

    st.markdown("""
**Key properties:**
- Uses **actual historical prices** from Yahoo Finance
- Each window's decisions compound into the next (realistic portfolio evolution)
- 90-day lookback gives technical indicators enough history
- All agent decisions are persisted to memory, enabling the learning loop
""")

    # ── 11. LangGraph Wiring ─────────────────────────────────────────────
    st.header("11. LangGraph Wiring")
    st.markdown(
        "The entire workflow is expressed as a **directed graph** using LangGraph's `StateGraph`:"
    )
    st.code("""
graph = StateGraph(AgentState)

# Add all nodes
graph.add_node("Value Analyst", value_agent)
graph.add_node("Growth Analyst", growth_agent)
graph.add_node("Contrarian Analyst", contrarian_agent)
graph.add_node("Technical Analyst", technical_agent)
graph.add_node("Fundamental Analyst", fundamental_agent)
graph.add_node("Sentiment Analyst", sentiment_agent)
graph.add_node("Agreement Check", check_agreement)
graph.add_node("Debate", debate_agents)
graph.add_node("Risk Manager", risk_manager)
graph.add_node("Portfolio Manager", portfolio_manager)

# START -> all analysts run in parallel
for name in analyst_names:
    graph.add_edge("__start__", name)

# All analysts -> Agreement Check (waits for all to finish)
for name in analyst_names:
    graph.add_edge(name, "Agreement Check")

# Conditional branch: debate or skip
graph.add_conditional_edges(
    "Agreement Check",
    should_debate,     # returns "Debate" or "Risk Manager"
    {"Debate": "Debate", "Risk Manager": "Risk Manager"},
)

# Debate -> Risk Manager -> Portfolio Manager -> END
graph.add_edge("Debate", "Risk Manager")
graph.add_edge("Risk Manager", "Portfolio Manager")
graph.add_edge("Portfolio Manager", END)
""", language="python")

    st.markdown("""
**State management patterns:**
- **Message accumulation** — Messages from all agents are appended to a shared list via
  `operator.add`, preserving the full conversation trail.
- **Dict merging** — Each analyst adds its signals to a shared `analyst_signals` dict
  via a custom `merge_dicts` reducer. No agent overwrites another's output.
""")

    # ── 12. Key Takeaways ────────────────────────────────────────────────
    st.header("12. Key Takeaways")

    st.success(
        "**1. Agentic ≠ one big prompt.** "
        "Each agent autonomously decides what data to fetch. The debate only fires when "
        "there's real disagreement. The graph branches dynamically."
    )
    st.success(
        "**2. Diverse perspectives beat a single model.** "
        "Six agents with different philosophies naturally surface different aspects of a stock. "
        "Their disagreements are information, not noise."
    )
    st.success(
        "**3. Memory enables meta-learning.** "
        "By tracking which agents are historically accurate and feeding that back into debate "
        "and PM, the system learns who to trust — without retraining any weights."
    )
    st.success(
        "**4. Smart resource allocation.** "
        "Fast 8B models for scoring agents, capable 70B models for the PM and debate. "
        "All on free tiers. Zero cost."
    )
    st.success(
        "**5. Risk constraints are non-negotiable.** "
        "The 20% per-ticker limit is enforced *after* the PM decides, not *by* the PM. "
        "Hard limits prevent the LLM from making catastrophic allocation errors."
    )

    st.warning(
        "**Disclaimer:** This is an educational simulator. The agents use simplified scoring "
        "heuristics and free-tier LLMs. Real hedge funds use proprietary data, vastly more "
        "compute, and human oversight. Do not use this for actual trading decisions."
    )
