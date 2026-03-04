"""MCP Server - Expose InvestAgent tools via Model Context Protocol.

This lets any MCP-compatible client (Claude Desktop, Cursor, VS Code Copilot)
use your stock analysis tools directly.

Usage:
    python -m src.mcp_server

Then add to your Claude Desktop config (claude_desktop_config.json):
{
    "mcpServers": {
        "investagent": {
            "command": "python",
            "args": ["-m", "src.mcp_server"],
            "cwd": "/path/to/investagent"
        }
    }
}
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "InvestAgent",
    description="AI hedge fund tools - stock data, analysis, and SEC filings. All free.",
)


# ---------------------------------------------------------------------------
# Stock Data Tools (yfinance)
# ---------------------------------------------------------------------------

@mcp.tool()
def get_stock_price(ticker: str, period: str = "1mo") -> dict:
    """Get recent stock price data for a ticker.

    Args:
        ticker: Stock symbol (e.g., AAPL, MSFT, NVDA)
        period: Time period - 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max
    """
    import yfinance as yf
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    if hist.empty:
        return {"error": f"No data found for {ticker}"}

    latest = hist.iloc[-1]
    return {
        "ticker": ticker.upper(),
        "current_price": round(float(latest["Close"]), 2),
        "open": round(float(latest["Open"]), 2),
        "high": round(float(latest["High"]), 2),
        "low": round(float(latest["Low"]), 2),
        "volume": int(latest["Volume"]),
        "period_high": round(float(hist["High"].max()), 2),
        "period_low": round(float(hist["Low"].min()), 2),
        "period_return": round(float((hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100), 2),
        "data_points": len(hist),
    }


@mcp.tool()
def get_stock_fundamentals(ticker: str) -> dict:
    """Get key financial metrics for a stock (P/E, P/B, margins, growth, etc.).

    Args:
        ticker: Stock symbol (e.g., AAPL, TSLA, GOOGL)
    """
    from src.tools.api import get_financial_metrics
    return get_financial_metrics(ticker)


@mcp.tool()
def get_stock_news(ticker: str) -> list[dict]:
    """Get recent news articles for a stock ticker.

    Args:
        ticker: Stock symbol
    """
    from src.tools.api import get_company_news
    return get_company_news(ticker)


# ---------------------------------------------------------------------------
# SEC EDGAR Tools (free, no API key)
# ---------------------------------------------------------------------------

@mcp.tool()
def get_sec_filings(ticker: str, filing_type: str = "10-K", count: int = 3) -> list[dict]:
    """Get recent SEC filings for a company.

    Args:
        ticker: Stock symbol
        filing_type: Filing type - 10-K (annual), 10-Q (quarterly), 8-K (events)
        count: Number of filings to return
    """
    from src.tools.sec_edgar import get_recent_filings
    return get_recent_filings(ticker, filing_type, count)


@mcp.tool()
def get_sec_financial_facts(ticker: str) -> dict:
    """Get structured financial data from SEC XBRL filings.

    Returns revenue, net income, assets, liabilities, equity, EPS, etc.
    with year-over-year comparisons. All from official SEC filings.

    Args:
        ticker: Stock symbol
    """
    from src.tools.sec_edgar import get_company_facts
    return get_company_facts(ticker)


@mcp.tool()
def get_sec_filing_text(ticker: str, filing_type: str = "10-K") -> str:
    """Get the text content of the most recent SEC filing.

    Returns cleaned text from the latest filing, suitable for analysis.

    Args:
        ticker: Stock symbol
        filing_type: Filing type - 10-K (annual), 10-Q (quarterly)
    """
    from src.tools.sec_edgar import get_filing_text
    text = get_filing_text(ticker, filing_type)
    return text if text else f"No {filing_type} filing found for {ticker}"


# ---------------------------------------------------------------------------
# Analysis Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def analyze_stock(ticker: str) -> dict:
    """Run all InvestAgent analysts on a single stock and return signals.

    Runs Value, Growth, Contrarian, Technical, Fundamental, and Sentiment
    analysis. Returns each agent's signal (bullish/bearish/neutral) with
    confidence scores.

    Args:
        ticker: Stock symbol to analyze
    """
    from datetime import datetime, timedelta

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    state = {
        "messages": [],
        "tickers": [ticker.upper()],
        "portfolio": {"cash": 100000, "positions": {}},
        "start_date": start_date,
        "end_date": end_date,
        "analyst_signals": {},
        "show_reasoning": False,
    }

    from src.agents import ANALYST_AGENTS
    results = {}
    for name, agent_func in ANALYST_AGENTS.items():
        try:
            result = agent_func(state)
            signals = result.get("analyst_signals", {}).get(name, {})
            results[name] = signals.get(ticker.upper(), {"signal": "error"})
        except Exception as e:
            results[name] = {"signal": "error", "error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Memory Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_analysis_history(ticker: str) -> list[dict]:
    """Get past analysis decisions for a stock from agent memory.

    Shows what signals each agent gave for this ticker in previous runs.

    Args:
        ticker: Stock symbol
    """
    from src.memory import get_past_decisions
    return get_past_decisions(ticker.upper(), limit=20)


@mcp.tool()
def get_agent_performance() -> list[dict]:
    """Get accuracy scores for all agents based on historical performance.

    Shows which agents have been most accurate in their predictions.
    """
    from src.memory import get_all_agent_accuracies
    return get_all_agent_accuracies()


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@mcp.resource("investagent://agents")
def list_agents() -> str:
    """List all available analyst agents and their descriptions."""
    return """InvestAgent Analysts:
1. Value Analyst - Classic value investing (Buffett/Graham style)
2. Growth Analyst - Revenue/earnings growth focus (Lynch/Wood style)
3. Contrarian Analyst - Oversold/contrarian opportunities (Burry style)
4. Technical Analyst - EMA, RSI, MACD, Bollinger Bands
5. Fundamental Analyst - Profitability, health, valuation metrics
6. Sentiment Analyst - News sentiment, insider trades, analyst ratings
7. Risk Manager - Position sizing and risk limits
8. Portfolio Manager - Final buy/sell/hold decisions (LLM-powered)"""


if __name__ == "__main__":
    mcp.run()
