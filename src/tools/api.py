"""Free stock data API using yfinance - no API key needed."""

import pandas as pd
import yfinance as yf
from functools import lru_cache


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------

def get_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch OHLCV price data. Returns DataFrame with columns:
    Open, High, Low, Close, Volume."""
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    if df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def get_current_price(ticker: str) -> float | None:
    """Get the latest closing price."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5d")
    if hist.empty:
        return None
    return float(hist["Close"].iloc[-1])


# ---------------------------------------------------------------------------
# Financial metrics
# ---------------------------------------------------------------------------

def get_financial_metrics(ticker: str) -> dict:
    """Get key financial metrics from Yahoo Finance (free)."""
    stock = yf.Ticker(ticker)
    info = stock.info or {}

    return {
        "ticker": ticker,
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "pb_ratio": info.get("priceToBook"),
        "ps_ratio": info.get("priceToSalesTrailing12Months"),
        "dividend_yield": info.get("dividendYield"),
        "beta": info.get("beta"),
        "eps": info.get("trailingEps"),
        "revenue": info.get("totalRevenue"),
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
        "profit_margin": info.get("profitMargins"),
        "operating_margin": info.get("operatingMargins"),
        "roe": info.get("returnOnEquity"),
        "roa": info.get("returnOnAssets"),
        "debt_to_equity": info.get("debtToEquity"),
        "current_ratio": info.get("currentRatio"),
        "quick_ratio": info.get("quickRatio"),
        "free_cash_flow": info.get("freeCashflow"),
        "total_debt": info.get("totalDebt"),
        "total_cash": info.get("totalCash"),
        "shares_outstanding": info.get("sharesOutstanding"),
        "52_week_high": info.get("fiftyTwoWeekHigh"),
        "52_week_low": info.get("fiftyTwoWeekLow"),
        "50_day_avg": info.get("fiftyDayAverage"),
        "200_day_avg": info.get("twoHundredDayAverage"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "company_name": info.get("shortName"),
    }


# ---------------------------------------------------------------------------
# Financial statements
# ---------------------------------------------------------------------------

def get_income_statement(ticker: str) -> pd.DataFrame:
    """Get income statement (annual)."""
    stock = yf.Ticker(ticker)
    return stock.financials


def get_balance_sheet(ticker: str) -> pd.DataFrame:
    """Get balance sheet (annual)."""
    stock = yf.Ticker(ticker)
    return stock.balance_sheet


def get_cash_flow(ticker: str) -> pd.DataFrame:
    """Get cash flow statement (annual)."""
    stock = yf.Ticker(ticker)
    return stock.cashflow


# ---------------------------------------------------------------------------
# News & insider trades
# ---------------------------------------------------------------------------

def get_company_news(ticker: str) -> list[dict]:
    """Get recent news articles for a ticker (free via yfinance)."""
    stock = yf.Ticker(ticker)
    news = stock.news or []
    results = []
    for article in news[:10]:
        results.append({
            "title": article.get("title", ""),
            "publisher": article.get("publisher", ""),
            "link": article.get("link", ""),
            "published": article.get("providerPublishTime", ""),
        })
    return results


def get_insider_trades(ticker: str) -> pd.DataFrame:
    """Get insider trading data (free via yfinance)."""
    stock = yf.Ticker(ticker)
    try:
        return stock.insider_transactions or pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def get_recommendations(ticker: str) -> pd.DataFrame:
    """Get analyst recommendations."""
    stock = yf.Ticker(ticker)
    try:
        return stock.recommendations or pd.DataFrame()
    except Exception:
        return pd.DataFrame()
