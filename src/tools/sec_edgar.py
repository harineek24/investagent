"""SEC EDGAR API - Free access to company filings.

SEC EDGAR is completely free. No API key required.
Just needs a User-Agent header with your email (SEC policy).

Set EDGAR_USER_AGENT in .env or it defaults to a generic one.
"""

import os
import gzip
import io
import json
import re
import urllib.request
import urllib.error
from functools import lru_cache

EDGAR_BASE = "https://efts.sec.gov/LATEST"
EDGAR_FILINGS = "https://data.sec.gov"
USER_AGENT = os.environ.get("EDGAR_USER_AGENT", "InvestAgent research@example.com")
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def _read_response(resp) -> bytes:
    """Read response, decompressing gzip if needed."""
    raw = resp.read()
    if resp.headers.get("Content-Encoding") == "gzip":
        raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
    return raw


def _fetch_json(url: str) -> dict | list | None:
    """Fetch JSON from SEC EDGAR."""
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(_read_response(resp).decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError):
        return None


def _fetch_text(url: str) -> str:
    """Fetch raw text from SEC EDGAR."""
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return _read_response(resp).decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return ""


@lru_cache(maxsize=100)
def get_cik(ticker: str) -> str | None:
    """Look up a company's CIK number from ticker."""
    data = _fetch_json("https://www.sec.gov/files/company_tickers.json")
    if not data:
        return None
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    return None


def get_recent_filings(ticker: str, filing_type: str = "10-K", count: int = 3) -> list[dict]:
    """Get recent filings metadata for a company.

    Args:
        ticker: Stock ticker symbol
        filing_type: "10-K" (annual), "10-Q" (quarterly), "8-K" (events)
        count: Number of filings to return
    """
    cik = get_cik(ticker)
    if not cik:
        return []

    url = f"{EDGAR_FILINGS}/submissions/CIK{cik}.json"
    data = _fetch_json(url)
    if not data:
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    results = []
    for i, form in enumerate(forms):
        if form == filing_type and len(results) < count:
            accession_clean = accessions[i].replace("-", "")
            results.append({
                "type": form,
                "date": dates[i],
                "accession": accessions[i],
                "url": f"{EDGAR_FILINGS}/Archives/edgar/data/{cik.lstrip('0')}/{accession_clean}/{primary_docs[i]}",
            })

    return results


def get_filing_text(ticker: str, filing_type: str = "10-K") -> str:
    """Get the most recent filing's text content (simplified).

    Returns a cleaned, truncated version suitable for LLM analysis.
    """
    filings = get_recent_filings(ticker, filing_type, count=1)
    if not filings:
        return ""

    raw = _fetch_text(filings[0]["url"])
    if not raw:
        return ""

    # Strip HTML tags for clean text
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)

    # Truncate to ~8000 chars (fits in most LLM context windows)
    return text[:8000].strip()


def get_company_facts(ticker: str) -> dict:
    """Get structured XBRL facts from SEC (revenue, net income, assets, etc.).

    This is structured data - no parsing needed. Free from SEC.
    """
    cik = get_cik(ticker)
    if not cik:
        return {}

    url = f"{EDGAR_FILINGS}/api/xbrl/companyfacts/CIK{cik}.json"
    data = _fetch_json(url)
    if not data:
        return {}

    facts = data.get("facts", {}).get("us-gaap", {})
    result = {}

    # Extract key financial metrics from XBRL
    key_metrics = {
        "Revenues": "revenue",
        "NetIncomeLoss": "net_income",
        "Assets": "total_assets",
        "Liabilities": "total_liabilities",
        "StockholdersEquity": "stockholders_equity",
        "OperatingIncomeLoss": "operating_income",
        "EarningsPerShareBasic": "eps_basic",
        "CashAndCashEquivalentsAtCarryingValue": "cash",
        "LongTermDebt": "long_term_debt",
        "CommonStockSharesOutstanding": "shares_outstanding",
    }

    for xbrl_key, friendly_name in key_metrics.items():
        fact = facts.get(xbrl_key)
        if not fact:
            continue
        units = fact.get("units", {})
        # Get USD values or shares
        values = units.get("USD") or units.get("USD/shares") or units.get("shares")
        if values:
            # Get most recent annual (10-K) values
            annual = [v for v in values if v.get("form") == "10-K"]
            if annual:
                recent = sorted(annual, key=lambda x: x.get("end", ""), reverse=True)
                result[friendly_name] = {
                    "value": recent[0].get("val"),
                    "period_end": recent[0].get("end"),
                    "filed": recent[0].get("filed"),
                }
                if len(recent) > 1:
                    result[friendly_name]["prior_value"] = recent[1].get("val")
                    result[friendly_name]["prior_period"] = recent[1].get("end")

    return result


def get_insider_filings(ticker: str, count: int = 10) -> list[dict]:
    """Get recent insider transaction filings (Form 4) from SEC."""
    cik = get_cik(ticker)
    if not cik:
        return []

    url = f"{EDGAR_FILINGS}/submissions/CIK{cik}.json"
    data = _fetch_json(url)
    if not data:
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])

    results = []
    for i, form in enumerate(forms):
        if form in ("4", "4/A") and len(results) < count:
            results.append({
                "type": form,
                "date": dates[i],
            })

    return results
