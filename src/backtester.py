"""Backtester - Simulate the hedge fund over historical periods.

Usage:
    python -m src.backtester --tickers AAPL MSFT --start-date 2024-01-01 --end-date 2024-12-31
"""

import argparse
import time
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.main import run_hedge_fund
from src.tools.api import get_prices

console = Console()


def backtest(
    tickers: list[str],
    start_date: str,
    end_date: str,
    provider: str = "groq",
    model: str | None = None,
    initial_cash: float = 100_000,
    step_days: int = 30,
    on_period_complete: callable = None,
    selected_analysts: list[str] | None = None,
    period_delay: float = 2.0,
):
    """Run the hedge fund over rolling time windows.

    Args:
        on_period_complete: Optional callback(period_num, total_periods, entry)
            called after each period finishes.  Used by the dashboard to
            update the progress bar and show intermediate results.
        selected_analysts: List of analyst names to use (None = all).
            Fewer analysts = faster backtest + fewer API calls.
        period_delay: Seconds to wait between periods to avoid rate limits.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Guard: step_days must be at least 1
    step_days = max(1, step_days)

    # Calculate total number of periods for progress reporting
    total_periods = 0
    t = start
    while t < end:
        t = min(t + timedelta(days=step_days), end)
        total_periods += 1
        if t == end:
            break

    portfolio = {
        "cash": initial_cash,
        "positions": {t: {"shares": 0, "avg_cost": 0} for t in tickers},
    }

    history = []
    current_start = start
    period_num = 0

    while current_start < end:
        window_end = min(current_start + timedelta(days=step_days), end)
        window_start = max(start, current_start - timedelta(days=90))  # lookback for indicators

        # Safety: ensure we always advance to prevent infinite loops
        if window_end <= current_start:
            break

        period_num += 1
        console.print(f"\n[bold]Period {period_num}/{total_periods}: {current_start.strftime('%Y-%m-%d')} to {window_end.strftime('%Y-%m-%d')}[/bold]")

        try:
            result = run_hedge_fund(
                tickers=tickers,
                start_date=window_start.strftime("%Y-%m-%d"),
                end_date=window_end.strftime("%Y-%m-%d"),
                provider=provider,
                model=model,
                selected_analysts=selected_analysts,
                initial_cash=portfolio["cash"],
                show_reasoning=False,
            )

            # Apply decisions to portfolio
            pm_decisions = result.get("analyst_signals", {}).get("Portfolio Manager", {})
            risk_data = result.get("analyst_signals", {}).get("Risk Manager", {})

            for ticker in tickers:
                decision = pm_decisions.get(ticker, {})
                action = decision.get("action", "hold")
                quantity = decision.get("quantity", 0)
                price = risk_data.get(ticker, {}).get("current_price", 0)

                if action == "buy" and quantity > 0 and price > 0:
                    cost = quantity * price
                    if cost <= portfolio["cash"]:
                        portfolio["cash"] -= cost
                        pos = portfolio["positions"][ticker]
                        total_shares = pos["shares"] + quantity
                        if total_shares > 0:
                            pos["avg_cost"] = (
                                (pos["avg_cost"] * pos["shares"] + cost) / total_shares
                            )
                        pos["shares"] = total_shares

                elif action == "sell" and quantity > 0:
                    pos = portfolio["positions"][ticker]
                    sell_qty = min(quantity, pos["shares"])
                    if sell_qty > 0 and price > 0:
                        portfolio["cash"] += sell_qty * price
                        pos["shares"] -= sell_qty

            # Calculate total value
            total_value = portfolio["cash"]
            for ticker in tickers:
                shares = portfolio["positions"][ticker]["shares"]
                price = risk_data.get(ticker, {}).get("current_price", 0)
                total_value += shares * price

            entry = {
                "date": window_end.strftime("%Y-%m-%d"),
                "cash": round(portfolio["cash"], 2),
                "total_value": round(total_value, 2),
                "return_pct": round((total_value / initial_cash - 1) * 100, 2),
                "decisions": {t: pm_decisions.get(t, {}).get("action", "hold") for t in tickers},
            }
            history.append(entry)

        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            entry = {
                "date": window_end.strftime("%Y-%m-%d"),
                "cash": portfolio["cash"],
                "total_value": portfolio["cash"],
                "return_pct": round((portfolio["cash"] / initial_cash - 1) * 100, 2),
                "decisions": {t: "error" for t in tickers},
            }
            history.append(entry)

        # Advance to next period
        current_start = window_end

        # Callback for progress updates
        if on_period_complete:
            on_period_complete(period_num, total_periods, entry)

        # If window_end == end, we're done (avoids float/datetime edge cases)
        if window_end >= end:
            break

        # Delay between periods to avoid rate-limiting (429 errors)
        if period_delay > 0:
            time.sleep(period_delay)

    return history, portfolio


def print_results(history: list[dict], portfolio: dict, tickers: list[str], initial_cash: float):
    """Print backtest results."""
    console.print(Panel("[bold]Backtest Results[/bold]", border_style="green"))

    # Timeline table
    table = Table(title="Performance Timeline")
    table.add_column("Date")
    table.add_column("Cash", justify="right")
    table.add_column("Total Value", justify="right")
    table.add_column("Return", justify="right")
    for t in tickers:
        table.add_column(t)

    for entry in history:
        ret = entry["return_pct"]
        ret_color = "green" if ret >= 0 else "red"
        table.add_row(
            entry["date"],
            f"${entry['cash']:,.0f}",
            f"${entry['total_value']:,.0f}",
            f"[{ret_color}]{ret:+.1f}%[/{ret_color}]",
            *[entry["decisions"].get(t, "-") for t in tickers],
        )

    console.print(table)

    # Final summary
    if history:
        final = history[-1]
        console.print(f"\n  Initial: ${initial_cash:,.0f}")
        console.print(f"  Final:   ${final['total_value']:,.0f}")
        console.print(f"  Return:  {final['return_pct']:+.1f}%")

    # Position summary
    console.print("\n[bold]Final Positions:[/bold]")
    for ticker in tickers:
        pos = portfolio["positions"][ticker]
        console.print(f"  {ticker}: {pos['shares']} shares (avg cost: ${pos['avg_cost']:.2f})")
    console.print(f"  Cash: ${portfolio['cash']:,.2f}")


def main():
    parser = argparse.ArgumentParser(description="InvestAgent Backtester")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--initial-cash", type=float, default=100_000)
    parser.add_argument("--step-days", type=int, default=30, help="Days between rebalance")
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers]
    history, portfolio = backtest(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        provider=args.provider,
        model=args.model,
        initial_cash=args.initial_cash,
        step_days=args.step_days,
    )
    print_results(history, portfolio, tickers, args.initial_cash)


if __name__ == "__main__":
    main()
