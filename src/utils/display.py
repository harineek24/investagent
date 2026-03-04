"""Display utilities for agent output."""

import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def show_agent_reasoning(reasoning: dict | str, agent_name: str):
    """Pretty-print an agent's reasoning."""
    if isinstance(reasoning, str):
        try:
            reasoning = json.loads(reasoning)
        except json.JSONDecodeError:
            pass

    content = json.dumps(reasoning, indent=2) if isinstance(reasoning, dict) else str(reasoning)
    console.print(Panel(content, title=f"[bold cyan]{agent_name} Reasoning[/bold cyan]", border_style="cyan"))


def show_portfolio_table(decisions: dict, portfolio: dict):
    """Display a table of portfolio decisions."""
    table = Table(title="Portfolio Decisions")
    table.add_column("Ticker", style="bold")
    table.add_column("Action", style="green")
    table.add_column("Quantity", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Reasoning")

    for ticker, decision in decisions.items():
        action = decision.get("action", "hold")
        qty = str(decision.get("quantity", 0))
        conf = f"{decision.get('confidence', 0):.0f}%"
        reason = decision.get("reasoning", "")[:60]
        table.add_row(ticker, action, qty, conf, reason)

    console.print(table)


def progress_message(agent_name: str, status: str = "running"):
    """Print a progress message."""
    if status == "running":
        console.print(f"  [yellow]>[/yellow] {agent_name} analyzing...", highlight=False)
    elif status == "done":
        console.print(f"  [green]✓[/green] {agent_name} complete", highlight=False)
