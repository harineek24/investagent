"""InvestAgent - AI-Powered Agentic Hedge Fund Simulator

Truly agentic multi-agent system:
    - ReAct agents that choose their own tools and reason step-by-step
    - Conditional routing: debate triggered when analysts disagree
    - Multi-turn Portfolio Manager with autonomous tool use
    - Agent memory that weights signals by historical accuracy

Usage:
    python -m src.main --tickers AAPL MSFT NVDA
    python -m src.main --tickers TSLA --provider ollama --show-reasoning
"""

import argparse
import json
import uuid
from datetime import datetime, timedelta

from langgraph.graph import StateGraph, END
from rich.console import Console
from rich.panel import Panel

from src.graph.state import AgentState
from src.agents import (
    ANALYST_AGENTS,
    risk_manager,
    portfolio_manager,
    check_agreement,
    debate_agents,
)
from src.agents.debate import should_debate
from src.memory import save_run, save_decision, update_run_result
from src.scoring import score_pending_decisions

console = Console()


def build_workflow(selected_analysts: list[str] | None = None):
    """Build the agentic LangGraph workflow with conditional routing.

    Flow:
        START → [Analysts in parallel] → Agreement Check → {
            if agree:    → Risk Manager → Portfolio Manager → END
            if disagree: → Debate → Risk Manager → Portfolio Manager → END
        }

    This is NOT a fixed pipeline. The Debate node only runs when
    analysts produce conflicting signals.
    """
    graph = StateGraph(AgentState)

    # Determine which analysts to use
    available = ANALYST_AGENTS
    if selected_analysts:
        available = {k: v for k, v in ANALYST_AGENTS.items() if k in selected_analysts}
    if not available:
        available = ANALYST_AGENTS

    analyst_names = list(available.keys())

    # --- Add nodes ---
    for name, func in available.items():
        graph.add_node(name, func)

    graph.add_node("Agreement Check", check_agreement)
    graph.add_node("Debate", debate_agents)
    graph.add_node("Risk Manager", risk_manager)
    graph.add_node("Portfolio Manager", portfolio_manager)

    # --- Wire: START → all analysts in parallel ---
    graph.set_entry_point(analyst_names[0])
    for name in analyst_names[1:]:
        graph.add_edge("__start__", name)

    # --- Wire: all analysts → Agreement Check ---
    for name in analyst_names:
        graph.add_edge(name, "Agreement Check")

    # --- Wire: Agreement Check → conditional branch ---
    graph.add_conditional_edges(
        "Agreement Check",
        should_debate,
        {
            "Debate": "Debate",
            "Risk Manager": "Risk Manager",
        },
    )

    # --- Wire: Debate → Risk Manager ---
    graph.add_edge("Debate", "Risk Manager")

    # --- Wire: Risk Manager → Portfolio Manager → END ---
    graph.add_edge("Risk Manager", "Portfolio Manager")
    graph.add_edge("Portfolio Manager", END)

    return graph.compile()


def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    provider: str = "groq",
    model: str | None = None,
    selected_analysts: list[str] | None = None,
    initial_cash: float = 100_000,
    existing_positions: dict | None = None,
    show_reasoning: bool = False,
):
    """Run the agentic hedge fund simulation."""
    run_id = str(uuid.uuid4())[:8]
    workflow = build_workflow(selected_analysts)

    # Score any past decisions that are now old enough to evaluate, so the
    # debate/PM track-record weighting has real accuracy data to work with.
    score_pending_decisions()

    # Save run to memory
    save_run(run_id, tickers, provider, initial_cash)

    # Build initial portfolio state, seeding any positions the user already holds
    existing_positions = existing_positions or {}
    portfolio = {
        "cash": initial_cash,
        "positions": {
            ticker: dict(existing_positions.get(ticker, {"shares": 0, "avg_cost": 0}))
            for ticker in tickers
        },
        "_llm_provider": provider,
        "_llm_model": model,
    }

    initial_state = {
        "messages": [],
        "tickers": tickers,
        "portfolio": portfolio,
        "start_date": start_date,
        "end_date": end_date,
        "analyst_signals": {},
        "show_reasoning": show_reasoning,
        "debate_required": False,
        "debate_summary": {},
        "agreement_score": 1.0,
    }

    result = workflow.invoke(initial_state)

    # Save all agent decisions to memory
    analyst_signals = result.get("analyst_signals", {})
    risk_data = analyst_signals.get("Risk Manager", {})

    for agent_name, agent_signals in analyst_signals.items():
        if agent_name in ("Risk Manager",):
            continue
        for ticker, sig in agent_signals.items():
            price = risk_data.get(ticker, {}).get("current_price", 0)
            save_decision(
                run_id=run_id,
                ticker=ticker,
                agent=agent_name,
                signal=sig.get("signal", sig.get("action", "")),
                confidence=sig.get("confidence", 0),
                reasoning=sig.get("reasoning", {}),
                action=sig.get("action", ""),
                quantity=sig.get("quantity", 0),
                price=price,
            )

    result["_run_id"] = run_id
    return result


def select_analysts():
    """Let user pick which analysts to run."""
    names = list(ANALYST_AGENTS.keys())
    console.print("\n[bold]Available Analysts:[/bold]")
    for i, name in enumerate(names, 1):
        console.print(f"  {i}. {name}")
    console.print(f"  {len(names) + 1}. All analysts")

    choice = input(f"\nSelect analysts (comma-separated numbers, or {len(names) + 1} for all): ").strip()
    if not choice or choice == str(len(names) + 1):
        return None  # all

    try:
        indices = [int(x.strip()) - 1 for x in choice.split(",")]
        return [names[i] for i in indices if 0 <= i < len(names)]
    except (ValueError, IndexError):
        console.print("[yellow]Invalid selection, using all analysts.[/yellow]")
        return None


def select_provider():
    """Let user pick LLM provider."""
    providers = [
        ("groq", "Groq (FREE tier - Llama 3.3 70B)"),
        ("gemini", "Google Gemini (FREE tier)"),
        ("ollama", "Ollama (FREE local - requires install)"),
        ("openai", "OpenAI (paid - GPT-4o-mini)"),
    ]
    console.print("\n[bold]LLM Providers:[/bold]")
    for i, (key, desc) in enumerate(providers, 1):
        console.print(f"  {i}. {desc}")

    choice = input("\nSelect provider (1-4, default=1): ").strip()
    try:
        idx = int(choice) - 1 if choice else 0
        return providers[idx][0]
    except (ValueError, IndexError):
        return "groq"


def main():
    parser = argparse.ArgumentParser(description="InvestAgent - Agentic AI Hedge Fund Simulator")
    parser.add_argument("--tickers", nargs="+", required=True, help="Stock tickers to analyze")
    parser.add_argument("--initial-cash", type=float, default=100_000, help="Starting cash (default: $100,000)")
    parser.add_argument(
        "--positions", type=str, default=None,
        help="Existing positions as TICKER:SHARES:AVG_COST, comma-separated "
             "(e.g. 'AAPL:50:180.00,MSFT:10:400.00')",
    )
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--provider", type=str, default=None, help="LLM provider: groq, gemini, ollama, openai")
    parser.add_argument("--model", type=str, default=None, help="Specific model name")
    parser.add_argument("--show-reasoning", action="store_true", help="Show detailed agent reasoning")
    parser.add_argument("--all-analysts", action="store_true", help="Use all analysts without prompting")
    args = parser.parse_args()

    # Defaults
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    start_date = args.start_date or (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    existing_positions = {}
    if args.positions:
        for entry in args.positions.split(","):
            ticker, shares, avg_cost = entry.strip().split(":")
            existing_positions[ticker.strip().upper()] = {
                "shares": int(shares), "avg_cost": float(avg_cost),
            }

    console.print(Panel(
        "[bold green]InvestAgent[/bold green] - Agentic AI Hedge Fund Simulator\n"
        "[dim]ReAct agents | Conditional routing | Multi-turn PM | Agent memory[/dim]\n"
        "[dim]Educational purposes only. Not financial advice.[/dim]",
        border_style="green",
    ))

    console.print(f"\n  Tickers: {', '.join(args.tickers)}")
    console.print(f"  Period:  {start_date} to {end_date}")
    console.print(f"  Cash:    ${args.initial_cash:,.0f}")

    # Interactive selection if not using flags
    provider = args.provider or select_provider()
    selected = None if args.all_analysts else select_analysts()

    console.print(f"\n  Provider: {provider}")
    console.print(f"  Analysts: {', '.join(selected) if selected else 'All'}")
    console.print()

    result = run_hedge_fund(
        tickers=[t.upper() for t in args.tickers],
        start_date=start_date,
        end_date=end_date,
        provider=provider,
        model=args.model,
        selected_analysts=selected,
        initial_cash=args.initial_cash,
        existing_positions=existing_positions,
        show_reasoning=args.show_reasoning,
    )

    # Print workflow path taken
    agreement = result.get("agreement_score", 1.0)
    debated = result.get("debate_required", False)
    path = "Analysts → Agreement Check → "
    path += "DEBATE → " if debated else ""
    path += "Risk Manager → Portfolio Manager"
    console.print(f"\n  [dim]Workflow: {path}[/dim]")
    console.print(f"  [dim]Agreement: {agreement:.0%}[/dim]")

    # Print final decisions
    pm_signals = result.get("analyst_signals", {}).get("Portfolio Manager", {})
    if pm_signals:
        console.print("\n[bold]Final Decisions:[/bold]")
        for ticker, decision in pm_signals.items():
            action = decision.get("action", "hold").upper()
            qty = decision.get("quantity", 0)
            conf = decision.get("confidence", 0)
            reason = decision.get("reasoning", "")
            color = {"BUY": "green", "SELL": "red"}.get(action, "yellow")
            console.print(f"  [{color}]{action}[/{color}] {ticker}: {qty} shares "
                          f"(confidence: {conf}%) - {reason}")

    console.print(f"\n  [dim]Run saved to memory (id: {result.get('_run_id', 'n/a')})[/dim]")


if __name__ == "__main__":
    main()
