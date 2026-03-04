"""Debate & Agreement Module — conditional routing for the agentic workflow.

When analysts disagree beyond a threshold, a debate is triggered:
1. check_agreement: measures consensus, sets debate_required flag
2. debate_agents: LLM-powered synthesis of conflicting signals

This is what makes the workflow truly agentic:
- The graph BRANCHES based on agent outputs (not a fixed pipeline)
- Agents' reasoning is challenged and weighed
- A synthesis step produces a balanced view before the Portfolio Manager decides
"""

import json
from collections import Counter
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState
from src.utils.llm import get_llm
from src.utils.display import progress_message, console
from src.memory import get_all_agent_accuracies

# If agreement score is below this, trigger a debate
AGREEMENT_THRESHOLD = 0.6


def check_agreement(state: AgentState) -> dict:
    """Measure how much analysts agree. Sets debate_required flag.

    Agreement score:
        1.0 = all analysts agree (same signal for every ticker)
        0.0 = max disagreement (even split bullish/bearish)
    """
    progress_message("Agreement Check", "running")
    analyst_signals = state.get("analyst_signals", {})
    tickers = state["tickers"]

    if not analyst_signals:
        progress_message("Agreement Check", "done")
        return {"agreement_score": 1.0, "debate_required": False}

    # For each ticker, count signal distribution
    per_ticker_scores = []
    for ticker in tickers:
        signals = []
        for agent_name, agent_signals in analyst_signals.items():
            if agent_name in ("Risk Manager", "Portfolio Manager"):
                continue
            sig = agent_signals.get(ticker, {})
            signal = sig.get("signal", "neutral")
            signals.append(signal)

        if not signals:
            per_ticker_scores.append(1.0)
            continue

        # Agreement = fraction of agents on the majority signal
        counts = Counter(signals)
        majority_count = counts.most_common(1)[0][1]
        per_ticker_scores.append(majority_count / len(signals))

    agreement = sum(per_ticker_scores) / len(per_ticker_scores) if per_ticker_scores else 1.0
    debate_needed = agreement < AGREEMENT_THRESHOLD

    if debate_needed:
        console.print(
            f"  [yellow]![/yellow] Agreement: {agreement:.0%} — debate triggered "
            f"(threshold: {AGREEMENT_THRESHOLD:.0%})"
        )
    else:
        console.print(f"  [green]✓[/green] Agreement: {agreement:.0%} — consensus reached")

    progress_message("Agreement Check", "done")

    return {
        "agreement_score": agreement,
        "debate_required": debate_needed,
    }


DEBATE_PROMPT = """You are a senior investment committee moderator.

Multiple analysts have given CONFLICTING signals for {ticker}. Your job:
1. Summarize each analyst's position
2. Identify the key point of disagreement
3. Weigh the arguments considering each analyst's track record
4. Produce a synthesis — which side has the stronger case?

## Analyst Signals
{signals_detail}

## Agent Track Records (historical accuracy)
{track_records}

## Your Task
Produce a balanced synthesis. Consider:
- Which analysts have the best track record?
- What data supports each side?
- Are there any red flags one side is missing?

Respond with ONLY valid JSON:
{{
    "synthesis": "<your analysis of the debate>",
    "leaning": "bullish|neutral|bearish",
    "confidence_adjustment": <-20 to +20>,
    "key_disagreement": "<what the analysts disagree about>",
    "strongest_argument": "<which analyst made the best case and why>"
}}"""


def debate_agents(state: AgentState) -> dict:
    """Run a debate when analysts disagree.

    The LLM acts as a moderator, weighing each analyst's argument
    and considering their historical track record.
    """
    progress_message("Debate", "running")
    analyst_signals = state.get("analyst_signals", {})
    tickers = state["tickers"]
    portfolio = state["portfolio"]

    llm_provider = portfolio.get("_llm_provider", "groq")
    llm_model = portfolio.get("_llm_model")
    llm = get_llm(llm_provider, llm_model, agent_name="Portfolio Manager")  # use smart model

    # Get track records
    accuracies = get_all_agent_accuracies()
    track_records = "No historical data yet (first run)."
    if accuracies:
        lines = []
        for a in accuracies:
            if a["accuracy"] is not None:
                lines.append(f"- {a['agent']}: {a['accuracy']}% accurate ({a['total_signals']} signals)")
        if lines:
            track_records = "\n".join(lines)

    debate_results = {}

    for ticker in tickers:
        # Build detailed signal summary
        signals_detail_parts = []
        for agent_name, agent_signals in analyst_signals.items():
            if agent_name in ("Risk Manager", "Portfolio Manager"):
                continue
            sig = agent_signals.get(ticker, {})
            signal = sig.get("signal", "neutral")
            conf = sig.get("confidence", 0)
            reasoning = sig.get("reasoning", "No reasoning provided")
            signals_detail_parts.append(
                f"### {agent_name}\n"
                f"Signal: {signal} (confidence: {conf}%)\n"
                f"Reasoning: {reasoning}"
            )

        signals_detail = "\n\n".join(signals_detail_parts) if signals_detail_parts else "No signals."

        try:
            prompt = DEBATE_PROMPT.format(
                ticker=ticker,
                signals_detail=signals_detail,
                track_records=track_records,
            )
            response = llm.invoke(prompt)
            content = response.content.strip()

            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            debate_results[ticker] = json.loads(content)

        except Exception as e:
            debate_results[ticker] = {
                "synthesis": f"Debate error: {e}",
                "leaning": "neutral",
                "confidence_adjustment": 0,
                "key_disagreement": "unknown",
                "strongest_argument": "unknown",
            }

    console.print(f"  [cyan]>[/cyan] Debate concluded for {', '.join(tickers)}")
    for ticker, result in debate_results.items():
        leaning = result.get("leaning", "neutral")
        color = {"bullish": "green", "bearish": "red"}.get(leaning, "yellow")
        console.print(
            f"    [{color}]{leaning}[/{color}] {ticker}: {result.get('key_disagreement', '')[:80]}"
        )

    progress_message("Debate", "done")

    return {
        "messages": [HumanMessage(content=json.dumps(debate_results), name="Debate")],
        "debate_summary": debate_results,
    }


def should_debate(state: AgentState) -> str:
    """Routing function for LangGraph conditional edge.

    Returns the name of the next node:
        "Debate" if analysts disagree
        "Risk Manager" if they agree
    """
    if state.get("debate_required", False):
        return "Debate"
    return "Risk Manager"
