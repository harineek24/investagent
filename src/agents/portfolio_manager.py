"""Portfolio Manager Agent — Multi-turn agentic decision maker.

Unlike the old one-shot prompt approach, this PM:
1. Reviews analyst signals and debate summary
2. Can call tools to get more data (autonomous tool use)
3. Reasons in multiple steps before deciding (ReAct loop)
4. Checks agent memory and weights signals by track record
5. Self-validates its decision against risk constraints

This is a true agent — it decides what it needs, fetches it, and reasons.
"""

import json
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState
from src.utils.display import show_agent_reasoning, show_portfolio_table, progress_message, console
from src.utils.llm import get_llm
from src.memory import get_past_decisions, get_all_agent_accuracies
from src.tools.sec_edgar import get_company_facts

def _majority_vote_fallback(ticker: str, analyst_signals: dict, risk_data: dict, portfolio: dict) -> dict:
    """Rule-based fallback when PM can't reach the LLM.

    Uses a simple majority vote of analyst signals weighted by confidence.
    """
    votes = {"bullish": 0, "bearish": 0, "neutral": 0}
    total_conf = 0
    valid_signals = 0

    for agent_name, agent_sigs in analyst_signals.items():
        if agent_name in ("Risk Manager", "Portfolio Manager"):
            continue
        sig = agent_sigs.get(ticker, {})
        signal = sig.get("signal", "neutral")
        conf = sig.get("confidence", 0)
        if conf > 0:
            votes[signal] = votes.get(signal, 0) + conf
            total_conf += conf
            valid_signals += 1

    if valid_signals == 0:
        return {
            "action": "hold", "quantity": 0, "confidence": 10,
            "reasoning": "No valid analyst signals available (rate-limited).",
        }

    majority = max(votes, key=votes.get)
    avg_conf = total_conf // max(valid_signals, 1)

    risk = risk_data.get(ticker, {})
    max_shares = risk.get("max_shares", 0)
    existing = portfolio.get("positions", {}).get(ticker, {}).get("shares", 0)

    if majority == "bullish" and avg_conf >= 50:
        qty = max(1, max_shares // 3)  # conservative: 1/3 of limit
        return {
            "action": "buy", "quantity": qty, "confidence": avg_conf,
            "reasoning": f"Majority-vote fallback: {valid_signals} analysts lean bullish (avg conf {avg_conf}%).",
        }
    elif majority == "bearish" and avg_conf >= 50 and existing > 0:
        qty = max(1, existing // 3)
        return {
            "action": "sell", "quantity": qty, "confidence": avg_conf,
            "reasoning": f"Majority-vote fallback: {valid_signals} analysts lean bearish (avg conf {avg_conf}%).",
        }

    return {
        "action": "hold", "quantity": 0, "confidence": max(avg_conf, 20),
        "reasoning": f"Majority-vote fallback: no strong consensus ({majority}, avg conf {avg_conf}%).",
    }


AGENT_NAME = "Portfolio Manager"
MAX_PM_ITERATIONS = 3  # PM can make up to 3 tool calls per ticker


PM_SYSTEM_PROMPT = """You are the Portfolio Manager — the final decision maker for an AI hedge fund.

You have analyst signals, risk constraints, and access to tools. Your job:
1. Review the analyst signals and any debate summary
2. Optionally call tools if you need more information
3. Make a final buy/sell/hold decision

## Analyst Signals for {ticker}
{analyst_summary}

## Debate Summary (if analysts disagreed)
{debate_summary}

## Agent Track Records
{track_records}

## Past Decisions for {ticker}
{past_decisions}

## Risk Constraints
- Current price: ${current_price}
- Max shares buyable: {max_shares}
- Available position limit: ${remaining_limit}
- Current position value: ${current_position}

## Portfolio
- Cash: ${cash}
- Existing shares of {ticker}: {existing_shares}

## Available Tools
Call these if you need more data before deciding:
- TOOL_CALL: get_financial_metrics({{"ticker": "{ticker}"}})
- TOOL_CALL: get_sec_financial_facts({{"ticker": "{ticker}"}})
- TOOL_CALL: get_company_news({{"ticker": "{ticker}"}})

## Decision Rules
1. Weight analyst signals by their track record accuracy
2. If debate occurred, seriously consider the synthesis
3. Only buy if you have strong conviction AND available limit
4. Only sell if you hold shares AND have bearish conviction
5. Never exceed max_shares limit
6. Be conservative — when in doubt, hold
7. Explain your reasoning chain, not just the conclusion

## Response Format
When calling a tool:
TOOL_CALL: tool_name({{"key": "value"}})

When ready to decide:
FINAL_DECISION: {{"action": "buy|sell|hold", "quantity": <integer>, "confidence": <0-100>, "reasoning": "<your multi-step reasoning>"}}"""


def _format_track_records() -> str:
    """Format agent accuracy for the prompt."""
    accuracies = get_all_agent_accuracies()
    if not accuracies:
        return "No historical track record yet."
    lines = []
    for a in accuracies:
        if a["accuracy"] is not None:
            lines.append(f"- {a['agent']}: {a['accuracy']}% accurate ({a['total_signals']} signals)")
    return "\n".join(lines) if lines else "No track record data yet."


def _format_past_decisions(ticker: str) -> str:
    past = get_past_decisions(ticker, limit=5)
    if not past:
        return "No past decisions for this ticker."
    lines = []
    for d in past:
        outcome = f" -> {d['outcome']}" if d.get("outcome") else ""
        lines.append(f"- [{d['timestamp'][:10]}] {d['agent']}: {d['signal']} "
                      f"(conf: {d['confidence']}%){outcome}")
    return "\n".join(lines)


def _execute_pm_tool(tool_name: str, args: dict) -> str:
    """Execute a tool call for the PM."""
    from src.tools.api import get_financial_metrics, get_company_news
    from src.tools.sec_edgar import get_company_facts

    tool_map = {
        "get_financial_metrics": lambda a: json.dumps(get_financial_metrics(a["ticker"]), default=str),
        "get_sec_financial_facts": lambda a: json.dumps(get_company_facts(a["ticker"]), default=str),
        "get_company_news": lambda a: json.dumps(get_company_news(a["ticker"]), default=str),
    }
    func = tool_map.get(tool_name)
    if not func:
        return f"Unknown tool: {tool_name}"
    try:
        return func(args)
    except Exception as e:
        return f"Tool error: {e}"


def _parse_tool_call(tool_line: str) -> tuple[str | None, dict]:
    try:
        paren_idx = tool_line.index("(")
        tool_name = tool_line[:paren_idx].strip()
        args_str = tool_line[paren_idx + 1:].rstrip(")")
        args = json.loads(args_str)
        return tool_name, args
    except (ValueError, json.JSONDecodeError):
        return None, {}


def portfolio_manager(state: AgentState):
    """Multi-turn agentic portfolio manager."""
    progress_message(AGENT_NAME, "running")
    tickers = state["tickers"]
    portfolio = state["portfolio"]
    analyst_signals = state.get("analyst_signals", {})
    debate_summary = state.get("debate_summary", {})
    decisions = {}

    risk_data = analyst_signals.get("Risk Manager", {})

    llm_provider = portfolio.get("_llm_provider", "groq")
    llm_model = portfolio.get("_llm_model")
    llm = get_llm(llm_provider, llm_model, agent_name=AGENT_NAME)

    for ticker in tickers:
        try:
            # Build analyst summary with confidence-weighted info
            analyst_summary_parts = []
            for agent_name, agent_sigs in analyst_signals.items():
                if agent_name in ("Risk Manager", "Portfolio Manager"):
                    continue
                if ticker in agent_sigs:
                    sig = agent_sigs[ticker]
                    analyst_summary_parts.append(
                        f"- {agent_name}: {sig.get('signal', 'neutral')} "
                        f"(confidence: {sig.get('confidence', 0)}%) "
                        f"— {sig.get('reasoning', 'no detail')}"
                    )

            analyst_summary = "\n".join(analyst_summary_parts) if analyst_summary_parts else "No analyst data."

            # Debate context
            ticker_debate = debate_summary.get(ticker, {})
            debate_text = "No debate (analysts agreed)."
            if ticker_debate:
                debate_text = (
                    f"Leaning: {ticker_debate.get('leaning', 'neutral')}\n"
                    f"Key disagreement: {ticker_debate.get('key_disagreement', 'unknown')}\n"
                    f"Strongest argument: {ticker_debate.get('strongest_argument', 'unknown')}\n"
                    f"Synthesis: {ticker_debate.get('synthesis', 'none')}"
                )

            # Risk constraints
            risk = risk_data.get(ticker, {})
            current_price = risk.get("current_price", 0)
            max_shares = risk.get("max_shares", 0)
            remaining_limit = risk.get("remaining_limit_usd", 0)
            current_position = risk.get("current_position_value", 0)
            existing_shares = portfolio.get("positions", {}).get(ticker, {}).get("shares", 0)
            cash = portfolio.get("cash", 0)

            system_prompt = PM_SYSTEM_PROMPT.format(
                ticker=ticker,
                analyst_summary=analyst_summary,
                debate_summary=debate_text,
                track_records=_format_track_records(),
                past_decisions=_format_past_decisions(ticker),
                current_price=current_price,
                max_shares=max_shares,
                remaining_limit=round(remaining_limit, 2),
                current_position=round(current_position, 2),
                cash=round(cash, 2),
                existing_shares=existing_shares,
            )

            # Multi-turn reasoning loop
            conversation = [system_prompt]
            tool_calls = 0

            for _ in range(MAX_PM_ITERATIONS + 2):
                response = llm.invoke("\n\n".join(conversation))
                content = response.content.strip()
                conversation.append(f"PM: {content}")

                # Tool call
                if "TOOL_CALL:" in content and tool_calls < MAX_PM_ITERATIONS:
                    tool_line = content.split("TOOL_CALL:")[-1].strip()
                    tool_name, tool_args = _parse_tool_call(tool_line)
                    if tool_name:
                        tool_args.setdefault("ticker", ticker)
                        result = _execute_pm_tool(tool_name, tool_args)
                        tool_calls += 1
                        conversation.append(f"TOOL_RESULT ({tool_name}):\n{result}")
                        continue

                # Final decision
                if "FINAL_DECISION:" in content:
                    json_str = content.split("FINAL_DECISION:")[-1].strip()
                    if "```" in json_str:
                        json_str = json_str.split("```")[1]
                        if json_str.startswith("json"):
                            json_str = json_str[4:]
                        json_str = json_str.strip()

                    decision = json.loads(json_str)

                    # Enforce hard limits
                    if decision["action"] == "buy":
                        decision["quantity"] = min(decision["quantity"], max_shares)
                    elif decision["action"] == "sell":
                        decision["quantity"] = min(decision["quantity"], max(0, existing_shares))

                    decisions[ticker] = decision
                    break
            else:
                # Fallback if loop didn't produce FINAL_DECISION
                decisions[ticker] = {
                    "action": "hold",
                    "quantity": 0,
                    "confidence": 20,
                    "reasoning": "PM could not reach a decision within iteration limit.",
                }

        except Exception as e:
            # Use majority-vote fallback instead of 0 confidence
            console.print(f"  [red]PM LLM error for {ticker}:[/red] {e}")
            fallback = _majority_vote_fallback(ticker, analyst_signals, risk_data, portfolio)
            fallback["reasoning"] = f"Used majority-vote fallback; {fallback['reasoning']}"
            decisions[ticker] = fallback

    if state.get("show_reasoning"):
        show_agent_reasoning(decisions, AGENT_NAME)

    show_portfolio_table(decisions, portfolio)
    progress_message(AGENT_NAME, "done")

    return {
        "messages": [HumanMessage(content=json.dumps(decisions), name=AGENT_NAME)],
        "analyst_signals": {AGENT_NAME: decisions},
    }
