from src.agents.react_agent import (
    _parse_tool_call,
    _parse_final_answer,
    _data_fallback,
    _run_react_loop,
    _build_tool_descriptions,
    AGENT_RECOMMENDED_TOOLS,
)


def test_parse_tool_call_valid():
    name, args = _parse_tool_call('get_financial_metrics({"ticker": "AAPL"})')
    assert name == "get_financial_metrics"
    assert args == {"ticker": "AAPL"}


def test_parse_tool_call_invalid_json():
    name, args = _parse_tool_call("get_prices(not json)")
    assert name is None
    assert args == {}


def test_parse_tool_call_no_parens():
    name, args = _parse_tool_call("get_prices")
    assert name is None
    assert args == {}


def test_parse_final_answer_plain_json():
    content = 'FINAL_ANSWER: {"signal": "bullish", "confidence": 80, "reasoning": "strong"}'
    result = _parse_final_answer(content, "Value Analyst")
    assert result["signal"] == "bullish"
    assert result["confidence"] == 80
    assert result["agent"] == "Value Analyst"


def test_parse_final_answer_markdown_block():
    content = (
        'FINAL_ANSWER: ```json\n{"signal": "bearish", "confidence": 60, "reasoning": "weak"}\n```'
    )
    result = _parse_final_answer(content, "Growth Analyst")
    assert result["signal"] == "bearish"
    assert result["agent"] == "Growth Analyst"


def test_parse_final_answer_malformed_falls_back_to_neutral():
    content = "FINAL_ANSWER: not valid json at all"
    result = _parse_final_answer(content, "Technical Analyst")
    assert result["signal"] == "neutral"
    assert result["confidence"] == 20
    assert result["agent"] == "Technical Analyst"


def test_data_fallback_bullish(monkeypatch):
    import pandas as pd
    import src.tools.api as api

    def fake_get_prices(ticker, start_date, end_date):
        return pd.DataFrame({"Close": [100.0, 110.0]})

    # _data_fallback imports api functions inside the function body, so patch the source module
    monkeypatch.setattr(api, "get_financial_metrics", lambda ticker: {
        "pe_ratio": 10, "return_on_equity": 0.20, "revenue_growth": 0.15,
    })
    monkeypatch.setattr(api, "get_prices", fake_get_prices)

    result = _data_fallback("AAPL", "2024-01-01", "2024-02-01")
    assert result["signal"] == "bullish"


def test_data_fallback_handles_exceptions_gracefully(monkeypatch):
    import src.tools.api as api

    def boom(ticker):
        raise RuntimeError("network down")

    monkeypatch.setattr(api, "get_financial_metrics", boom)

    result = _data_fallback("AAPL", "2024-01-01", "2024-02-01")
    assert result["signal"] == "neutral"
    assert result["confidence"] == 20


def test_tool_descriptions_flag_recommended_tools_for_agent():
    desc = _build_tool_descriptions("Technical Analyst")
    assert "get_prices" in desc
    assert desc.index("get_prices") < desc.index("get_financial_metrics")
    assert "[RECOMMENDED for your philosophy]" in desc


def test_tool_descriptions_unknown_agent_has_no_recommendations():
    desc = _build_tool_descriptions("Unknown Analyst")
    assert "[RECOMMENDED for your philosophy]" not in desc


def test_every_agent_recommendation_list_only_uses_real_tool_names():
    from src.agents.react_agent import _TOOL_DESCRIPTIONS

    for agent_name, tools in AGENT_RECOMMENDED_TOOLS.items():
        for tool in tools:
            assert tool in _TOOL_DESCRIPTIONS, f"{agent_name} recommends unknown tool {tool}"


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    """Returns canned responses in sequence on each .invoke() call."""

    def __init__(self, responses):
        self._responses = list(responses)

    def invoke(self, prompt):
        return _FakeResponse(self._responses.pop(0))


def test_run_react_loop_immediate_final_answer():
    llm = _FakeLLM([
        'FINAL_ANSWER: {"signal": "bullish", "confidence": 75, "reasoning": "looks good"}',
    ])
    result = _run_react_loop(
        llm=llm, agent_name="Value Analyst", philosophy="test philosophy",
        ticker="AAPL", start_date="2024-01-01", end_date="2024-02-01",
    )
    assert result["signal"] == "bullish"
    assert result["confidence"] == 75
    assert result["agent"] == "Value Analyst"


def test_run_react_loop_tool_call_then_final_answer(monkeypatch):
    import src.agents.react_agent as react_agent

    monkeypatch.setattr(react_agent, "_execute_tool", lambda name, args: "fake tool result")

    llm = _FakeLLM([
        'TOOL_CALL: get_financial_metrics({"ticker": "AAPL"})',
        'FINAL_ANSWER: {"signal": "neutral", "confidence": 50, "reasoning": "mixed signals"}',
    ])
    result = _run_react_loop(
        llm=llm, agent_name="Fundamental Analyst", philosophy="test philosophy",
        ticker="AAPL", start_date="2024-01-01", end_date="2024-02-01",
    )
    assert result["signal"] == "neutral"
    assert result["agent"] == "Fundamental Analyst"


def test_run_react_loop_exhausts_iterations_falls_back_to_neutral(monkeypatch):
    import src.agents.react_agent as react_agent

    monkeypatch.setattr(react_agent, "_execute_tool", lambda name, args: "fake tool result")

    # Agent keeps calling tools forever, never gives FINAL_ANSWER
    responses = [
        f'TOOL_CALL: get_financial_metrics({{"ticker": "AAPL"}})'
        for _ in range(react_agent.MAX_ITERATIONS + 2)
    ]
    llm = _FakeLLM(responses)
    result = _run_react_loop(
        llm=llm, agent_name="Sentiment Analyst", philosophy="test philosophy",
        ticker="AAPL", start_date="2024-01-01", end_date="2024-02-01",
    )
    assert result["signal"] == "neutral"
    assert result["confidence"] == 30
    assert result["agent"] == "Sentiment Analyst"
