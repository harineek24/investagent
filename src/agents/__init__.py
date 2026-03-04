from src.agents.react_agent import create_react_agent
from src.agents.risk_manager import risk_manager
from src.agents.portfolio_manager import portfolio_manager
from src.agents.debate import check_agreement, debate_agents

# Create ReAct analyst agents (tool-using, reasoning agents)
value_agent = create_react_agent("Value Analyst")
growth_agent = create_react_agent("Growth Analyst")
contrarian_agent = create_react_agent("Contrarian Analyst")
technical_agent = create_react_agent("Technical Analyst")
fundamental_agent = create_react_agent("Fundamental Analyst")
sentiment_agent = create_react_agent("Sentiment Analyst")

ANALYST_AGENTS = {
    "Value Analyst": value_agent,
    "Growth Analyst": growth_agent,
    "Contrarian Analyst": contrarian_agent,
    "Technical Analyst": technical_agent,
    "Fundamental Analyst": fundamental_agent,
    "Sentiment Analyst": sentiment_agent,
}

__all__ = [
    "value_agent",
    "growth_agent",
    "contrarian_agent",
    "technical_agent",
    "fundamental_agent",
    "sentiment_agent",
    "risk_manager",
    "portfolio_manager",
    "check_agreement",
    "debate_agents",
    "ANALYST_AGENTS",
]
