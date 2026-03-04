"""LLM provider configuration with multi-model routing.

Supports free options and intelligent routing so different agents
can use different models (fast/cheap for analysts, smart for portfolio manager).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Model routing: map agent roles to model tiers
# ---------------------------------------------------------------------------

AGENT_MODEL_TIER = {
    "Value Analyst": "fast",
    "Growth Analyst": "fast",
    "Contrarian Analyst": "fast",
    "Technical Analyst": "fast",
    "Fundamental Analyst": "fast",
    "Sentiment Analyst": "fast",
    "Risk Manager": "fast",
    "Portfolio Manager": "smart",
}

# Default models per provider per tier
PROVIDER_MODELS = {
    "groq": {
        "fast": "llama-3.1-8b-instant",
        "smart": "llama-3.3-70b-versatile",
    },
    "gemini": {
        "fast": "gemini-2.0-flash-lite",
        "smart": "gemini-2.0-flash",
    },
    "ollama": {
        "fast": "llama3.2",
        "smart": "llama3.2",
    },
    "openai": {
        "fast": "gpt-4o-mini",
        "smart": "gpt-4o",
    },
}


def get_llm(provider: str = "groq", model: str | None = None, agent_name: str | None = None):
    """Get an LLM instance with optional multi-model routing.

    If agent_name is provided and model is not explicitly set,
    automatically picks the right model tier (fast vs smart).

    Free providers:
        - "ollama": Free local models (requires Ollama installed)
        - "groq": Free tier with Llama models
        - "gemini": Free tier with Google Gemini

    Paid providers:
        - "openai": GPT-4o, GPT-4o-mini
    """
    # Auto-select model based on agent tier
    if model is None and agent_name:
        tier = AGENT_MODEL_TIER.get(agent_name, "fast")
        model = PROVIDER_MODELS.get(provider, {}).get(tier)

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model or "llama3.2",
            temperature=0,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model or "llama-3.3-70b-versatile",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
            max_retries=3,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model or "gemini-2.0-flash",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_retries=3,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_retries=3,
        )

    raise ValueError(f"Unknown provider: {provider}")
