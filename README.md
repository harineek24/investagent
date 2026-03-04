# InvestAgent - Agentic AI Hedge Fund Simulator

A **truly agentic**, 100% free-to-run multi-agent hedge fund simulator. Agents autonomously choose tools, reason step-by-step, debate when they disagree, and learn from past decisions.

> **Disclaimer:** Educational and research purposes only. Not financial advice.

---

## What Makes This Agentic (Not Just GenAI)

| Agentic Property | Implementation | Before (GenAI wrapper) |
|---|---|---|
| **Autonomous tool use** | ReAct agents decide WHAT data to fetch | Hardcoded `get_prices()` calls |
| **Multi-step reasoning** | Think → Act → Observe → Repeat loop | Single-pass score calculation |
| **Conditional routing** | Debate triggered when agents disagree | Fixed pipeline |
| **Inter-agent debate** | LLM moderator weighs conflicting signals | No agent interaction |
| **Multi-turn PM** | Portfolio Manager calls tools, reasons in loops | One-shot LLM prompt |
| **Adaptive memory** | Signals weighted by historical agent accuracy | Memory was just logging |

### The Workflow (Dynamic, Not Fixed)

```
                    ┌──────────────────────────────────────┐
                    │          User Input (tickers)         │
                    └──────────────┬───────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
              ▼                    ▼                     ▼
     ┌─────────────┐    ┌──────────────┐    ┌────────────────┐
     │    Value     │    │   Technical  │    │   Sentiment    │
     │  (ReAct ×4) │    │  (ReAct ×4)  │    │   (ReAct ×4)   │  ... ×6 agents
     │  Think→Act  │    │  Think→Act   │    │   Think→Act    │
     │  →Observe   │    │  →Observe    │    │   →Observe     │
     └──────┬──────┘    └──────┬───────┘    └───────┬────────┘
            │                  │                     │
            └──────────────────┼─────────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │  Agreement Check    │
                    │  (measure consensus)│
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
               agree ≥60%           disagree <60%
                    │                     │
                    │          ┌──────────▼──────────┐
                    │          │      DEBATE          │
                    │          │  LLM moderator       │
                    │          │  weighs arguments    │
                    │          │  + track records     │
                    │          └──────────┬──────────┘
                    │                     │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │   Risk Manager      │
                    │   (position limits) │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  Portfolio Manager   │
                    │  (multi-turn ReAct) │
                    │  Can call tools     │
                    │  Weighs debate      │
                    │  Checks memory      │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  BUY / SELL / HOLD  │
                    │  → Saved to memory  │
                    └─────────────────────┘
```

**Key difference:** The graph BRANCHES based on agent outputs. Debate only runs when needed. The PM can request additional data mid-decision.

---

## Quick Start

```bash
git clone <this-repo>
cd investagent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add GROQ_API_KEY (free)

# CLI
python -m src.main --tickers AAPL MSFT NVDA --provider groq --all-analysts

# Web dashboard
streamlit run src/dashboard.py

# MCP server
python -m src.mcp_server
```

---

## LLM Providers (All Free Options)

| Provider | Setup | Cost |
|----------|-------|------|
| **Groq** | Get key at [console.groq.com](https://console.groq.com) | **Free tier** |
| **Gemini** | Get key at [aistudio.google.com](https://aistudio.google.com/apikey) | **Free tier** |
| **Ollama** | Install from [ollama.com](https://ollama.com), `ollama pull llama3.2` | **Free** (local) |
| OpenAI | Set `OPENAI_API_KEY` | Paid |

---

## Project Structure

```
investagent/
├── requirements.txt
├── .env.example
├── .streamlit/
│   ├── config.toml              # Streamlit Cloud config
│   └── secrets.toml.example     # Secrets format
└── src/
    ├── main.py                  # CLI + agentic workflow with conditional routing
    ├── dashboard.py             # Streamlit web dashboard
    ├── mcp_server.py            # MCP server for Claude/Cursor
    ├── backtester.py            # Historical backtesting
    ├── memory.py                # SQLite agent memory
    ├── graph/
    │   └── state.py             # Shared state (includes debate fields)
    ├── tools/
    │   ├── api.py               # yfinance stock data (free)
    │   └── sec_edgar.py         # SEC EDGAR filings (free)
    ├── utils/
    │   ├── llm.py               # Multi-model routing (fast/smart tiers)
    │   └── display.py           # Terminal display helpers
    └── agents/
        ├── react_agent.py       # ReAct agent factory (core agentic logic)
        ├── debate.py            # Agreement check + debate node
        ├── risk_manager.py      # Position sizing & limits
        ├── portfolio_manager.py # Multi-turn agentic PM
        ├── value_agent.py       # (legacy deterministic, kept as reference)
        ├── growth_agent.py      # (legacy deterministic, kept as reference)
        ├── contrarian_agent.py  # (legacy deterministic, kept as reference)
        ├── technical_agent.py   # (legacy deterministic, kept as reference)
        ├── fundamental_agent.py # (legacy deterministic, kept as reference)
        └── sentiment_agent.py   # (legacy deterministic, kept as reference)
```

---

## Agentic Features In Depth

### 1. ReAct Agents (Autonomous Tool Use)

Each analyst agent is a ReAct loop that:
1. **THINK**: "What do I need to know about this stock?"
2. **ACT**: Calls a tool (`get_financial_metrics`, `get_prices`, `get_sec_financial_facts`, etc.)
3. **OBSERVE**: Reads the tool result
4. **REPEAT**: Decides if it needs more data (up to 4 tool calls)
5. **CONCLUDE**: Produces a bullish/neutral/bearish signal with reasoning

The agent **chooses which tools to call** — a Value Analyst might check P/E and FCF yield, while a Technical Analyst might check prices and volume. This is autonomous, not scripted.

### 2. Conditional Debate

After analysts run, the Agreement Check node measures consensus:
- **≥60% agree** → skip debate, go straight to Risk Manager
- **<60% agree** → trigger Debate node

The Debate node:
- Presents each analyst's signal AND reasoning to an LLM moderator
- Weights arguments by historical agent accuracy (from memory)
- Produces a synthesis with the key disagreement and strongest argument
- This synthesis is passed to the Portfolio Manager

### 3. Multi-Turn Portfolio Manager

The PM is itself a ReAct agent that can:
- Review all analyst signals + debate synthesis
- Call tools if it needs more data (`get_financial_metrics`, `get_company_news`, `get_sec_financial_facts`)
- Reason about conflicting signals across multiple turns
- Make a final decision only when it has enough information

### 4. Agent Memory (Adaptive)

SQLite-based persistent memory:
- Stores every decision (agent, ticker, signal, confidence, price)
- Tracks agent accuracy over time
- **Debate moderator** and **Portfolio Manager** weight signals by track record
- An agent that's been 90% accurate gets more weight than one at 40%

### 5. Multi-Model Routing

- **Analysts**: Use fast/cheap models (`llama-3.1-8b-instant` on Groq = free, fast)
- **Debate + PM**: Use smart models (`llama-3.3-70b-versatile` on Groq = free, capable)
- Keeps costs at $0 while maximizing decision quality where it matters

### 6. Rate Limit Safety

Designed for Groq free tier:
- Max 4 tool calls per analyst per ticker
- Max 3 tool calls for Portfolio Manager per ticker
- Debate only runs when needed (not every time)
- Fast models for analysts = lower token usage

---

## Deployment (All Free)

### Streamlit Community Cloud (Recommended)
1. Push to GitHub (public repo)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Set main file: `src/dashboard.py`
4. Add secrets: `GROQ_API_KEY = "your-key"`
5. Deploy — free forever

### Render (Free Tier)
1. Push to GitHub
2. New **Web Service** on [render.com](https://render.com)
3. Start command: `streamlit run src/dashboard.py --server.port $PORT --server.address 0.0.0.0`

### Locally
```bash
pip install -r requirements.txt
streamlit run src/dashboard.py
```

---

## Complete Cost: $0

| Component | Free Option |
|-----------|-------------|
| Stock data | yfinance |
| SEC filings | SEC EDGAR API (no key) |
| Agent memory | SQLite (local file) |
| LLM | Groq free tier / Ollama / Gemini free |
| Dashboard | Streamlit Cloud |
| **Total** | **$0** |

---

## License

MIT - Use freely for learning and research.
