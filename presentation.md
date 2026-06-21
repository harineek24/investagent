# InvestAgent — 7-Slide Presentation
*Speaker notes included. Diagrams are plain-text boxes — easy to redraw in PowerPoint/Keynote/Google Slides.*

---

## Slide 1 — The Idea

### Visual
```
        💭  "What if 6 different investors,
              each with their own strategy,
              all analyzed the same stock —
              and an AI ran every one of them, for you?"

        ┌─────────────────────────────────────────┐
        │   InvestAgent                            │
        │   A free, AI-powered hedge fund          │
        │   simulator — no real money involved      │
        └─────────────────────────────────────────┘
```

### On-slide text
- A hedge fund = professionals who research stocks and decide buy / sell / hold to grow money.
- InvestAgent simulates a tiny version of that — entirely run by AI, entirely free, for learning.

### Speaker notes
Open with the hook question — don't define "hedge fund" first, let curiosity lead. Then land the one-liner: this is a sandbox where AI plays the role of a research team, not real trading advice. Say plainly: *no real money, $0 cost to run, educational.* This sets audience expectations before anything technical comes up.

---

## Slide 2 — Meet the 6 "Analysts"

### Visual
```
 🏛️ VALUE        📈 GROWTH       😬 CONTRARIAN
 "Is it cheap     "Is it growing   "Has everyone
  for what it's    fast?"           panicked too
  worth?"                            much?"

 📊 TECHNICAL    🧩 FUNDAMENTAL   📰 SENTIMENT
 "What does the   "Big-picture     "What is the
  price chart      health check     news/insiders/
  say?"            across the      Wall St. saying?"
                   board"
```

### On-slide text
- Same stock, 6 completely different instincts.
- None of them talk to each other yet — they each form an independent opinion first.

### Speaker notes
This is the most memorable slide — spend real time here. Use a relatable analogy: "imagine asking 6 friends with very different personalities whether to buy a stock — one only cares if it's a bargain, one only cares about hype/growth, one is a chart-watcher who's never read a balance sheet." Each "friend" is actually the same AI model, just given a different personality and a different way of looking at the same data. That reframing — same engine, different lens — sets up Slide 3 perfectly.

---

## Slide 3 — How One Analyst "Thinks"

### Visual
```
   THINK  →  ACT (look something up)  →  OBSERVE  →  repeat...  →  DECIDE
     │              │                       │                        │
  "What do      "Check the           "Okay, P/E is        "Bullish,
   I need to     P/E ratio"           low, that's           70%
   know?"                             a good sign"          confidence"
```

### On-slide text
- The AI doesn't run a fixed formula — it reasons step by step, like a person would.
- It chooses what to look up (price history, financials, news, insider trades...) and stops once it has enough evidence.
- Max 4 lookups per analyst, per stock — keeps it fast and free.

### Speaker notes
Demystify the "AI" part here without using the word "LLM" more than once. Compare directly to a human analyst: you'd open a stock app, check a number, maybe check another, then form an opinion — the AI does the same sequence of steps, just automatically and consistently. The key contrast to plant: earlier versions of projects like this just compute one number from a fixed formula. This one reasons in natural language and decides what data even matters for that specific case.

---

## Slide 4 — When Analysts Disagree (the "memory" of the system)

### Visual
```
 [Value] [Growth] [Contrarian] [Technical] [Fundamental] [Sentiment]
     │       │         │            │            │            │
     └───────┴─────────┴────────────┴────────────┴────────────┘
                            │
                    "How much do they agree?"
                            │
              ┌─────────────┴─────────────┐
              │                           │
        Mostly agree                Big disagreement
              │                           │
              │                      🗣️ DEBATE
              │                  (weigh arguments by
              │                   each analyst's past
              │                   track record)
              │                           │
              └─────────────┬─────────────┘
                             │
                     💰 Risk Manager
                 ("how much can we safely buy?")
                             │
                   🧭 Portfolio Manager
                  (final buy/sell/hold call)
```

### On-slide text
- The system doesn't run a fixed checklist — its path changes depending on what happens.
- If analysts mostly agree → go straight to a decision.
- If they clash → trigger a debate step that weighs each analyst by how often it's been right before.
- This is built with **LangGraph** — a framework for building flows that branch based on results, not just a straight line.

### Speaker notes — this is your technical centerpiece, take it slow
Explain LangGraph as "a flowchart that the program actually follows live, and can take different branches each run." Concretely walk through what's shared between every step — a single object called **state** that all 6 analysts and the later steps read from and write to. State is the one thing every box on this flowchart can see: which tickers we're analyzing, what each analyst said, whether a debate is needed, how much agreement there was, etc.

Say it like this: *"Imagine a clipboard that gets passed around the room. Each analyst writes their opinion on it. Before it reaches the final decision-maker, someone checks the clipboard — 'do these opinions agree?' If yes, skip to the manager. If no, pass it through a debate huddle first, then to the manager."* That clipboard is the **state**; the room layout (who passes to whom, and when) is the **graph**.

Concretely, the state holds things like: the list of stocks, each analyst's opinion so far, whether a debate was triggered, and how much agreement there was — and it accumulates as it flows through the system, rather than starting fresh at each step. That accumulation is what makes the debate/risk/portfolio-manager steps "aware" of everything that happened before them.

---

## Slide 5 — Built to Fail Safely (Agent Design Best Practices)

### Visual
```
   Layer 1: LLM call fails (rate limit, timeout, bad key)
        │
        └──→ 🛟 Data Fallback — quick rule-based signal from raw
              financial metrics + price data, instead of a 0%-confidence blank

   Layer 2: Agent loops past its iteration cap without concluding
        │
        └──→ 🛟 Neutral Signal — low-confidence "inconclusive," not a crash

   Layer 3: Portfolio Manager itself fails
        │
        └──→ 🛟 Majority Vote — confidence-weighted vote across the 6
              analyst signals stands in as the final call

        Plus, every analyst is hard-capped at 4 tool calls.
        The Portfolio Manager is capped at 3.
        No agent can loop forever and burn unlimited free-tier quota.
```

### On-slide text
- Bounded autonomy: agents reason freely, but inside hard limits (max tool calls per analyst, max steps for the Portfolio Manager).
- Three layers of fallback mean one bad API response never crashes the whole run — it degrades gracefully instead.
- This is a recognized pattern in agent design: give the AI room to reason, but always have a deterministic safety net underneath it.

### Speaker notes
This slide answers the question a skeptical audience member will ask: "what happens when the AI gets it wrong, or the API goes down?" Walk through the three layers in order, from most-graceful to most-basic: first try a quick rule-based read of the actual data (still useful, just not reasoned), then a plain "I don't know yet" neutral signal if an agent loops too long, and only as a last resort fall back to a simple vote-counting rule if the final decision-maker itself can't run. Tie it back to cost control too — the iteration caps exist because this runs on free LLM tiers, so unconstrained reasoning loops would be slow and could exhaust the free quota fast. The takeaway: "autonomous" doesn't mean "unsupervised" — every agent here has a leash.

---

## Slide 6 — Why This Is Different / Honest Limitations

### Visual
```
 ✅ $0 cost              ✅ Fully autonomous         ✅ Has a memory
 (free data + free AI)   (no hand-written rules)     (tracks which analysts
                                                        tend to be right)

 ⚠️  Educational simulation, not real financial advice
 ⚠️  Some pieces (e.g. accuracy tracking) are still being wired up fully
```

### On-slide text
- Free data: Yahoo Finance + SEC filings. Free AI: Groq/Gemini free tiers.
- No human pre-wrote "if P/E < 15, buy" — the reasoning is genuinely generated per stock.
- Future work: more automated testing, validating predictions over a longer time window.

### Speaker notes
Be upfront about the caveat slide rather than burying it — it builds credibility instead of undermining it. Frame limitations as "known next steps," not flaws: every real system has a roadmap. This is also the natural place to mention you've already added unit tests and fixed bugs as part of hardening it — shows ongoing diligence, not a one-and-done build.

---

## Slide 7 — Reusing This Elsewhere

### Visual
```
        InvestAgent's actual reusable pattern:

   "Multiple expert opinions"  →  "Check agreement"  →  "Escalate if conflict"  →  "Final decision"

        Swap the experts, keep the skeleton:

   ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │ Hiring panel     │   │ Medical diagnosis │   │ Content moderation│
   │ AI: 4 interviewer│   │ AI: 3 specialist   │   │ AI: multiple       │
   │ personas score a │   │ "opinions" on a    │   │ reviewers flag a   │
   │ candidate;        │   │ symptom; disagree  │   │ post; disagree =   │
   │ disagree = panel  │   │ = escalate to a    │   │ escalate to human  │
   │ discussion        │   │ senior review      │   │ moderator          │
   └─────────────────┘   └──────────────────┘   └──────────────────┘
```

### On-slide text
- The reusable idea isn't "stock picking" — it's: *multiple AI perspectives → measure agreement → escalate only when needed → one accountable final decision.*
- This pattern (LangGraph state + conditional branching) fits anywhere you want several "expert" viewpoints to inform one confident decision instead of trusting a single AI call.
- Other reuse angle: the **scoring/memory layer** — logging every decision and quietly grading it later — works for any system where you want to know "is this AI actually getting better, or just confident?"

### Speaker notes
This is your "so what" slide — it turns a stock-picker demo into a transferable architecture lesson. Land on two reusable pieces specifically: (1) the *branch-only-when-needed* pattern (cheap when everyone agrees, expensive reasoning only on genuine conflict — efficient by design), and (2) the *decision memory* pattern (every output gets logged with enough context to later check if it was right, building an accuracy track record over time). Both are valuable independent of investing — name-drop a domain the audience cares about (hiring, support tickets, content review, etc.) to make it concrete for them specifically.
