# Payment Planner

A standalone, read-only personal finance agent based on the earlier course
Payment Planner work and the Lesson 07 memory patterns.

## Project structure

| File | Purpose |
|---|---|
| `main.py` | CLI, model setup, chat loop, commands, and offline tests |
| `agent.py` | Agent assembly and dynamic profile injection |
| `instructions.py` | Role, workflow, memory policy, and safety boundaries |
| `tools.py` | Deterministic parsing, categorization, filtered retrieval, account summaries, monthly analytics, cash flow, comparisons, and recurring-pattern tools |
| `TOOLS_README.md` | Plain-language, step-by-step guide to every function in `tools.py` |
| `memory.py` | SQLite session, tiktoken counts, six-item window, compaction, structured profile, and idempotency |
| `long_term_memory.py` | P3 write routing, user-scoped Chroma recall, prompt fusion, topic replacement, and forgetting |
| `observability.py` | Local Agent trace tree, latency/error reporting, and optional Langfuse export |
| `.env.example` | OpenAI/Ollama and session configuration template |
| `requirements.txt` | Python dependencies |
| `setup.sh` | Reproducible Python environment setup |

## Setup

```bash
cd PaymentPlanner
bash setup.sh
source .venv/bin/activate
```

Then either add an OpenAI API key to `.env` or start Ollama:

```bash
ollama pull qwen2.5:7b
ollama serve
```

Do not put banking credentials in `.env` or transaction input.

## Run

```bash
python main.py
```

Useful chat commands:

- `/profile` — show structured non-sensitive preferences.
- `/memory` — show stored history size versus the context-window limit.
- `/compact` — force an OpenAI Responses compaction of old history.
- `/long-memory` — show the number of user-scoped durable preferences.
- `/forget TOPIC` — remove one long-term topic, such as `monthly_budget`.
- `/reset` — clear chat history while preserving structured preferences.
- `exit` — stop the application.

Use a separate session ID for a separate conversation:

```bash
python main.py --session-id payment-planner:alice:thread-02
```

## Tracing and Langfuse

Every normal chat run prints a small local trace tree covering the Agent run, tool
calls, memory retrieval/write decisions, errors, latency, and token usage. Run the
offline example without a model or API key:

```bash
python observability.py
```

Langfuse export is opt-in. Add `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and the
appropriate `LANGFUSE_BASE_URL` to `.env`, then run:

```bash
python main.py --langfuse
```

The Langfuse configuration hides prompt, response, and message contents by default.
It retains execution structure and operational measurements without intentionally
exporting raw transaction text or tool arguments.

## Offline verification

```bash
python main.py --selftest
python main.py --memory-demo
python -m unittest -v test_analytics_tools.py
```

The memory demo stores 16 session items in SQLite but sends only the latest six
history items plus the new input to the model, matching Lesson 07's
`04_sliding_window.py`. It prints both token counts using `tiktoken` and proves that
the same normalized input is processed once.

## Memory design

- `data/sessions.db` keeps conversation history for each stable session ID.
- The model receives only a sliding window of recent session items.
- OpenAI Responses compaction summarizes older conversation items when history grows;
  `/compact` can trigger it manually. Ollama chat-completions mode keeps the sliding
  window but disables Responses-only compaction.
- `data/planner_profile.json` stores only default currency, monthly budget, and savings
  goal. These facts are injected separately and survive trimming and compaction.
- `data/idempotency.db` hashes normalized input per session and reuses the first result
  when identical input is submitted again.
- `data/long_term_memory/` stores user-scoped durable preferences in Chroma. Stable
  topics replace older values when the user changes a preference. Raw transactions
  and sensitive inputs are not promoted.
- Credentials and transaction records are never written to the structured profile.
- Both local data files are ignored by Git.

## Safety boundary

This project analyzes user-provided transaction information. It does not create,
authorize, schedule, submit, or execute payments, and it must not request banking
credentials or authentication information.
