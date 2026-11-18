# Personal Finance AI Agent

Payment Planner turns plain-language transactions into spending summaries, cash-flow calculations, recurring-charge estimates, and month-to-month comparisons. It groups amounts by currency. The agent runs 11 tools through the OpenAI Agents SDK. A single-owner FastAPI service provides the browser interface.

The [hosted app](https://payment-planner-45x8.onrender.com/) requires an owner token for chat. The [health endpoint](https://payment-planner-45x8.onrender.com/healthz) is public. In a live two-turn check, the agent categorized a synthetic $37.82 Uber ride as Transportation. When asked for the amount and merchant in the same conversation, it recalled both. To try the project yourself, use the local demo below.

## Try the transaction demo

Run these commands from the repository root with Python 3.11. Create a fresh environment and generate a private app sign-in token.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r PaymentPlannerDeployment/requirements-dev.txt
python - <<'PY'
from pathlib import Path
from secrets import token_urlsafe
folder = Path("PaymentPlannerDeployment")
with (folder / ".env").open("x") as f:
    template = (folder / ".env.example").read_text()
    f.write(template.replace("PAYMENT_PLANNER_ACCESS_TOKEN=\n",
                             f"PAYMENT_PLANNER_ACCESS_TOKEN={token_urlsafe(32)}\n"))
PY
python -m uvicorn PaymentPlannerDeployment.server:app --workers 1 --no-access-log
```

Open http://127.0.0.1:8000 and enter the app token from your local `PaymentPlannerDeployment/.env`. In the default fixture mode, try `2026-09-15 UBER *TRIP $23.47`. The deterministic tools extract the amount and identify transportation. This mode runs without an OpenAI key. More synthetic examples are in [sample_transactions.json](PaymentPlannerDeployment/sample_transactions.json).

## Run a conversation with the model

Set `PAYMENT_PLANNER_MODE=openai` and provide `OPENAI_API_KEY` in your local deployment environment. The default model is `gpt-4.1-mini`. Run `python PaymentPlannerDeployment/warm_caches.py` to download the tokenizer and embedding model, then restart the server. Use the same transaction and ask “What was the amount I just gave you?” to exercise conversation history.

SQLite stores turns under a thread ID. The interface creates a fresh thread on reload and displays the latest answer. API clients can resume a thread by retaining its ID. Chroma retrieves eligible owner preferences across threads. The app uses one shared owner token, so use synthetic data for demonstrations.

## Deploy on Render

Create a Blueprint from this repository using the root [render.yaml](render.yaml). Leave the root directory blank. The template starts a single free service in fixture mode, generates the app token, and warms model caches during build. The sign-in token is available in the service environment.

To enable the agent, configure the model settings described above in the Render service environment and redeploy. Verify the mode at `/healthz`, then use the two-turn example above to verify an actual model call. The free template stores SQLite and Chroma under `/tmp/payment-planner`. Render can discard these files when the instance sleeps, restarts, or redeploys. For history across these events, mount a persistent disk on a supported plan and point `PAYMENT_PLANNER_DATA_DIR` at it. [Render storage documentation](https://render.com/docs/disks)

## Code and tests

| Location | Contents |
| --- | --- |
| [PaymentPlanner](PaymentPlanner) | Agent instructions, analytics tools, CLI, memory and synthetic merchant examples |
| [PaymentPlannerDeployment](PaymentPlannerDeployment) | FastAPI endpoints, web interface, configuration, and deployment tests |
| [Tool guide](PaymentPlanner/TOOLS_README.md) | Inputs, calculations and examples for the transaction tools |

```sh
python -m unittest PaymentPlannerDeployment.test_deployment
python -m unittest discover -s PaymentPlanner -p 'test_*.py'
```

The deployment tests use synthetic inputs and include a scripted SDK model. A separate authenticated message to the hosted app checks the configured provider.
