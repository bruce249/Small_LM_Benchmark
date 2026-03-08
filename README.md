# 🏟️ Open Source Model Evaluation Arena

This project helps you compare Hugging Face models across summarization, question answering, coding, and reasoning tasks. It runs evaluation jobs, collects outputs, scores the results, and produces a leaderboard based on quality, latency, and cost.

---

## Architecture

```
User Request (API / Dashboard)
        │
        ▼
┌─────────────────────┐
│  ExperimentManager   │   Orchestrator – coordinates the full pipeline
└────────┬────────────┘
         │
    ┌────▼─────────────┐   1. Load dataset + generate prompts
    │ TaskGeneratorAgent│
    └────┬─────────────┘
         │  List[EvalTask]
    ┌────▼─────────────┐   2. Run prompts on N models (async parallel)
    │  ModelRunnerAgent │
    └────┬─────────────┘
         │  List[ModelOutput]
    ┌────▼─────────────┐   3. Score with ROUGE / BLEU / accuracy
    │   EvaluatorAgent  │
    └────┬─────────────┘
         │  List[EvalResult]
    ┌────▼─────────────┐   4. Aggregate → leaderboard
    │    ReportAgent    │
    └──────────────────┘
         │
         ▼
   ExperimentReport (JSON)
```

### Agent Communication

Agents are **stateless** and exchange typed dataclasses defined in `arena/schemas.py`. The `ExperimentManager` moves data from one stage to the next. Longer runs can be handed off to **Celery** workers through Redis.

---

## Project Structure

```
arena/
├── __init__.py              # Package metadata
├── config.py                # Centralized settings (env vars)
├── logging_config.py        # Structured logging
├── schemas.py               # Shared data-transfer objects
│
├── agents/
│   ├── task_generator.py    # Loads datasets, builds prompts
│   ├── model_runner.py      # Calls HF Inference API in parallel
│   ├── evaluator.py         # ROUGE, BLEU, accuracy scoring
│   └── report_agent.py      # Aggregates results → leaderboard
│
├── services/
│   ├── dataset_loader.py    # HuggingFace Datasets wrapper
│   ├── model_registry.py    # Model metadata registry
│   └── prompt_templates.py  # Prompt templates per task type
│
├── api/
│   ├── main.py              # FastAPI application & routes
│   └── schemas.py           # Pydantic request/response models
│
├── db/
│   ├── session.py           # Async SQLAlchemy engine
│   └── models.py            # ORM models (Experiment, EvalResult)
│
├── experiments/
│   └── experiment_manager.py # Pipeline orchestrator
│
└── worker/
    ├── __init__.py           # Celery app configuration
    └── tasks.py              # Celery task definitions

dashboard/
└── app.py                    # Streamlit interactive UI

examples/
├── run_evaluation.py         # Direct async evaluation script
└── run_via_api.py            # Evaluation via REST API
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env and set your HF_API_TOKEN
```

### 3. Run a quick evaluation (no server needed)

```bash
export HF_API_TOKEN="hf_..."
python -m examples.run_evaluation
```

### 4. Start the API server

```bash
uvicorn arena.api.main:app --reload --port 8000
```

The API docs are available at `http://localhost:8000/docs`.

### 5. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

### 6. (Optional) Start Celery worker

```bash
celery -A arena.worker worker --loglevel=info
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/models` | List registered models |
| `POST` | `/experiments` | Launch async experiment |
| `POST` | `/experiments/sync` | Run experiment synchronously |
| `GET` | `/experiments/{id}` | Get experiment results |
| `GET` | `/leaderboard/{id}` | Get leaderboard only |

### Example: Launch an experiment

```bash
curl -X POST http://localhost:8000/experiments/sync \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "summarization",
    "model_ids": [
      "mistralai/Mistral-7B-Instruct-v0.3",
      "meta-llama/Meta-Llama-3-8B-Instruct",
      "google/gemma-2-9b-it"
    ],
    "max_samples": 5
  }'
```

---

## Supported Task Types

| Task | Dataset (default) | Metrics |
|------|-------------------|---------|
| `summarization` | CNN/DailyMail | ROUGE-1, ROUGE-2, ROUGE-L, BLEU |
| `qa` | SQuAD | Exact Match, F1 |
| `coding` | HumanEval | BLEU, Token Overlap |
| `reasoning` | GSM8K | Numeric Accuracy |

---

## Configuration

Settings are loaded from environment variables with sensible defaults. See `arena/config.py` or `.env.example` for the full list.

Key settings:
- `HF_API_TOKEN` – HuggingFace API token (required for gated models)
- `DATABASE_URL` – PostgreSQL connection string
- `REDIS_URL` – Redis URL for Celery broker
- `MAX_CONCURRENT_MODELS` – Max parallel API calls (default: 5)
- `DEFAULT_TIMEOUT_SECONDS` – Per-request timeout (default: 120s)

---

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, SQLAlchemy (async), PostgreSQL
- **Task Queue**: Celery + Redis
- **Model Execution**: HuggingFace Inference API, httpx (async)
- **Evaluation**: rouge-score, NLTK (BLEU), custom accuracy metrics
- **Dashboard**: Streamlit
- **Data**: HuggingFace Datasets

---

## License

MIT
