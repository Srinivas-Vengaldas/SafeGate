# SafeGate

An OpenAI-compatible LLM guardrails gateway. SafeGate sits between any application and any LLM,
screens traffic in layers (cheap checks first, expensive ones last), logs every decision for audit,
and ships with an evaluation harness that measures how well it works, including how often it
wrongly blocks legitimate users.

> Status: **Week 1 of 4.** Proxy, rules rail, PII rail, decision log, metrics, Docker and CI are
> in. The fine-tuned injection classifier, output rails and benchmark come next (see Roadmap).

## Demo

Open `http://localhost:8000/` after `docker compose up`. The page has two panels:

- **Playground:** type a prompt (or pick an example attack) and see each rail's verdict, score and
  reason, plus exactly what the LLM would receive after redaction.
- **Live monitor:** request counts by verdict, blocks and redactions per rail, p50/p95 screening
  latency, and the most recent audit-log entries, refreshed every few seconds.

To host it publicly for free, see [docs/deploy-demo.md](docs/deploy-demo.md).

## How it works

```
client (any OpenAI SDK)
   │  base_url = http://safegate:8000/v1
   ▼
SafeGate ── input rails ──► rules (length, denylist, regex) ─► PII (Presidio) ─► …
   │            │ block → 400 safegate_blocked, LLM never called
   │            │ redact → redacted text is what the LLM sees
   ▼            ▼
LLM provider   decision log (PostgreSQL, input hashed, never stored raw) + Prometheus metrics
```

Each rail returns `allow`, `block` or `redact` with a score and a reason. A block stops the chain;
a redaction feeds the redacted text to every later rail and to the LLM.

## Quick start

```bash
cp .env.example .env          # set SAFEGATE_UPSTREAM_API_KEY, or leave it empty to pass callers' keys through
docker compose up --build
```

Point any OpenAI client at SafeGate by changing only the base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-...")
client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "Hi"}])
```

A blocked request raises `openai.BadRequestError` with `error.type == "safegate_blocked"` and
`error.code` naming the rail.

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI-compatible proxy, including `stream: true` passthrough |
| `POST /v1/check` | Screen text without calling an LLM; returns per-rail verdicts |
| `GET /v1/decisions` | Audit log; filters: `app`, `action`, `rail` (blocking rail), `since`, `until`, `limit`, `offset` |
| `GET /v1/stats` | Dashboard aggregates over recent decisions: counts by verdict and rail, p50/p95 latency |
| `GET /v1/policy` | The active policy (rails and their order) |
| `GET /` | Demo page: playground and live monitor |
| `GET /metrics` | Prometheus: requests by action, verdicts by rail, rail and screening latency |
| `GET /healthz` | Liveness; returns 503 until policies and models are loaded |

Every response carries an `X-SafeGate-Request-Id` header that matches the decision log row.

## Policies

Policies live in `policies/<app>.yaml`. Send `X-SafeGate-App: <app>` to select one; unknown apps
fall back to `policies/default.yaml`. Rails run in the order the file lists them.

```yaml
screen_roles: [user]          # add `tool` to screen tool outputs
input_rails:
  rules:
    max_chars: 8000
    denylist: [ignore previous instructions]
    patterns: ['reveal\s+your\s+system\s+prompt']
  pii:
    mode: redact              # or block
    threshold: 0.4
    entities: [EMAIL_ADDRESS, PHONE_NUMBER, US_SSN, CREDIT_CARD]
```

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]" && python -m spacy download en_core_web_sm
pytest -q
ruff check . && ruff format --check .
uvicorn app.main:app --reload   # uses SQLite by default; set SAFEGATE_DATABASE_URL for Postgres
```

Configuration is read from `SAFEGATE_*` environment variables or `.env` (see `.env.example`).
Never commit `.env`.

## Roadmap

- [x] **Week 1:** streaming proxy, rules rail, Presidio PII rail, PostgreSQL decision log, Prometheus metrics, Docker compose, pytest, CI
- [ ] **Week 2:** dataset build (dedupe across sources, hold out one full dataset), fine-tune `microsoft/deberta-v3-small`, evaluation with PR curve, injection rail
- [x] Demo page (playground + live monitor) and one-click public deploy to Hugging Face Spaces
- [ ] **Week 3:** output rails (PII redaction, toxicity), Redis rate limiting and cache, garak scan of bare LLM vs. SafeGate, latency benchmark
- [ ] **Week 4:** AWS deployment, results table, architecture diagram, demo video; stretch: ONNX export, baseline comparison, NLI grounding check

## Results

Benchmark numbers (precision, recall, cross-dataset recall, tricky-benign false-positive rate,
garak attack success rate, added p50/p95 latency) will be filled in from real runs in Weeks 2 and 3.

## License

MIT, see [LICENSE](LICENSE).
