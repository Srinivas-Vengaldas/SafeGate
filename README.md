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

To host it publicly on Render's free tier, see [docs/deploy-demo.md](docs/deploy-demo.md).

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

## Injection classifier

Rules catch known phrasings; the classifier catches paraphrases. `microsoft/deberta-v3-small` is
fine-tuned on public datasets and served on CPU as the `injection` rail.

| Source | Role |
| --- | --- |
| `deepset/prompt-injections` | Injection vs. benign (training) |
| `jackhhao/jailbreak-classification` | Jailbreak vs. benign (training) |
| `databricks/databricks-dolly-15k` | Benign instructions, 2,500 sampled (training) |
| `sahil2801/CodeAlpaca-20k` | Benign coding instructions as hard negatives, 2,500 sampled (training, from v2) |
| `Lakera/gandalf_ignore_instructions` | Real attacks, **fully held out** to test generalization |
| `eval/tricky_benign.jsonl` | 150 hand-written safe prompts that look dangerous, to measure false positives |

Leakage controls: training-side prompts that nearly match the tricky-benign evaluation set are removed; exact and near-duplicate prompts (MinHash, Jaccard ≥ 0.8 on word 5-grams) are
collapsed before splitting, groups with conflicting labels are dropped, and held-out prompts that
nearly duplicate anything on the training side are removed. The blocking threshold is chosen on
the validation split, never on a test set.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -e ".[train]"
python -m training.build_dataset --out data
python -m training.train --data data --out models/injection-deberta
python -m training.evaluate --model models/injection-deberta --data data --out reports
python -m training.export_onnx --model models/injection-deberta --out models/injection-onnx
```

`export_onnx.py` writes an int8 ONNX model that the gateway serves on ONNX Runtime without
PyTorch (`pip install -e ".[onnx]"`); point a policy's `injection.model` at its directory.

The same pipeline runs on GitHub Actions (**Train injection classifier**, manual trigger) and
publishes metrics to the job summary. To enable the rail, add it to a policy after `rules`:

```yaml
  injection:
    model: models/injection-deberta   # or a Hugging Face Hub id
    threshold: 0.5                    # use the threshold reported by evaluate.py
```

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]" && python -m spacy download en_core_web_sm   # add ",ml" to test the classifier
pytest -q
ruff check . && ruff format --check .
uvicorn app.main:app --reload   # uses SQLite by default; set SAFEGATE_DATABASE_URL for Postgres
```

Configuration is read from `SAFEGATE_*` environment variables or `.env` (see `.env.example`).
Never commit `.env`.

## Roadmap

- [x] **Week 1:** streaming proxy, rules rail, Presidio PII rail, PostgreSQL decision log, Prometheus metrics, Docker compose, pytest, CI
- [x] **Week 2 (code):** dataset build with cross-source dedupe and a held-out source, DeBERTa fine-tuning, evaluation vs. rules baseline with PR curve, injection rail
- [x] **Week 2 (results v1):** trained and benchmarked; see Results
- [x] **Week 2 (results v2):** retrained with hard-negative benign data; tricky-benign false positives 18.7% → 4.7%
- [x] Demo page (playground + live monitor) and automatic public deploy (GHCR image, Render free tier)
- [x] ONNX int8 export served without PyTorch; demo fits in 512 MB
- [ ] **Week 3:** output rails (PII redaction, toxicity), Redis rate limiting and cache, garak scan of bare LLM vs. SafeGate, latency benchmark
- [ ] **Week 4:** AWS deployment, results table, architecture diagram, demo video; stretch: ONNX export, baseline comparison, NLI grounding check

## Results

### Injection classifier

DeBERTa-v3-small, 3 epochs on CPU (GitHub Actions). Full reports:
[v1](reports/v1/metrics.md), [v2](reports/v2/metrics.md).

| Metric | Rules only | v1 | **v2 (current)** |
| --- | ---: | ---: | ---: |
| Test recall | 2.5% | 100.0% | 91.2% |
| Test precision | 100.0% | 97.6% | 97.3% |
| Test false-positive rate | 0.0% | 0.6% | 0.3% |
| Held-out source recall (Gandalf, never seen in training) | 12.7% | 91.4% | 87.2% |
| **Tricky-benign false-positive rate** | 0.0% | 18.7% | **4.7%** |
| CPU latency per prompt, p50 / p95 | <1 ms | 71 / 379 ms | 79 / 431 ms |

Rules-only numbers are on the v2 test split; v1's rules-only recall was 1.2%.

What this shows:

- **Rules alone are not a defense.** The denylist caught 2 of 80 test attacks and 13% of the
  held-out Gandalf attacks.
- **The classifier generalizes across sources,** with recall dropping from 91% in-distribution to
  87% on a dataset it never saw. The held-out number is the honest one.
- **v1 over-blocked safe prompts that use attack-like words.** It flagged 28 of 150 hand-written
  safe prompts (for example "Please ignore case when comparing these two strings") at every
  threshold, because its training data had almost no benign prompts with words like *ignore*,
  *override* or shell commands.
- **v2 fixes most of that with hard negatives.** Adding 2,500 benign coding instructions
  (CodeAlpaca) cut tricky-benign false positives from 18.7% to 4.7%, at the cost of some recall.
  Most of the 7 prompts it still flags talk about ignoring or forgetting earlier text, or about a
  bot's instructions and rules, which is genuinely close to real attacks. Any training prompt that nearly matched the tricky-benign set
  was removed first, so that set still measures unseen prompts.
- **Long prompts are slow.** p95 latency comes from multi-window jailbreak prompts. The served
  ONNX model cuts it by about 40% and caps each prompt at its first and last windows.

Operating point: the threshold is chosen on validation data (0.47). Raising it trades recall for
fewer false positives:

| Threshold | Held-out recall | Tricky-benign FPR |
| ---: | ---: | ---: |
| 0.47 (default) | 87.2% | 4.7% |
| 0.90 | 83.8% | 2.0% |
| 0.99 | 78.2% | 0.7% |

**Served model (ONNX, int8).** The same v2 model exported to ONNX with int8 weights and scored as
the gateway serves it (at most 4 windows per prompt). Full report:
[reports/v2-onnx/metrics.md](reports/v2-onnx/metrics.md).

| Metric | PyTorch fp32 | ONNX int8 |
| --- | ---: | ---: |
| Held-out recall | 87.2% | 83.8% |
| Tricky-benign FPR | 4.7% | 3.3% |
| Test false-positive rate | 0.3% | 0.2% |
| CPU latency per prompt, p50 / p95 (same runner) | 65 / 343 ms | 43 / 209 ms |
| Weights on disk | about 540 MB | 172 MB |
| Serving image needs PyTorch | yes | no |

Quantization cuts latency by about a third (p50) to two fifths (p95). Its validation-chosen
threshold lands higher (0.975), so it blocks a little less: at 0.5 the int8 model reaches 89.9%
held-out recall with 5.3% tricky-benign FPR. The whole demo container (gateway, PII rail and
classifier) runs at about 460 MB, inside a 512 MB free-tier limit.

![Precision-recall curve on the test split](reports/v2/pr_curve.png)

garak attack-success rates and end-to-end gateway latency arrive in Week 3.

## License

MIT, see [LICENSE](LICENSE).
