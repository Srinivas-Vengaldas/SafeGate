# API, monitoring and deployment

## API endpoints
POST /v1/chat/completions is the OpenAI-compatible proxy, including streaming. POST /v1/check screens text, and optionally files, without calling an LLM and returns each rail's verdict. GET /v1/decisions returns the audit log with filters by application, verdict, blocking rail and time. GET /v1/stats returns dashboard aggregates, GET /v1/policy the active policy, GET /metrics Prometheus metrics and GET /healthz the health check. The /v1/rag endpoints add documents, load the sample knowledge base and answer questions. Every response carries an X-SafeGate-Request-Id header that matches its row in the decision log.

## Audit log and monitoring
Every decision is recorded with the application, endpoint, verdict, the rail that blocked or redacted, each rail's score and reason, and the screening time. The input itself is stored only as a SHA-256 hash, never as raw text. Prometheus metrics cover requests by verdict, verdicts by rail and stage, per-rail and total screening latency, cache hits and rate-limited requests, and a provisioned Grafana dashboard charts them.

## The demo page
The demo page has a playground where a visitor types a prompt or picks an example attack and sees each rail's verdict, score and reason, plus exactly what the LLM would receive. It can also screen a model reply with the output rails, screen attached images and files, and answer questions from documents with the guarded RAG pipeline. A live monitor shows request counts, blocks and redactions per rail, latency percentiles and recent audit log entries.

## Deployment
Every push to the main branch runs the tests, builds a Docker image with the latest trained classifier, starts it with a 448 MB memory limit and smoke-tests it, including a screenshot through OCR and a RAG query, and only then pushes it to GitHub Container Registry and redeploys the public demo on Render's free tier. The free tier has 512 MB of memory and a tenth of a CPU, and the service sleeps after 15 minutes without traffic, so the first visit after a quiet spell takes about a minute. Production deployments use PostgreSQL and Redis through Docker Compose.

## Roadmap
Planned next steps are a benchmark of poisoned retrieval, running SafeGate's classifier and ProtectAI's model together and blocking when either fires, a grounding check that verifies answers against their sources, and training data for indirect injection hidden in documents.
