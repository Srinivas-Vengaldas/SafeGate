# SafeGate overview

## What SafeGate is
SafeGate is an open-source LLM guardrails gateway built by Srinivas Vengaldas (Das) as a portfolio project. It sits between an application and any LLM, speaks the OpenAI API, and screens traffic in both directions: the user's prompt before it reaches the model, and the model's reply before it reaches the user. Every decision is logged for audit. The source code is at github.com/Srinivas-Vengaldas/SafeGate under the MIT license, and a public demo runs at safegate-latest.onrender.com.

## Why it exists
Teams that put an LLM in front of users need to stop prompt injection and jailbreak attempts, keep personal data and secrets from leaking to a third-party model, and keep toxic or sensitive replies away from users. SafeGate does this in one place, as a drop-in proxy, instead of every application re-implementing its own checks. It also ships with an evaluation harness that measures how well the guardrails work, including how often they wrongly block legitimate users, because a guardrail that blocks everything is not useful.

## How an application uses it
An application switches to SafeGate by changing only the base URL of its OpenAI client to point at SafeGate's /v1 endpoint. Nothing else in the application changes. A blocked request comes back as an OpenAI-style 400 error whose type is safegate_blocked and whose code names the rail that blocked it, or, with the refuse setting, as a normal completion that explains the block with finish_reason content_filter.

## Tech stack
SafeGate is written in Python with FastAPI. It uses Microsoft Presidio for personal data detection, a fine-tuned DeBERTa-v3-small classifier served with ONNX Runtime for injection detection, Detoxify's toxicity model for replies, Tesseract OCR for images, PostgreSQL or SQLite for the decision log, Redis for rate limiting and caching, Prometheus and Grafana for monitoring, Docker for packaging, GitHub Actions for CI, training and deployment, and Render's free tier for the public demo.
