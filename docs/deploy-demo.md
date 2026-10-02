# Deploying the public demo

The demo page (`/`) and API run as one container. The free Hugging Face Spaces CPU tier is
enough: it serves the playground, the live monitor and `/docs` at a public URL.

## One-time setup

1. Create a free account at <https://huggingface.co> if you do not have one.
2. Create a new Space: **Spaces → Create new Space**, name it `safegate`, choose **Docker**
   as the SDK and the **Blank** template, hardware **CPU basic (free)**, visibility **Public**.
3. Create a token: **Settings → Access Tokens → Create new token**, type **Write**.
4. In this GitHub repo: **Settings → Secrets and variables → Actions**
   - Secrets tab: add `HF_TOKEN` with the token.
   - Variables tab: add `HF_SPACE` with `<your-hf-username>/safegate`.
5. Run the **Deploy demo** workflow from the Actions tab (or push to `main`).

The Space builds the Docker image and serves the app at
`https://<your-hf-username>-safegate.hf.space`. Every later push to `main` redeploys it.

## Notes

- The demo uses SQLite inside the container, so the decision log resets when the Space restarts.
  Production deployments use PostgreSQL via `docker-compose.yml`.
- No LLM key is configured, so the demo only screens prompts (`/v1/check`). Callers of
  `/v1/chat/completions` must send their own key, which is passed through and never stored.
- Free Spaces sleep after a period without traffic; the first visit afterwards takes a minute to wake.
