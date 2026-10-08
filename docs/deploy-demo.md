# Deploying the public demo

The demo page (`/`), the API and the trained injection classifier run as one container on
Render's free tier (512 MB of memory, no payment details needed).

To fit in 512 MB, the demo image:

- serves the classifier as an int8 ONNX model on ONNX Runtime, with no PyTorch;
- keeps the weights in a page-aligned `model.onnx.data` file that ONNX Runtime memory-maps, so
  loading them does not briefly double the heap;
- scores at most 4 windows (the first and last) of very long prompts;
- uses a tokenizer-only spaCy pipeline for PII, which keeps every pattern-based entity
  (email, phone, SSN, card, IBAN, IP) but drops name detection.

The **Deploy demo** workflow builds this image on every push to `main` and after every
successful training run, checks that it starts and screens long prompts under a 448 MB limit,
leaving headroom under 512 MB, and only then pushes it to `ghcr.io/srinivas-vengaldas/safegate:latest`.

## One-time setup

1. **Make the image public.** After the first **Deploy demo** run, open the repo's main page,
   click **safegate** under *Packages* in the right sidebar, then **Package settings →
   Change visibility → Public**.
2. **Create the Render service.** Sign up at <https://render.com> with GitHub (free, no card).
   Choose **New → Web Service → Existing Image**, enter
   `ghcr.io/srinivas-vengaldas/safegate:latest`, pick the **Free** instance type, and set the
   health check path to `/healthz` under *Advanced*. Create it. The public URL appears at the top
   of the service page (`https://safegate-xxxx.onrender.com`).
3. **Turn on automatic redeploys.** In the Render service, open **Settings → Deploy Hook** and
   copy the URL. In this GitHub repo, open **Settings → Secrets and variables → Actions** and add
   it as the secret `RENDER_DEPLOY_HOOK`.

## Notes

- Free Render services sleep after 15 minutes without traffic; the next visit takes about a
  minute to wake. Open the link shortly before an interview.
- The decision log uses SQLite inside the container, so it resets when the service restarts.
  Production deployments use PostgreSQL via `docker-compose.yml`.
- No LLM key is configured, so the demo only screens prompts. Callers of
  `/v1/chat/completions` must send their own key, which is passed through and never stored.
